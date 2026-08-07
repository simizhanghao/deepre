"""Phase 2C: Direct-label HotpotQA train pool with Qwen2.5-3B (GPU).

Single GPU:
    CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python scripts/label_direct_train.py \
      --eval-file data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl \
      --max-samples 4000 --seed 42 --run-tag phase2c_direct

4-way data parallel (GPUs 4,5,6,7) — ~4x wall-clock:
    RUN_DIR=results/phase2c_direct_label_n4000_$(date +%Y%m%d_%H%M%S)_phase2c_direct
    mkdir -p "$RUN_DIR"
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$((4+i)) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python scripts/label_direct_train.py \
        --eval-file data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl \
        --max-samples 4000 --seed 42 \
        --num-shards 4 --shard-id $i \
        --run-dir "$RUN_DIR" &
    done
    wait
    python scripts/label_direct_train.py --merge-dir "$RUN_DIR"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import exact_match, token_f1  # noqa: E402

DEFAULT_MODEL_DIR = "/data1/hcc/.hf_home/Qwen2.5-3B-Instruct"
DIRECT_SYSTEM_PROMPT = (
    "You are a question answering assistant. "
    "Answer the question with a short answer only, no explanation."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct-label train pool for internal SFT.")
    p.add_argument("--eval-file", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_DIR)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results"))
    p.add_argument("--run-tag", type=str, default="phase2c_direct")
    p.add_argument("--run-dir", type=str, default=None, help="Shared output dir")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument(
        "--merge-dir",
        type=str,
        default=None,
        help="If set, only merge labels_shard*.jsonl in this dir and exit.",
    )
    return p.parse_args()


def load_eval_jsonl(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    return rows


def merge_shards(run_dir: Path) -> None:
    shard_files = sorted(run_dir.glob("labels_shard*.jsonl"))
    if not shard_files:
        raise SystemExit(f"no labels_shard*.jsonl in {run_dir}")
    rows: List[Dict[str, Any]] = []
    seen = set()
    for path in shard_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = row["sample_id"]
                if sid in seen:
                    continue
                seen.add(sid)
                rows.append(row)
    # keep stable order by sample_id for reproducibility of downstream sampling
    rows.sort(key=lambda r: r["sample_id"])
    out = run_dir / "labels.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_em = sum(1 for r in rows if r.get("direct_correct") or r.get("exact_match") == 1.0)
    summary = {
        "num_samples": len(rows),
        "num_direct_correct": n_em,
        "mean_em": round(n_em / max(len(rows), 1), 4),
        "labels_path": str(out),
        "shard_files": [str(p) for p in shard_files],
        "phase": "2C",
        "purpose": "internal_positive_labeling",
        "merged": True,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"merged {len(shard_files)} shards -> {out} n={len(rows)}")
    print(f"direct_correct={n_em}/{len(rows)} ({summary['mean_em']:.1%})")


def main() -> None:
    args = parse_args()

    if args.merge_dir:
        merge_path = Path(args.merge_dir)
        if not merge_path.is_absolute():
            merge_path = REPO_ROOT / merge_path
        merge_shards(merge_path)
        return

    if not args.eval_file:
        raise SystemExit("--eval-file is required unless --merge-dir is set")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("--shard-id must be in [0, num_shards)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    eval_path = Path(args.eval_file)
    if not eval_path.is_absolute():
        eval_path = REPO_ROOT / eval_path
    all_samples = load_eval_jsonl(eval_path, args.max_samples)
    if not all_samples:
        raise SystemExit(f"no samples in {eval_path}")

    # Data-parallel shard: sample i goes to shard (i % num_shards)
    samples = [
        s for i, s in enumerate(all_samples) if i % args.num_shards == args.shard_id
    ]

    bad = [s["sample_id"] for s in samples if "_train_" not in s["sample_id"]]
    if bad:
        raise SystemExit(
            f"refusing non-train sample_ids (n={len(bad)}), e.g. {bad[0]}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"phase2c_direct_label_n{args.max_samples}_{stamp}_{args.run_tag}"
        )
        run_dir = Path(args.output_dir)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        run_dir = run_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[label_direct] device={device} shard={args.shard_id}/{args.num_shards} "
        f"n_shard={len(samples)} n_pool={len(all_samples)} model={args.model_path}"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # One visible GPU per process (via CUDA_VISIBLE_DEVICES); do NOT shard model.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    labels_path = (
        run_dir / f"labels_shard{args.shard_id}.jsonl"
        if args.num_shards > 1
        else run_dir / "labels.jsonl"
    )
    n_em = 0
    t0 = time.time()
    with labels_path.open("w", encoding="utf-8") as fout:
        for i, sample in enumerate(samples):
            messages = [
                {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
                {"role": "user", "content": sample["question"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prompt_len = int(inputs["input_ids"].shape[-1])
            pred = tokenizer.decode(
                out[0, prompt_len:], skip_special_tokens=True
            ).strip()
            golds = sample.get("gold_answers") or []
            em = exact_match(pred, golds)
            f1 = token_f1(pred, golds)
            if em >= 1.0:
                n_em += 1
            row = {
                "sample_id": sample["sample_id"],
                "prediction": pred,
                "gold_answers": golds,
                "exact_match": em,
                "token_f1": f1,
                "direct_correct": bool(em >= 1.0),
                "prompt_tokens": prompt_len,
                "generated_tokens": int(out.shape[-1] - prompt_len),
                "shard_id": args.shard_id,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                print(
                    f"  [shard{args.shard_id} {i+1}/{len(samples)}] "
                    f"em_so_far={n_em/(i+1):.3f} elapsed={time.time()-t0:.0f}s"
                )

    meta = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "num_samples_shard": len(samples),
        "num_direct_correct_shard": n_em,
        "labels_path": str(labels_path),
        "eval_file": str(eval_path),
        "model_path": args.model_path,
        "seed": args.seed,
    }
    (run_dir / f"summary_shard{args.shard_id}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"wrote {labels_path}")
    if args.num_shards == 1:
        summary = {
            "num_samples": len(samples),
            "num_direct_correct": n_em,
            "mean_em": round(n_em / max(len(samples), 1), 4),
            "eval_file": str(eval_path),
            "model_path": args.model_path,
            "seed": args.seed,
            "labels_path": str(labels_path),
            "phase": "2C",
            "purpose": "internal_positive_labeling",
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            f"direct_correct={n_em}/{len(samples)} "
            f"({summary['mean_em']:.1%}) — use these for internal SFT"
        )
    else:
        print(
            f"shard {args.shard_id} done. After all shards finish, run:\n"
            f"  python scripts/label_direct_train.py --merge-dir {run_dir}"
        )


if __name__ == "__main__":
    main()
