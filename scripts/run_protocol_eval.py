"""Phase 2D3-C: protocol evaluation with the SFT agent system prompt.

Modes:
  evidence_oracle      — Question + Oracle docs → <evidence>/<think>/<answer>
  evidence_candidate   — Question + BM25 Top-K docs → same
  routing              — Question only → <internal>… or <search>…

Usage (deepresearch env):
  CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/run_protocol_eval.py \
    --mode evidence_oracle \
    --model-path outputs/sft_qwen25_3b_coldstart_v0_merged \
    --eval-file data/eval/hotpotqa_200.jsonl --max-samples 200 \
    --run-tag phase2d3c
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.protocol import (  # noqa: E402
    agent_system_prompt,
    score_evidence_use,
    score_routing,
)
from src.sft.prototype_builder import (  # noqa: E402
    format_documents_for_user,
    oracle_documents,
)

DEFAULT_MODEL = str(REPO_ROOT / "outputs" / "sft_qwen25_3b_coldstart_v0_merged")
MODES = ("evidence_oracle", "evidence_candidate", "routing")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Protocol eval for cold-start SFT.")
    p.add_argument("--mode", type=str, required=True, choices=MODES)
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--eval-file", type=str, default="data/eval/hotpotqa_200.jsonl")
    p.add_argument(
        "--retrieval-cache",
        type=str,
        default=(
            "results/retrieval_candidate_bm25_n200_20260807_154802/"
            "retrieval_results.jsonl"
        ),
    )
    p.add_argument(
        "--base-direct-metrics",
        type=str,
        default=(
            "results/baseline_direct_n200_20260807_154900_phase1_final_n200/"
            "metrics.json"
        ),
        help="For routing conditional rates (Base Direct correctness).",
    )
    p.add_argument(
        "--base-oracle-metrics",
        type=str,
        default=(
            "results/baseline_oracle_n200_20260807_154912_phase1_final_n200/"
            "metrics.json"
        ),
        help="For routing: Direct❌ Oracle✅ subset.",
    )
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results"))
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-tag", type=str, default="phase2d3c")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_jsonl(path: Path, max_samples: int) -> List[Dict[str, Any]]:
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


def load_retrieval_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["sample_id"]] = row
    return out


def load_em_map(path: Path) -> Dict[str, bool]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        r["sample_id"]: float(r.get("metrics", {}).get("exact_match", 0.0)) == 1.0
        for r in rows
    }


def documents_from_cache(cache_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for d in cache_row.get("documents") or []:
        docs.append(
            {
                "document_id": d["document_id"],
                "title": d.get("title", ""),
                "text": d.get("text", ""),
                "rank": d.get("rank"),
                "score": d.get("score"),
                "metadata": dict(d.get("metadata") or {}),
            }
        )
    return docs


def build_messages(mode: str, sample: Dict[str, Any], docs: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    system = agent_system_prompt()
    if mode == "routing":
        user = f"Question: {sample['question']}"
    else:
        assert docs is not None
        user = (
            f"Question: {sample['question']}\n\n"
            f"Documents:\n{format_documents_for_user(docs)}"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def git_commit_short() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    args = parse_args()
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    eval_path = resolve(args.eval_file)
    samples = load_jsonl(eval_path, args.max_samples)
    if not samples:
        raise SystemExit(f"no samples in {eval_path}")

    retrieval_cache: Dict[str, Dict[str, Any]] = {}
    if args.mode == "evidence_candidate":
        cache_path = resolve(args.retrieval_cache)
        retrieval_cache = load_retrieval_cache(cache_path)
        missing = [s["sample_id"] for s in samples if s["sample_id"] not in retrieval_cache]
        if missing:
            raise SystemExit(f"cache missing {len(missing)} ids e.g. {missing[0]}")

    base_direct_ok = load_em_map(resolve(args.base_direct_metrics))
    base_oracle_ok = load_em_map(resolve(args.base_oracle_metrics))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"protocol_{args.mode}_n{len(samples)}_{stamp}_{args.run_tag}"
    run_dir = resolve(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve(args.model_path)
    print(f"[protocol] mode={args.mode} device={device}")
    print(f"[protocol] model={model_path}")
    print(f"[protocol] run_dir={run_dir}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    print(f"[protocol] model loaded in {time.perf_counter() - t0:.1f}s")

    rows: List[Dict[str, Any]] = []
    with (run_dir / "generations.jsonl").open("w", encoding="utf-8") as gen_f:
        for i, sample in enumerate(samples, 1):
            sid = sample["sample_id"]
            docs: Optional[List[Dict[str, Any]]] = None
            if args.mode == "evidence_oracle":
                docs = oracle_documents(sample)
            elif args.mode == "evidence_candidate":
                docs = documents_from_cache(retrieval_cache[sid])

            messages = build_messages(args.mode, sample, docs)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_tokens = int(inputs["input_ids"].shape[-1])

            t_gen = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            latency_ms = (time.perf_counter() - t_gen) * 1000
            gen_ids = out[0][inputs["input_ids"].shape[-1] :]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_tokens = int(gen_ids.shape[-1])

            if args.mode == "routing":
                scored = score_routing(text, sample)
            else:
                scored = score_evidence_use(text, sample)

            row = {
                "sample_id": sid,
                "mode": args.mode,
                "generation": text,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": gen_tokens,
                "latency_ms": round(latency_ms, 1),
                "base_direct_correct": bool(base_direct_ok.get(sid, False)),
                "base_oracle_correct": bool(base_oracle_ok.get(sid, False)),
                **{k: v for k, v in scored.items() if k != "pred_evidence_refs"},
                "n_input_docs": len(docs) if docs is not None else 0,
            }
            # keep refs for offline debug
            if "pred_evidence_refs" in scored:
                row["pred_evidence_refs"] = scored["pred_evidence_refs"]
            rows.append(row)
            gen_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(samples):
                print(
                    f"[{args.mode}] {i}/{len(samples)} {sid} "
                    f"valid={row.get('protocol_valid')} "
                    f"EM={row.get('exact_match')} "
                    f"evF1={row.get('evidence_f1', '-')}"
                )

    n = len(rows)
    summary: Dict[str, Any] = {
        "mode": args.mode,
        "num_samples": n,
        "model_path": str(model_path),
        "git_commit": git_commit_short(),
        "max_new_tokens": args.max_new_tokens,
        "protocol_valid_rate": round(mean([float(r["protocol_valid"]) for r in rows]), 4),
        "answer_tag_rate": round(mean([float(r["answer_tag"]) for r in rows]), 4),
        "mean_em": round(mean([float(r["exact_match"]) for r in rows]), 4),
        "mean_token_f1": round(mean([float(r["token_f1"]) for r in rows]), 4),
        "mean_prompt_tokens": round(mean([float(r["prompt_tokens"]) for r in rows]), 1),
        "mean_generated_tokens": round(
            mean([float(r["generated_tokens"]) for r in rows]), 1
        ),
        "mean_latency_ms": round(mean([float(r["latency_ms"]) for r in rows]), 1),
    }

    if args.mode.startswith("evidence_"):
        summary.update(
            {
                "evidence_tag_rate": round(
                    mean([float(r["evidence_tag"]) for r in rows]), 4
                ),
                "think_tag_rate": round(mean([float(r["think_tag"]) for r in rows]), 4),
                "mean_evidence_precision": round(
                    mean([float(r["evidence_precision"]) for r in rows]), 4
                ),
                "mean_evidence_recall": round(
                    mean([float(r["evidence_recall"]) for r in rows]), 4
                ),
                "mean_evidence_f1": round(
                    mean([float(r["evidence_f1"]) for r in rows]), 4
                ),
            }
        )
    else:
        internal_rate = mean([1.0 if r["route"] == "internal" else 0.0 for r in rows])
        search_rate = mean([1.0 if r["route"] == "search" else 0.0 for r in rows])
        none_rate = mean([1.0 if r["route"] == "none" else 0.0 for r in rows])
        summary.update(
            {
                "internal_rate": round(internal_rate, 4),
                "search_rate": round(search_rate, 4),
                "none_rate": round(none_rate, 4),
            }
        )

        def _subset_rate(pred, subset_ids: List[str]) -> Dict[str, float]:
            sub = [r for r in rows if r["sample_id"] in subset_ids]
            if not sub:
                return {"n": 0}
            return {
                "n": len(sub),
                "internal_rate": round(
                    mean([1.0 if r["route"] == "internal" else 0.0 for r in sub]), 4
                ),
                "search_rate": round(
                    mean([1.0 if r["route"] == "search" else 0.0 for r in sub]), 4
                ),
                "protocol_valid_rate": round(
                    mean([float(r["protocol_valid"]) for r in sub]), 4
                ),
            }

        direct_correct = [s["sample_id"] for s in samples if base_direct_ok.get(s["sample_id"])]
        need_search = [
            s["sample_id"]
            for s in samples
            if (not base_direct_ok.get(s["sample_id"], False))
            and base_oracle_ok.get(s["sample_id"], False)
        ]
        summary["routing_conditional"] = {
            "base_direct_correct": _subset_rate(rows, direct_correct),
            "base_direct_wrong_oracle_correct": _subset_rate(rows, need_search),
        }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[protocol] artifacts -> {run_dir}")


if __name__ == "__main__":
    main()
