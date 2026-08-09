#!/usr/bin/env python3
"""Build / refresh Phase 3D2 p_int(q) capability table (tool-free, n rollouts).

p_int(q) = (# tool-free EM-correct rollouts) / n
         ∈ {0, 0.25, 0.5, 0.75, 1.0} for n=4.

Prompt: same core Agent system + tools DISABLED (not plain Direct QA).
Sampling: T=0.9, n=4, per-rollout reproducible seeds.

Usage:
  CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/build_capability_pint_table.py \
      --model-path outputs/sft_qwen25_3b_coldstart_v1_merged \
      --train-parquet data/rl/grpo_smoke_128/train.parquet \
      --n 4 --temperature 0.9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.metrics import exact_match  # noqa: E402
from src.rl.rewards_3c import extract_answer  # noqa: E402
# Same agent identity / answer protocol, but tools unavailable.
# Do NOT paste the full AGENT_SYSTEM_PROMPT verbatim — it advertises <search>
# and SFT-v1 then almost always emits search (collapsing p_int→0).
CAPABILITY_SYSTEM_PROMPT = (
    "You are an evidence-cost-aware research agent. "
    "CAPABILITY PROBE — TOOLS DISABLED. "
    "External search/retrieval is unavailable for this probe. "
    "Allowed tags only: <internal>, <think>, <answer>. "
    "Do NOT emit <search>, <observation>, or document <evidence>. "
    "Answer from internal knowledge only. "
    "You must finish with <answer>...</answer> containing a short final answer."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 3D2 tool-free p_int table")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument(
        "--train-parquet",
        type=str,
        default=str(REPO / "data/rl/grpo_smoke_128/train.parquet"),
    )
    p.add_argument("--n", type=int, default=4, help="tool-free rollouts per question")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=42, help="base seed; rollout k uses seed+…")
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--out", type=str, default="")
    p.add_argument(
        "--symlink-latest",
        type=str,
        default=str(REPO / "outputs/rl/capability/p_int_latest.json"),
    )
    p.add_argument(
        "--prompt-mode",
        choices=("agent_notool",),
        default="agent_notool",
    )
    return p.parse_args()


def _golds_from_row(rm: Dict[str, Any]) -> List[str]:
    gt = rm.get("ground_truth")
    if isinstance(gt, dict):
        t = gt.get("target", gt.get("gold_answers"))
        if isinstance(t, list):
            return [str(x) for x in t]
        if t is not None:
            return [str(t)]
    if isinstance(gt, list):
        return [str(x) for x in gt]
    if gt is not None:
        return [str(gt)]
    return []


def load_rows(parquet_path: Path, max_samples: int) -> List[Dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        ei = r["extra_info"]
        if not isinstance(ei, dict):
            ei = dict(ei)
        rm = r["reward_model"]
        if not isinstance(rm, dict):
            rm = dict(rm)
        q = ei.get("question") or ""
        if not q:
            prompt = r["prompt"]
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            for turn in reversed(list(prompt or [])):
                if isinstance(turn, dict) and turn.get("role") == "user":
                    q = str(turn.get("content") or "")
                    break
        sid = ei.get("sample_id")
        if sid is None or str(sid).strip() == "":
            raise SystemExit(f"missing sample_id in parquet row index={len(rows)}")
        rows.append(
            {
                "sample_id": str(sid),
                "question": str(q),
                "gold_answers": _golds_from_row(rm),
            }
        )
        if max_samples and len(rows) >= max_samples:
            break
    return rows


def _rollout_seed(base: int, qi: int, k: int) -> int:
    # Stable, sparse mapping so (qi,k) don't collide across refreshes.
    return int(base) + qi * 1009 + k * 17


def generate_answers(
    model,
    tokenizer,
    questions: Sequence[str],
    *,
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    base_seed: int,
) -> List[List[str]]:
    import torch

    all_preds: List[List[str]] = [[] for _ in questions]
    for qi, q in enumerate(questions):
        messages = [
            {"role": "system", "content": CAPABILITY_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        preds_k: List[str] = []
        for k in range(n):
            seed = _rollout_seed(base_seed, qi, k)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[-1]
            gen = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
            # Prefer <answer>; if model still emits <search>, count as failed probe.
            ans = extract_answer(gen)
            if ans is None and "<search>" in (gen or "").lower():
                preds_k.append("")  # incorrect
            else:
                preds_k.append((ans if ans is not None else gen).strip())
        all_preds[qi] = preds_k
        if (qi + 1) % 16 == 0 or qi + 1 == len(questions):
            print(f"[pint] generated {qi+1}/{len(questions)}", flush=True)
    return all_preds


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(Path(args.train_parquet), args.max_samples)
    print(
        f"[pint] samples={len(rows)} n={args.n} T={args.temperature} "
        f"prompt={args.prompt_mode} model={args.model_path}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    t0 = time.time()
    preds = generate_answers(
        model,
        tok,
        [r["question"] for r in rows],
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        base_seed=args.seed,
    )

    table: Dict[str, Any] = {}
    n_correct_total = 0
    hist: Counter = Counter()
    for qi, (r, gens) in enumerate(zip(rows, preds)):
        golds = r["gold_answers"]
        flags = [1.0 if exact_match(g, golds) else 0.0 for g in gens]
        k = int(sum(flags))
        p_int = round(k / float(args.n) * args.n) / float(args.n)
        table[r["sample_id"]] = {
            "p_int": p_int,
            "n_correct": k,
            "n": args.n,
            "preds": gens,
            "seeds": [_rollout_seed(args.seed, qi, kk) for kk in range(args.n)],
        }
        n_correct_total += k
        hist[f"{p_int:.2f}"] += 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else (
        REPO / "outputs/rl/capability" / f"p_int_agent_notool_{stamp}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "3D2",
        "protocol": "tool_free_periodic_capability",
        "prompt_mode": args.prompt_mode,
        "system_prompt": CAPABILITY_SYSTEM_PROMPT,
        "model_path": str(args.model_path),
        "train_parquet": str(args.train_parquet),
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "num_samples": len(rows),
        "coverage": 1.0,
        "missing_count": 0,
        "mean_p_int": sum(v["p_int"] for v in table.values()) / max(len(table), 1),
        "mean_em_rate": n_correct_total / max(len(rows) * args.n, 1),
        "histogram": dict(sorted(hist.items())),
        "elapsed_sec": round(time.time() - t0, 1),
        "p_int": {k: v["p_int"] for k, v in table.items()},
        "detail": table,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    digest = file_sha256(out)
    payload["sha256"] = digest
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    # recompute sha after embedding sha field (document both)
    digest2 = file_sha256(out)
    print(f"[pint] wrote {out}")
    print(
        f"[pint] mean_p_int={payload['mean_p_int']:.3f} "
        f"mean_em_rate={payload['mean_em_rate']:.3f} "
        f"hist={payload['histogram']} "
        f"sha256={digest2} elapsed={payload['elapsed_sec']}s"
    )

    if args.symlink_latest:
        latest = Path(args.symlink_latest)
        latest.parent.mkdir(parents=True, exist_ok=True)
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        # Relative symlink so Docker bind-mounts resolve inside the container.
        rel = os.path.relpath(out.resolve(), start=latest.parent.resolve())
        latest.symlink_to(rel)
        print(f"[pint] symlink {latest} -> {rel}")

    # sidecar audit always
    audit = {
        "table_path": str(out.resolve()),
        "sha256": digest2,
        "num_samples": len(rows),
        "coverage": 1.0,
        "missing_count": 0,
        "mean_p_int": payload["mean_p_int"],
        "histogram": payload["histogram"],
        "prompt_mode": args.prompt_mode,
        "model_path": str(args.model_path),
    }
    (out.parent / (out.stem + ".audit.json")).write_text(
        json.dumps(audit, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
