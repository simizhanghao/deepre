#!/usr/bin/env python3
"""Build Phase 3D2b search-boundary table (SAAS-style dual probe).

Per question under current policy:
  - n search-disabled rollouts (agent_notool capability prompt)
  - n search-enabled rollouts  (full Candidate-BM25 agent loop)

Labels (δ=2 default, n=4):
  NoSearch      if n_d >= δ
  NeedSearch    if n_d == 0 and n_e > 0
  Undetermined  otherwise

Usage:
  CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/build_search_boundary_table.py \
      --model-path outputs/rl/hf_merged/grpo_sftv1_evidence_3c_step400 \
      --train-parquet data/rl/grpo_smoke_128/train.parquet \
      --contexts-index data/rl/grpo_smoke_128/contexts_index.jsonl \
      --n 4 --delta 2 --temperature 0.9
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

from src.agents.react_loop import RolloutConfig, run_search_agent_rollout  # noqa: E402
from src.eval.metrics import exact_match  # noqa: E402
from src.rl.rewards_3c import extract_answer  # noqa: E402
from src.sft.prototype_builder import AGENT_SYSTEM_PROMPT  # noqa: E402

# Same as 3D2-v0 capability probe — do NOT advertise <search>.
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
    p = argparse.ArgumentParser(description="Build 3D2b search-boundary table")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument(
        "--train-parquet",
        type=str,
        default=str(REPO / "data/rl/grpo_smoke_128/train.parquet"),
    )
    p.add_argument(
        "--contexts-index",
        type=str,
        default=str(REPO / "data/rl/grpo_smoke_128/contexts_index.jsonl"),
    )
    p.add_argument("--n", type=int, default=4, help="rollouts per arm")
    p.add_argument("--delta", type=int, default=2, help="NoSearch threshold on n_d")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens-disabled", type=int, default=128)
    p.add_argument("--max-new-tokens-enabled", type=int, default=512)
    p.add_argument("--max-search-turns", type=int, default=2)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--out", type=str, default="")
    p.add_argument(
        "--symlink-latest",
        type=str,
        default=str(REPO / "outputs/rl/boundary/boundary_latest.json"),
    )
    p.add_argument(
        "--skip-enabled",
        action="store_true",
        help="debug: only disabled arm (labels will be mostly Undetermined)",
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


def load_contexts_index(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row["sample_id"])
            out[sid] = row
    return out


def load_rows(
    parquet_path: Path, contexts: Dict[str, Dict[str, Any]], max_samples: int
) -> List[Dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    rows: List[Dict[str, Any]] = []
    missing_ctx = 0
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
        sid = str(sid)
        ctx_row = contexts.get(sid)
        if ctx_row is None:
            missing_ctx += 1
            ctxs: List[Any] = []
        else:
            ctxs = list(ctx_row.get("contexts") or [])
        sf = ei.get("supporting_facts")
        if sf is None:
            sf_list: List[Any] = []
        elif hasattr(sf, "tolist"):
            sf_list = list(sf.tolist())
        elif isinstance(sf, list):
            sf_list = sf
        else:
            sf_list = list(sf)
        rows.append(
            {
                "sample_id": sid,
                "question": str(q),
                "gold_answers": _golds_from_row(rm),
                "contexts": ctxs,
                "supporting_facts": sf_list,
            }
        )
        if max_samples and len(rows) >= max_samples:
            break
    if missing_ctx:
        print(f"[boundary] WARN missing contexts for {missing_ctx} samples", flush=True)
    return rows


def _rollout_seed(base: int, qi: int, arm: str, k: int) -> int:
    arm_off = 0 if arm == "disabled" else 10_000
    return int(base) + qi * 1009 + arm_off + k * 17


def label_boundary(n_d: int, n_e: int, delta: int) -> str:
    if n_d >= delta:
        return "NoSearch"
    if n_d == 0 and n_e > 0:
        return "NeedSearch"
    return "Undetermined"


def generate_disabled(
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
            seed = _rollout_seed(base_seed, qi, "disabled", k)
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
            ans = extract_answer(gen)
            if ans is None and "<search>" in (gen or "").lower():
                preds_k.append("")
            else:
                preds_k.append((ans if ans is not None else gen).strip())
        all_preds[qi] = preds_k
        if (qi + 1) % 8 == 0 or qi + 1 == len(questions):
            print(f"[boundary] disabled {qi+1}/{len(questions)}", flush=True)
    return all_preds


def generate_enabled(
    model,
    tokenizer,
    rows: Sequence[Dict[str, Any]],
    *,
    n: int,
    temperature: float,
    max_new_tokens: int,
    max_search_turns: int,
    top_k: int,
    base_seed: int,
) -> List[List[Dict[str, Any]]]:
    import torch

    cfg = RolloutConfig(
        top_k=top_k,
        max_search_turns=max_search_turns,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
    all_out: List[List[Dict[str, Any]]] = [[] for _ in rows]
    for qi, sample in enumerate(rows):
        arm: List[Dict[str, Any]] = []
        for k in range(n):
            seed = _rollout_seed(base_seed, qi, "enabled", k)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            result = run_search_agent_rollout(sample, model, tokenizer, cfg)
            em = float(result.metrics.get("exact_match") or 0.0)
            arm.append(
                {
                    "em": em,
                    "pred": result.metrics.get("pred")
                    or (
                        next(
                            (
                                s.content
                                for s in reversed(result.trace.steps)
                                if s.step_type == "answer"
                                and s.content != "[UNFINISHED]"
                            ),
                            "",
                        )
                    ),
                    "search_count": int(result.metrics.get("search_count") or 0),
                    "route_first": result.route_first,
                    "finished": bool(result.finished),
                    "seed": seed,
                }
            )
        all_out[qi] = arm
        if (qi + 1) % 4 == 0 or qi + 1 == len(rows):
            n_ok = sum(1 for x in arm if x["em"] >= 0.5)
            print(
                f"[boundary] enabled {qi+1}/{len(rows)} "
                f"last_n_e≈{n_ok}/{n} search≈{[x['search_count'] for x in arm]}",
                flush=True,
            )
    return all_out


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

    contexts = load_contexts_index(Path(args.contexts_index))
    rows = load_rows(Path(args.train_parquet), contexts, args.max_samples)
    print(
        f"[boundary] samples={len(rows)} n={args.n} δ={args.delta} "
        f"T={args.temperature} model={args.model_path}",
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
    disabled_preds = generate_disabled(
        model,
        tok,
        [r["question"] for r in rows],
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens_disabled,
        base_seed=args.seed,
    )

    if args.skip_enabled:
        enabled_meta: List[List[Dict[str, Any]]] = [
            [{"em": 0.0, "pred": "", "search_count": 0, "route_first": "none",
              "finished": False, "seed": 0}
             for _ in range(args.n)]
            for _ in rows
        ]
    else:
        enabled_meta = generate_enabled(
            model,
            tok,
            rows,
            n=args.n,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens_enabled,
            max_search_turns=args.max_search_turns,
            top_k=args.top_k,
            base_seed=args.seed,
        )

    table: Dict[str, Any] = {}
    hist: Counter = Counter()
    for qi, r in enumerate(rows):
        golds = r["gold_answers"]
        d_flags = [1 if exact_match(g, golds) else 0 for g in disabled_preds[qi]]
        e_flags = [1 if float(m.get("em") or 0) >= 0.5 else 0 for m in enabled_meta[qi]]
        # Prefer EM via exact_match on pred when available
        e_preds = [str(m.get("pred") or "") for m in enabled_meta[qi]]
        e_flags = [1 if exact_match(p, golds) else 0 for p in e_preds]
        n_d = int(sum(d_flags))
        n_e = int(sum(e_flags))
        lab = label_boundary(n_d, n_e, args.delta)
        # Nmin among correct enabled trajectories (SAAS); None if none correct
        nmin = None
        correct_ns = [
            int(m["search_count"])
            for m, ok in zip(enabled_meta[qi], e_flags)
            if ok
        ]
        if correct_ns:
            nmin = int(min(correct_ns))
        table[r["sample_id"]] = {
            "boundary": lab,
            "n_d": n_d,
            "n_e": n_e,
            "n": args.n,
            "delta": args.delta,
            "n_min": nmin,
            "disabled_preds": disabled_preds[qi],
            "enabled": enabled_meta[qi],
            "disabled_seeds": [
                _rollout_seed(args.seed, qi, "disabled", k) for k in range(args.n)
            ],
        }
        hist[lab] += 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else (
        REPO / "outputs/rl/boundary" / f"boundary_3c400_{stamp}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    n_no = hist.get("NoSearch", 0)
    n_need = hist.get("NeedSearch", 0)
    n_und = hist.get("Undetermined", 0)
    payload = {
        "phase": "3D2b",
        "protocol": "saas_dual_probe_search_boundary",
        "delta": args.delta,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_search_turns": args.max_search_turns,
        "model_path": str(args.model_path),
        "train_parquet": str(args.train_parquet),
        "contexts_index": str(args.contexts_index),
        "seed": args.seed,
        "num_samples": len(rows),
        "coverage": 1.0,
        "missing_count": 0,
        "histogram": dict(hist),
        "frac_NoSearch": n_no / max(len(rows), 1),
        "frac_NeedSearch": n_need / max(len(rows), 1),
        "frac_Undetermined": n_und / max(len(rows), 1),
        "mean_n_d": sum(v["n_d"] for v in table.values()) / max(len(table), 1),
        "mean_n_e": sum(v["n_e"] for v in table.values()) / max(len(table), 1),
        "elapsed_sec": round(time.time() - t0, 1),
        "capability_system_prompt": CAPABILITY_SYSTEM_PROMPT,
        "boundary": {k: v["boundary"] for k, v in table.items()},
        "detail": table,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    digest = file_sha256(out)
    payload["sha256"] = digest
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    digest2 = file_sha256(out)
    print(f"[boundary] wrote {out}")
    print(
        f"[boundary] hist={dict(hist)} "
        f"mean_n_d={payload['mean_n_d']:.2f} mean_n_e={payload['mean_n_e']:.2f} "
        f"sha256={digest2} elapsed={payload['elapsed_sec']}s",
        flush=True,
    )

    if args.symlink_latest:
        latest = Path(args.symlink_latest)
        latest.parent.mkdir(parents=True, exist_ok=True)
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        rel = os.path.relpath(out.resolve(), start=latest.parent.resolve())
        latest.symlink_to(rel)
        print(f"[boundary] symlink {latest} -> {rel}", flush=True)

    audit = {
        "table_path": str(out.resolve()),
        "sha256": digest2,
        "num_samples": len(rows),
        "coverage": 1.0,
        "histogram": dict(hist),
        "delta": args.delta,
        "n": args.n,
        "model_path": str(args.model_path),
    }
    (out.parent / (out.stem + ".audit.json")).write_text(
        json.dumps(audit, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
