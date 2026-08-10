#!/usr/bin/env python3
"""Routing Exploration Audit — tools-enabled first-action sampling (no training).

Answers whether Evidence@400 (+ boundary labels) still has spontaneous
``internal`` mass, or first-action ``search`` is near-deterministic.

Metrics (per temperature, stratified by frozen boundary label):
  - P(search|label), P(internal|label)
  - mixed_action_group_rate  (group has both search & internal first-actions)
  - first_action_entropy

Example (smoke):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    python scripts/audit_routing_exploration.py \\
      --model-path outputs/rl/03_hf_evidence_step400 \\
      --boundary-table outputs/rl/04_table_search_boundary/boundary_latest.json \\
      --output-dir results/16_audit_routing_exploration \\
      --max-samples 8 --n-rollouts 4 --temperatures 0.9,1.1,1.3 \\
      --seed 42 --debug
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agents.react_loop import RolloutConfig, run_search_agent_rollout  # noqa: E402

VALID_LABELS = ("NoSearch", "NeedSearch", "Undetermined")
ACTIONS = ("search", "internal", "other")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Routing Exploration Audit (no training).")
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional JSON with overrides (temperatures, n_rollouts, ...).",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=str(REPO / "outputs/rl/03_hf_evidence_step400"),
    )
    p.add_argument(
        "--boundary-table",
        type=str,
        default=str(REPO / "outputs/rl/04_table_search_boundary/boundary_latest.json"),
    )
    p.add_argument(
        "--train-parquet",
        type=str,
        default=str(REPO / "data/rl/train_smoke_128/train.parquet"),
    )
    p.add_argument(
        "--contexts-index",
        type=str,
        default=str(REPO / "data/rl/train_smoke_128/contexts_index.jsonl"),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO / "results/16_audit_routing_exploration"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=32, help="0 = all with labels")
    p.add_argument("--n-rollouts", type=int, default=4, help="rollouts per question × T")
    p.add_argument(
        "--temperatures",
        type=str,
        default="0.9,1.1,1.3",
        help="comma-separated temperatures",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--debug",
        action="store_true",
        help="Shrink defaults: max-samples<=8, n-rollouts<=2 if still defaulted large.",
    )
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(REPO),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_boundary(path: Path) -> Dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("boundary"), dict):
        m = raw["boundary"]
    else:
        m = raw
    out: Dict[str, str] = {}
    for k, v in m.items():
        if isinstance(v, dict):
            lab = str(v.get("boundary") or v.get("label") or "Undetermined")
        else:
            lab = str(v)
        if lab not in VALID_LABELS:
            lab = "Undetermined"
        out[str(k)] = lab
    return out


def load_contexts_index(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["sample_id"])] = row
    return out


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


def load_rows(
    parquet_path: Path, contexts: Dict[str, Dict[str, Any]], max_samples: int
) -> List[Dict[str, Any]]:
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
            raise SystemExit(f"missing sample_id at row={len(rows)}")
        sid = str(sid)
        ctx_row = contexts.get(sid)
        ctxs = list(ctx_row.get("contexts") or []) if ctx_row else []
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
    return rows


def stratify_select(
    rows: Sequence[Dict[str, Any]],
    labels: Dict[str, str],
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Prefer balanced NoSearch / NeedSearch / Undetermined when capping."""
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        lab = labels.get(str(r["sample_id"]))
        if lab is None:
            continue
        rr = dict(r)
        rr["boundary"] = lab
        by[lab].append(rr)
    rng = random.Random(seed)
    for lab in by:
        rng.shuffle(by[lab])
    if max_samples <= 0:
        out: List[Dict[str, Any]] = []
        for lab in VALID_LABELS:
            out.extend(by.get(lab, []))
        return out
    # Round-robin pull for balance.
    out = []
    idxs = {lab: 0 for lab in VALID_LABELS}
    while len(out) < max_samples:
        progressed = False
        for lab in VALID_LABELS:
            i = idxs[lab]
            bucket = by.get(lab, [])
            if i < len(bucket):
                out.append(bucket[i])
                idxs[lab] = i + 1
                progressed = True
                if len(out) >= max_samples:
                    break
        if not progressed:
            break
    return out


def normalize_route(route_first: str) -> str:
    if route_first == "search":
        return "search"
    if route_first == "internal":
        return "internal"
    # both / none / answer-first → other for exploration mass
    return "other"


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p + 1e-12, 2)
    return float(h)


def apply_config_json(args: argparse.Namespace) -> None:
    if not args.config:
        return
    path = resolve(args.config)
    if not path.is_file():
        raise SystemExit(f"missing --config: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit("config must be a JSON object")
    mapping = {
        "model_path": "model_path",
        "boundary_table": "boundary_table",
        "train_parquet": "train_parquet",
        "contexts_index": "contexts_index",
        "output_dir": "output_dir",
        "temperatures": "temperatures",
        "seed": "seed",
        "max_samples": "max_samples",
        "n_rollouts": "n_rollouts",
        "top_k": "top_k",
        "max_new_tokens": "max_new_tokens",
    }
    for k, attr in mapping.items():
        if k in cfg and cfg[k] is not None:
            setattr(args, attr, cfg[k])


def recommend_gate(by_t: Dict[str, Any]) -> Dict[str, Any]:
    """Gate on NoSearch internal mass at highest T."""
    temps = sorted(by_t.keys(), key=float)
    if not temps:
        return {"gate": "UNKNOWN", "reason": "no temperatures"}
    t_hi = temps[-1]
    block = by_t[t_hi].get("by_boundary", {}).get("NoSearch", {})
    p_int = float(block.get("p_internal", 0.0))
    mixed = float(by_t[t_hi].get("mixed_action_group_rate", 0.0))
    if p_int >= 0.05 and mixed > 0.0:
        return {
            "gate": "NATURAL_EXPLORATION_OK",
            "next": "mixed_action_grpo",
            "t": float(t_hi),
            "p_internal_NoSearch": p_int,
            "mixed_action_group_rate": mixed,
            "reason": "NoSearch still yields internal≥5% with mixed groups at high T",
        }
    if p_int < 0.02:
        return {
            "gate": "INTERNAL_SATURATED",
            "next": "dual_arm_grpo",
            "t": float(t_hi),
            "p_internal_NoSearch": p_int,
            "mixed_action_group_rate": mixed,
            "reason": "internal≈0 even at high T → root sampling saturated",
        }
    return {
        "gate": "MARGINAL",
        "next": "mixed_action_grpo_or_dual_arm",
        "t": float(t_hi),
        "p_internal_NoSearch": p_int,
        "mixed_action_group_rate": mixed,
        "reason": "weak but nonzero internal; prefer mixed-group before dual-arm",
    }


def main() -> None:
    args = parse_args()
    apply_config_json(args)
    if args.debug:
        if args.max_samples <= 0 or args.max_samples > 8:
            args.max_samples = 8
        if args.n_rollouts > 2:
            args.n_rollouts = 2

    temps = [float(x.strip()) for x in str(args.temperatures).split(",") if x.strip()]
    if not temps:
        raise SystemExit("empty --temperatures")

    model_path = resolve(args.model_path)
    table_path = resolve(args.boundary_table)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.is_dir():
        raise SystemExit(f"missing model: {model_path}")
    if not table_path.is_file():
        raise SystemExit(f"missing boundary table: {table_path}")

    labels = load_boundary(table_path)
    contexts = load_contexts_index(resolve(args.contexts_index))
    # Load all rows then stratify (max_samples applied in stratify).
    all_rows = load_rows(resolve(args.train_parquet), contexts, max_samples=0)
    samples = stratify_select(all_rows, labels, args.max_samples, args.seed)
    if not samples:
        raise SystemExit("no samples with boundary labels")

    hist_sel = Counter(s["boundary"] for s in samples)
    print(
        f"[routing_explore] model={model_path} n={len(samples)} "
        f"n_rollouts={args.n_rollouts} T={temps} hist={dict(hist_sel)}",
        flush=True,
    )
    print(f"[routing_explore] out={out_dir}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()

    # first-action focus: single agent round; stop before multi-hop search loop.
    traces_path = out_dir / "rollouts.jsonl"
    t0 = time.time()
    records: List[Dict[str, Any]] = []

    with traces_path.open("w", encoding="utf-8") as tf:
        for ti, T in enumerate(temps):
            for qi, sample in enumerate(samples):
                group_actions: List[str] = []
                for k in range(args.n_rollouts):
                    seed = (
                        int(args.seed)
                        + ti * 100_003
                        + qi * 1009
                        + k * 17
                    )
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    cfg = RolloutConfig(
                        top_k=args.top_k,
                        max_search_turns=1,
                        max_new_tokens=args.max_new_tokens,
                        max_rounds=1,
                        temperature=T,
                    )
                    result = run_search_agent_rollout(sample, model, tokenizer, cfg)
                    action = normalize_route(result.route_first)
                    group_actions.append(action)
                    rec = {
                        "sample_id": sample["sample_id"],
                        "boundary": sample["boundary"],
                        "temperature": T,
                        "rollout_k": k,
                        "seed": seed,
                        "route_first": result.route_first,
                        "action": action,
                        "finished": result.finished,
                        "search_count": int(result.metrics.get("search_count") or 0),
                    }
                    records.append(rec)
                    tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    tf.flush()
                print(
                    f"[T={T}] {qi+1}/{len(samples)} {sample['sample_id']} "
                    f"{sample['boundary']} actions={group_actions}",
                    flush=True,
                )

    # Aggregate
    by_t: Dict[str, Any] = {}
    for T in temps:
        t_recs = [r for r in records if float(r["temperature"]) == float(T)]
        by_lab: Dict[str, Any] = {}
        for lab in VALID_LABELS:
            lab_recs = [r for r in t_recs if r["boundary"] == lab]
            n = len(lab_recs)
            c = Counter(r["action"] for r in lab_recs)
            by_lab[lab] = {
                "n": n,
                "n_questions": len({r["sample_id"] for r in lab_recs}),
                "p_search": (c.get("search", 0) / n) if n else 0.0,
                "p_internal": (c.get("internal", 0) / n) if n else 0.0,
                "p_other": (c.get("other", 0) / n) if n else 0.0,
                "counts": dict(c),
                "first_action_entropy": entropy(c),
            }
        # mixed-action groups: group by sample_id
        groups: Dict[str, List[str]] = defaultdict(list)
        for r in t_recs:
            groups[str(r["sample_id"])].append(r["action"])
        n_g = len(groups)
        n_mixed = 0
        for acts in groups.values():
            s = set(acts)
            if "search" in s and "internal" in s:
                n_mixed += 1
        global_c = Counter(r["action"] for r in t_recs)
        by_t[str(T)] = {
            "by_boundary": by_lab,
            "mixed_action_group_rate": (n_mixed / n_g) if n_g else 0.0,
            "n_groups": n_g,
            "n_mixed_groups": n_mixed,
            "global_counts": dict(global_c),
            "global_first_action_entropy": entropy(global_c),
            "p_search": (global_c.get("search", 0) / max(len(t_recs), 1)),
            "p_internal": (global_c.get("internal", 0) / max(len(t_recs), 1)),
        }

    gate = recommend_gate(by_t)
    summary: Dict[str, Any] = {
        "purpose": "routing_exploration_audit",
        "git_commit": git_commit(),
        "model_path": str(model_path),
        "boundary_table": str(table_path),
        "train_parquet": str(resolve(args.train_parquet)),
        "seed": args.seed,
        "max_samples": args.max_samples,
        "n_rollouts": args.n_rollouts,
        "temperatures": temps,
        "debug": bool(args.debug),
        "selected_boundary_hist": dict(hist_sel),
        "num_questions": len(samples),
        "num_rollouts_total": len(records),
        "by_temperature": by_t,
        "gate": gate,
        "elapsed_seconds": round(time.time() - t0, 2),
        "output_dir": str(out_dir),
        "protocol": {
            "tools": "enabled",
            "max_rounds": 1,
            "note": "first-action focused; acceptance still on spontaneous routing",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Compact CSV for quick glance
    lines = [
        "temperature,boundary,n,p_search,p_internal,p_other,entropy,"
        "mixed_action_group_rate"
    ]
    for T in temps:
        block = by_t[str(T)]
        mixed = block["mixed_action_group_rate"]
        for lab in VALID_LABELS:
            b = block["by_boundary"][lab]
            lines.append(
                f"{T},{lab},{b['n']},{b['p_search']:.4f},{b['p_internal']:.4f},"
                f"{b['p_other']:.4f},{b['first_action_entropy']:.4f},{mixed:.4f}"
            )
    (out_dir / "by_temperature_boundary.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(json.dumps({"gate": gate, "by_temperature": by_t}, indent=2), flush=True)
    print(f"[routing_explore] wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
