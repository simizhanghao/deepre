#!/usr/bin/env python3
"""Routing Exploration — training-parity audit via real EcaSearchAgentLoop (SGLang).

Does NOT implement Mixed-action training. Orchestrates:
  prepare  → stratified 32-row parquet under results/16_.../parity_sglang_32x4/
  print-cmd → docker/eca-verl command (user runs; STEPS=1, lr=0, dump rollouts)
  aggregate → gate from dumped jsonl + boundary table

Protocol (must match Boundary Stage-II worker):
  EcaSearchAgentLoop · max_search_turns=2 · max_assistant_turns=6
  T=0.9 (main) / 1.3 (aux) · top_p=0.95 · N=4 · Candidate-BM25 :8001

Example:
  python scripts/audit_routing_parity_sglang.py --phase prepare --max-samples 32 --seed 42
  python scripts/audit_routing_parity_sglang.py --phase print-cmd --temperature 0.9
  # user runs printed docker block
  python scripts/audit_routing_parity_sglang.py --phase aggregate --temperature 0.9
  python scripts/audit_routing_parity_sglang.py --phase print-cmd --temperature 1.3
  python scripts/audit_routing_parity_sglang.py --phase aggregate --temperature 1.3
  python scripts/audit_routing_parity_sglang.py --phase summarize
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results/16_audit_routing_exploration/parity_sglang_32x4"
VALID = ("NoSearch", "NeedSearch", "Undetermined")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SGLang parity Routing Exploration audit")
    p.add_argument(
        "--phase",
        choices=("prepare", "print-cmd", "aggregate", "summarize"),
        required=True,
    )
    p.add_argument("--config", type=str, default="", help="optional JSON overrides")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--n-rollouts", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUT),
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
    p.add_argument("--container", type=str, default="eca-verl")
    p.add_argument(
        "--cuda-devices",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="CUDA_VISIBLE_DEVICES inside container (must be <= docker-bound GPUs)",
    )
    p.add_argument("--n-gpus", type=int, default=8)
    p.add_argument("--debug", action="store_true", help="max-samples<=8 for smoke")
    return p.parse_args()


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO / path


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
    m = raw["boundary"] if isinstance(raw, dict) and isinstance(raw.get("boundary"), dict) else raw
    out: Dict[str, str] = {}
    for k, v in m.items():
        if isinstance(v, dict):
            lab = str(v.get("boundary") or v.get("label") or "Undetermined")
        else:
            lab = str(v)
        out[str(k)] = lab if lab in VALID else "Undetermined"
    return out


def stratify_ids(labels: Dict[str, str], max_samples: int, seed: int) -> List[str]:
    by: Dict[str, List[str]] = defaultdict(list)
    for sid, lab in labels.items():
        by[lab].append(sid)
    rng = random.Random(seed)
    for lab in by:
        rng.shuffle(by[lab])
    out: List[str] = []
    idxs = {lab: 0 for lab in VALID}
    while len(out) < max_samples:
        progressed = False
        for lab in VALID:
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


def phase_prepare(args: argparse.Namespace) -> None:
    import pandas as pd

    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_n = 8 if args.debug and args.max_samples > 8 else args.max_samples
    labels = load_boundary(resolve(args.boundary_table))
    want = set(stratify_ids(labels, max_n, args.seed))
    df = pd.read_parquet(resolve(args.train_parquet))

    def sid_of(row) -> str:
        ei = row["extra_info"]
        if not isinstance(ei, dict):
            ei = dict(ei)
        return str(ei.get("sample_id"))

    mask = df.apply(lambda r: sid_of(r) in want, axis=1)
    sub = df.loc[mask].copy()
    # Preserve round-robin order roughly by reindex on want list
    sid_to_i = {s: i for i, s in enumerate(want)}
    sub["_ord"] = sub.apply(lambda r: sid_to_i.get(sid_of(r), 10**9), axis=1)
    sub = sub.sort_values("_ord").drop(columns=["_ord"])
    if len(sub) != len(want):
        raise SystemExit(f"parquet subset size={len(sub)} want={len(want)}")

    pq = out_dir / "train_parity_32.parquet"
    if args.debug:
        pq = out_dir / f"train_parity_smoke{len(sub)}.parquet"
    sub.to_parquet(pq, index=False)
    hist = Counter(labels[s] for s in want)
    meta = {
        "purpose": "routing_exploration_parity_prepare",
        "git_commit": git_commit(),
        "n_questions": len(sub),
        "n_rollouts": args.n_rollouts,
        "selected_ids": list(want),
        "boundary_hist": dict(hist),
        "parquet": str(pq),
        "protocol": {
            "agent_loop": "eca_search_agent",
            "max_search_turns": 2,
            "max_assistant_turns": 6,
            "top_p": 0.95,
            "backend": "sglang_verl",
        },
    }
    (out_dir / "prepare_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[parity] wrote {pq}", flush=True)


def _ws(host_path: Path) -> str:
    """Map host repo path → /workspace/deepresearch/..."""
    try:
        rel = host_path.resolve().relative_to(REPO.resolve())
        return f"/workspace/deepresearch/{rel.as_posix()}"
    except ValueError:
        return str(host_path)


def phase_print_cmd(args: argparse.Namespace) -> None:
    out_dir = resolve(args.output_dir)
    meta_path = out_dir / "prepare_meta.json"
    if not meta_path.is_file():
        raise SystemExit("run --phase prepare first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pq_host = Path(meta["parquet"])
    n_q = int(meta["n_questions"])
    n = int(args.n_rollouts)
    t = float(args.temperature)
    dump_host = out_dir / f"rollouts_t{t}.jsonl"
    dump_ws = _ws(dump_host)
    pq_ws = _ws(pq_host)
    out_ws = _ws(out_dir / f"ckpt_scratch_t{t}")
    log_host = out_dir / f"run_t{t}.log"

    # STEPS=1 + lr=0: still enters trainer once, but no weight change; dump via EcaSearchAgentLoop.
    cmd = f"""# === Training-parity Routing Exploration (T={t}) ===
# Requires: eca-verl up, Candidate-BM25 on :8001, GPU free on {args.cuda_devices}

cd /data1/hcc/deepresearch
mkdir -p "{out_dir}"
# avoid host Permission denied if dump was root-owned from a prior docker run
docker exec {args.container} bash -lc 'rm -f {dump_ws}' 2>/dev/null || rm -f "{dump_host}" || true

curl -sf http://127.0.0.1:8001/health || \\
  echo "START RETRIEVER first (see scripts/start_candidate_retrieval_server.py)"

docker start {args.container} >/dev/null

docker exec -e CUDA_VISIBLE_DEVICES={args.cuda_devices} {args.container} bash -lc '
set -euo pipefail
test -d /workspace/verl || {{ echo "MISSING /workspace/verl — recreate from eca-verl-pre8gpu"; exit 1; }}
export PYTHONPATH=/workspace/deepresearch:/workspace/verl
export ECA_BOUNDARY_TABLE=/workspace/deepresearch/outputs/rl/04_table_search_boundary/boundary_latest.json
export ECA_BOUNDARY_STRICT=1
export ECA_EVIDENCE_WEIGHT=0.5
export ECA_SEARCH_COST_WEIGHT=0.30
export ECA_PARITY_DUMP={dump_ws}
cd /workspace/deepresearch
rm -f {dump_ws}
STEPS=1 TOTAL_EPOCHS=1 BATCH={n_q} N={n} \\
  ROLLOUT_TEMP={t} ROLLOUT_TOP_P=0.95 ACTOR_LR=0 N_GPUS={args.n_gpus} \\
  TRAIN_FILE={pq_ws} VAL_FILE={pq_ws} \\
  MODEL_PATH=/workspace/deepresearch/outputs/rl/03_hf_evidence_step400 \\
  OUT_DIR={out_ws} EXPERIMENT_NAME=parity_routing_t{t} \\
  SAVE_FREQ=9999 RESUME_MODE=disable VAL_BEFORE_TRAIN=False \\
  GPU_MEM_UTIL=0.55 MICRO_BATCH=1 \\
  bash scripts/run_grpo_boundary.sh
' 2>&1 | tee "{log_host}"

echo "DUMP_LINES=$(wc -l < "{dump_host}" || echo 0)"
python scripts/audit_routing_parity_sglang.py --phase aggregate --temperature {t} --output-dir "{out_dir}"
"""
    cmd_path = out_dir / f"run_cmd_t{t}.sh"
    cmd_path.write_text(cmd, encoding="utf-8")
    print(cmd)
    print(f"\n# also saved: {cmd_path}", flush=True)


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


def normalize_action(route_first: str, search_count: int, used_internal: int) -> str:
    if route_first in ("search", "internal"):
        return route_first
    if route_first == "both":
        return "other"
    # fallback like grpo_metrics
    if search_count > 0:
        return "search"
    if used_internal:
        return "internal"
    return "other"


def phase_aggregate(args: argparse.Namespace) -> None:
    out_dir = resolve(args.output_dir)
    t = float(args.temperature)
    dump = out_dir / f"rollouts_t{t}.jsonl"
    if not dump.is_file():
        raise SystemExit(f"missing dump: {dump}")
    labels = load_boundary(resolve(args.boundary_table))
    rows: List[Dict[str, Any]] = []
    with dump.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = str(rec["sample_id"])
            lab = labels.get(sid, "Undetermined")
            action = normalize_action(
                str(rec.get("route_first") or "none"),
                int(rec.get("search_count") or 0),
                int(rec.get("used_internal") or 0),
            )
            rec["boundary"] = lab
            rec["action"] = action
            rec["temperature"] = t
            rows.append(rec)

    by_lab: Dict[str, Any] = {}
    for lab in VALID:
        lab_rows = [r for r in rows if r["boundary"] == lab]
        n = len(lab_rows)
        c = Counter(r["action"] for r in lab_rows)
        by_lab[lab] = {
            "n": n,
            "n_questions": len({r["sample_id"] for r in lab_rows}),
            "p_search": (c.get("search", 0) / n) if n else 0.0,
            "p_internal": (c.get("internal", 0) / n) if n else 0.0,
            "p_other": (c.get("other", 0) / n) if n else 0.0,
            "counts": dict(c),
            "first_action_entropy": entropy(c),
        }

    groups: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        groups[str(r["sample_id"])].append(r["action"])
    n_g = len(groups)
    n_mixed = sum(1 for acts in groups.values() if "search" in acts and "internal" in acts)
    # NoSearch-only mixed rate
    nos_groups = {
        sid: acts
        for sid, acts in groups.items()
        if labels.get(sid) == "NoSearch"
    }
    n_nos = len(nos_groups)
    n_mixed_nos = sum(
        1 for acts in nos_groups.values() if "search" in acts and "internal" in acts
    )

    block = {
        "temperature": t,
        "n_rollouts_dumped": len(rows),
        "by_boundary": by_lab,
        "mixed_action_group_rate": (n_mixed / n_g) if n_g else 0.0,
        "mixed_action_group_rate_NoSearch": (n_mixed_nos / n_nos) if n_nos else 0.0,
        "n_groups": n_g,
        "n_mixed_groups": n_mixed,
        "dump": str(dump),
        "backend": "sglang_eca_search_agent_loop",
    }
    (out_dir / f"aggregate_t{t}.json").write_text(
        json.dumps(block, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(block, indent=2), flush=True)


def gate_from_t09(block: Dict[str, Any]) -> Dict[str, Any]:
    nos = block.get("by_boundary", {}).get("NoSearch", {})
    need = block.get("by_boundary", {}).get("NeedSearch", {})
    p_int = float(nos.get("p_internal", 0.0))
    p_search_nos = float(nos.get("p_search", 0.0))
    p_search_need = float(need.get("p_search", 0.0))
    mixed = float(block.get("mixed_action_group_rate", 0.0))
    mixed_nos = float(block.get("mixed_action_group_rate_NoSearch", 0.0))
    ok = p_int > 0.0 and mixed > 0.0
    ideal = p_int >= 0.10
    need_ok = p_search_need > p_search_nos
    if ok:
        return {
            "gate": "TRAINING_PARITY_EXPLORATION_OK",
            "next": "mixed_action_grpo",
            "p_internal_NoSearch": p_int,
            "mixed_action_group_rate": mixed,
            "mixed_action_group_rate_NoSearch": mixed_nos,
            "p_search_NeedSearch": p_search_need,
            "p_search_NoSearch": p_search_nos,
            "need_search_gt_nosearch": need_ok,
            "ideal_p_internal_ge_0.10": ideal,
            "reason": "Real EcaSearchAgentLoop still produces internal on NoSearch with mixed groups",
        }
    return {
        "gate": "TRAINING_PARITY_EXPLORATION_FAIL",
        "next": "dual_arm_or_fix_rollout_mismatch",
        "p_internal_NoSearch": p_int,
        "mixed_action_group_rate": mixed,
        "mixed_action_group_rate_NoSearch": mixed_nos,
        "p_search_NeedSearch": p_search_need,
        "p_search_NoSearch": p_search_nos,
        "need_search_gt_nosearch": need_ok,
        "ideal_p_internal_ge_0.10": ideal,
        "reason": "Real worker path: internal≈0 / no mixed groups — do not Mixed-action yet",
    }


def phase_summarize(args: argparse.Namespace) -> None:
    out_dir = resolve(args.output_dir)
    by_t: Dict[str, Any] = {}
    for t in (0.9, 1.3):
        p = out_dir / f"aggregate_t{t}.json"
        if p.is_file():
            by_t[str(t)] = json.loads(p.read_text(encoding="utf-8"))
    if "0.9" not in by_t:
        raise SystemExit("need aggregate for T=0.9 first")
    gate = gate_from_t09(by_t["0.9"])
    summary = {
        "purpose": "routing_exploration_parity_sglang",
        "git_commit": git_commit(),
        "output_dir": str(out_dir),
        "protocol": {
            "agent_loop": "eca_search_agent / EcaSearchAgentLoop",
            "backend": "veRL + SGLang (eca-verl)",
            "max_search_turns": 2,
            "max_assistant_turns": 6,
            "n_rollouts": args.n_rollouts,
            "temperatures": sorted(float(x) for x in by_t.keys()),
            "actor_lr": 0.0,
            "steps": 1,
            "note": "STEPS=1 lr=0 still enters trainer once; trajectories from ECA_PARITY_DUMP",
        },
        "by_temperature": by_t,
        "gate": gate,
        "primary_temperature": 0.9,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate": gate, "temps": list(by_t.keys())}, indent=2), flush=True)
    print(f"[parity] wrote {out_dir / 'summary.json'}", flush=True)


def main() -> None:
    args = parse_args()
    if args.config:
        cfg = json.loads(resolve(args.config).read_text(encoding="utf-8"))
        for k, v in cfg.items():
            if hasattr(args, k) and v is not None:
                setattr(args, k, v)
    if args.phase == "prepare":
        phase_prepare(args)
    elif args.phase == "print-cmd":
        phase_print_cmd(args)
    elif args.phase == "aggregate":
        phase_aggregate(args)
    else:
        phase_summarize(args)


if __name__ == "__main__":
    main()
