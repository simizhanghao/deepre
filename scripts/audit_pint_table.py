#!/usr/bin/env python3
"""Audit 3D2 p_int table vs train parquet — coverage must be 100% for training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_table(path: Path) -> Dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("p_int"), dict):
        m = raw["p_int"]
    else:
        m = raw
    out: Dict[str, float] = {}
    for k, v in m.items():
        if isinstance(v, dict):
            out[str(k)] = float(v["p_int"])
        else:
            out[str(k)] = float(v)
    return out


def parquet_sample_ids(path: Path) -> List[str]:
    import pandas as pd

    df = pd.read_parquet(path)
    ids: List[str] = []
    for _, r in df.iterrows():
        ei = r["extra_info"]
        if not isinstance(ei, dict):
            ei = dict(ei)
        sid = ei.get("sample_id")
        if sid is None or str(sid).strip() == "":
            raise SystemExit("parquet contains empty sample_id")
        ids.append(str(sid))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=str, required=True)
    ap.add_argument(
        "--train-parquet",
        type=str,
        default=str(REPO / "data/rl/grpo_smoke_128/train.parquet"),
    )
    ap.add_argument(
        "--direct-labels",
        type=str,
        default="",
        help="optional Phase2 Direct labels.jsonl for sanity overlap",
    )
    ap.add_argument("--require-full-coverage", action="store_true", default=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    table_path = Path(args.table).resolve()
    if table_path.is_symlink():
        table_path = table_path.resolve()
    if not table_path.is_file():
        raise SystemExit(f"missing table: {args.table}")

    pint = load_table(table_path)
    sids = parquet_sample_ids(Path(args.train_parquet))
    sid_set: Set[str] = set(sids)
    missing = sorted(sid_set - set(pint.keys()))
    extra = sorted(set(pint.keys()) - sid_set)
    covered = [pint[s] for s in sids if s in pint]
    hist = Counter(f"{p:.2f}" for p in covered)
    digest = sha256_file(table_path)
    coverage = 1.0 - len(missing) / max(len(sid_set), 1)

    sanity: Dict[str, Any] = {}
    if args.direct_labels:
        dpath = Path(args.direct_labels)
        if dpath.is_file():
            direct_ok = {}
            with dpath.open() as f:
                for line in f:
                    row = json.loads(line)
                    sid = str(row.get("sample_id"))
                    ok = bool(row.get("direct_correct") or row.get("exact_match") == 1.0)
                    direct_ok[sid] = ok
            both = [s for s in sids if s in direct_ok and s in pint]
            if both:
                # high p_int should correlate with Direct✓ more often
                high = [s for s in both if pint[s] >= 0.75]
                low = [s for s in both if pint[s] <= 0.25]
                sanity = {
                    "n_overlap": len(both),
                    "direct_rate_high_pint": (
                        sum(direct_ok[s] for s in high) / len(high) if high else None
                    ),
                    "direct_rate_low_pint": (
                        sum(direct_ok[s] for s in low) / len(low) if low else None
                    ),
                    "note": "sanity only; Direct labels are NOT reward labels",
                }

    report = {
        "ok": len(missing) == 0,
        "table_path": str(table_path),
        "sha256": digest,
        "num_train": len(sids),
        "num_table": len(pint),
        "missing_count": len(missing),
        "missing_sample_ids": missing[:20],
        "extra_count": len(extra),
        "coverage": coverage,
        "mean_p_int": sum(covered) / max(len(covered), 1),
        "histogram": dict(sorted(hist.items())),
        "phase2_direct_sanity": sanity,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")

    if args.require_full_coverage and missing:
        raise SystemExit(
            f"FAIL: p_int coverage incomplete missing={len(missing)} coverage={coverage:.4f}"
        )
    if not missing:
        print("[audit] PASS coverage=1.0", flush=True)


if __name__ == "__main__":
    main()
