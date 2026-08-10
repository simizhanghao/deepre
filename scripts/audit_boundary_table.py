#!/usr/bin/env python3
"""Audit 3D2b boundary table vs train parquet — coverage must be 100% for training."""

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

VALID = ("NoSearch", "NeedSearch", "Undetermined")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_table(path: Path) -> Dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("boundary"), dict):
        m = raw["boundary"]
    else:
        m = raw
    out: Dict[str, str] = {}
    for k, v in m.items():
        if isinstance(v, dict):
            out[str(k)] = str(v.get("boundary") or v.get("label"))
        else:
            out[str(k)] = str(v)
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
        default=str(REPO / "data/rl/train_smoke_128/train.parquet"),
    )
    ap.add_argument("--require-full-coverage", action="store_true", default=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    table_path = Path(args.table).resolve()
    if not table_path.is_file():
        raise SystemExit(f"missing table: {args.table}")

    labels = load_table(table_path)
    sids = parquet_sample_ids(Path(args.train_parquet))
    sid_set: Set[str] = set(sids)
    missing = sorted(sid_set - set(labels.keys()))
    extra = sorted(set(labels.keys()) - sid_set)
    covered = [labels[s] for s in sids if s in labels]
    bad = sorted({lab for lab in covered if lab not in VALID})
    hist = Counter(covered)
    digest = sha256_file(table_path)
    coverage = 1.0 - len(missing) / max(len(sid_set), 1)

    report: Dict[str, Any] = {
        "ok": len(missing) == 0 and not bad,
        "table_path": str(table_path),
        "sha256": digest,
        "num_train": len(sids),
        "num_table": len(labels),
        "missing_count": len(missing),
        "missing_sample_ids": missing[:20],
        "extra_count": len(extra),
        "coverage": coverage,
        "histogram": dict(hist),
        "invalid_labels": bad,
        "frac_NoSearch": hist.get("NoSearch", 0) / max(len(covered), 1),
        "frac_NeedSearch": hist.get("NeedSearch", 0) / max(len(covered), 1),
        "frac_Undetermined": hist.get("Undetermined", 0) / max(len(covered), 1),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")

    if args.require_full_coverage and missing:
        raise SystemExit(
            f"FAIL: boundary coverage incomplete missing={len(missing)} "
            f"coverage={coverage:.4f}"
        )
    if bad:
        raise SystemExit(f"FAIL: invalid boundary labels: {bad}")
    if not missing:
        print("[audit] PASS coverage=1.0", flush=True)


if __name__ == "__main__":
    main()
