#!/usr/bin/env python3
"""Freeze a deterministic 8 NeedSearch + 8 NoSearch Root-Pivot batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.rl.rewards_boundary import _load_boundary_table


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--boundary", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--per-class", type=int, default=8)
    p.add_argument("--subset", choices=("all", "need", "no"), default="all")
    args = p.parse_args()

    source, boundary = Path(args.source), Path(args.boundary)
    out, manifest = Path(args.output), Path(args.manifest)
    frame = pd.read_parquet(source)
    table, meta = _load_boundary_table(str(boundary))

    rows = {"NeedSearch": [], "NoSearch": []}
    for idx, row in frame.iterrows():
        extra = row["extra_info"]
        sample_id = str(extra.get("sample_id"))
        label = table.get(sample_id)
        if label in rows and len(rows[label]) < args.per_class:
            rows[label].append(idx)
    for label, indices in rows.items():
        if len(indices) != args.per_class:
            raise SystemExit(f"need {args.per_class} {label}, found {len(indices)}")

    if args.subset == "all":
        # Interleave classes so every contiguous two-question slice is balanced.
        selected = [i for pair in zip(rows["NeedSearch"], rows["NoSearch"], strict=True) for i in pair]
    elif args.subset == "need":
        selected = rows["NeedSearch"]
    else:
        selected = rows["NoSearch"]
    frozen = frame.loc[selected].reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_parquet(out, index=False)
    ids = [str(x.get("sample_id")) for x in frozen["extra_info"]]
    payload = {
        "gate": "ROOT_PIVOT_BALANCED_DATA_PASS",
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "boundary": meta,
        "per_class": args.per_class,
        "subset": args.subset,
        "n": len(frozen),
        "order": [
            {"sample_id": sid, "boundary": table[sid]}
            for sid in ids
        ],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
