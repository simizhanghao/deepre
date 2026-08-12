#!/usr/bin/env python3
"""Merge CUR base and supplemental captures with arm-count validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--supplement", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for p in (args.base, args.supplement) for x in p.read_text().splitlines() if x.strip()]
    counts = Counter((x["sample_id"], x["cur_forced_arm"]) for x in rows)
    dist = Counter(counts.values())
    if set(dist) - {4, 8} or len(counts) != 256:
        raise SystemExit(f"invalid merged counts: {dict(dist)} keys={len(counts)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    print(json.dumps({"gate": "CUR_MERGE_PASS", "rows": len(rows), "arm_count_distribution": dict(dist)}, indent=2))


if __name__ == "__main__":
    main()
