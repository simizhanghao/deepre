#!/usr/bin/env python3
"""Build additional four-rollout/arm data for CUR N=4 borderline questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", type=Path, required=True)
    ap.add_argument("--outcomes", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from datasets import Dataset

    borderline = {
        x["sample_id"] for x in map(json.loads, args.outcomes.read_text().splitlines())
        if x["high_confidence_direction"] == "borderline"
    }
    source = Dataset.from_parquet(str(args.paired))
    rows = [row for row in source if str(row["extra_info"]["sample_id"]) in borderline]
    if len(rows) != 2 * len(borderline):
        raise SystemExit(f"rows={len(rows)} borderline={len(borderline)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(args.output))
    summary = {
        "gate": "CUR_N8_DATA_PASS",
        "questions": len(borderline),
        "rows": len(rows),
        "additional_rollouts_per_arm": 4,
        "target_total_rollouts_per_arm": 8,
        "selection": "N=4 bootstrap delta-F1 CI95 crosses zero",
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
