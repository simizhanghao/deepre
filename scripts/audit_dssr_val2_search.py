#!/usr/bin/env python3
"""Audit DSSR Val2 Search N=4 capture without any candidate evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    frozen_ids = [x for x in args.ids.read_text().splitlines() if x]
    rows = [json.loads(x) for x in args.input.read_text().splitlines() if x.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    errors = []
    if len(frozen_ids) != 128 or len(rows) != 512 or set(grouped) != set(frozen_ids):
        errors.append(f"IDs/rows mismatch: frozen={len(frozen_ids)} groups={len(grouped)} rows={len(rows)}")
    for sample_id in frozen_ids:
        items = grouped.get(sample_id, [])
        if Counter(str(x["cur_forced_arm"]) for x in items) != {"search": 4}:
            errors.append(f"{sample_id}: expected Search x4")
        if len({x["canonical_prompt_sha256"] for x in items}) != 1:
            errors.append(f"{sample_id}: canonical prompt hashes differ")
    summary = {
        "gate": "DSSR_VAL2_SEARCH_CAPTURE_PASS" if not errors else "DSSR_VAL2_SEARCH_CAPTURE_FAIL",
        "questions": len(grouped),
        "rows": len(rows),
        "rollouts_per_question": 4,
        "action_valid_rate": sum(int(x["cur_forced_action_valid"]) for x in rows) / max(1, len(rows)),
        "policy_failures_retained": sum(int(x["cur_policy_failure"]) for x in rows),
        "finish_rate": sum(int(x["finish"]) for x in rows) / max(1, len(rows)),
        "mean_search_f1": sum(float(x["answer_f1"]) for x in rows) / max(1, len(rows)),
        "errors": errors[:20],
        "error_count": len(errors),
        "test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
