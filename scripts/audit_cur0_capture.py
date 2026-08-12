#!/usr/bin/env python3
"""Hard identity/action audit for CUR-0 paired captures."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--questions", type=int, required=True)
    ap.add_argument("--rollouts", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.input.read_text().splitlines() if x.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample_id"]].append(row)
    expected_rows = args.questions * 2 * args.rollouts
    errors = []
    if len(rows) != expected_rows:
        errors.append(f"rows={len(rows)} expected={expected_rows}")
    if len(groups) != args.questions:
        errors.append(f"questions={len(groups)} expected={args.questions}")
    for sid, items in groups.items():
        counts = Counter(x["cur_forced_arm"] for x in items)
        if counts != {"search": args.rollouts, "internal": args.rollouts}:
            errors.append(f"{sid}: arms={dict(counts)}")
        hashes = {x["canonical_prompt_sha256"] for x in items}
        lengths = {x["canonical_prompt_len"] for x in items}
        if len(hashes) != 1 or "" in hashes or len(lengths) != 1:
            errors.append(f"{sid}: canonical prompt identity failed")
    action_valid_rate = sum(int(x["cur_forced_action_valid"]) for x in rows) / max(1, len(rows))
    internal_tool_violations = sum(
        int(x["search_count"] > 0 or x["cur_forbidden_search_attempts"] > 0)
        for x in rows if x["cur_forced_arm"] == "internal"
    )
    summary = {
        "gate": "CUR_CAPTURE_PASS" if not errors else "CUR_CAPTURE_FAIL",
        "questions": len(groups),
        "rows": len(rows),
        "rollouts_per_arm": args.rollouts,
        "canonical_prompt_pairing": not any("canonical prompt" in x for x in errors),
        "action_valid_rate": action_valid_rate,
        "policy_failures_retained": sum(int(x["cur_policy_failure"]) for x in rows),
        "internal_tool_violations": internal_tool_violations,
        "finish_rate": sum(int(x["finish"]) for x in rows) / max(1, len(rows)),
        "mean_f1": sum(float(x["answer_f1"]) for x in rows) / max(1, len(rows)),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
