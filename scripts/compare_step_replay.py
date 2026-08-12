#!/usr/bin/env python3
"""Require exact deterministic equivalence between two Phase-25 replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["sample_id"]: row for row in rows}


def trajectory_view(row: dict) -> dict:
    return {
        "canonical_prompt_sha256": row["canonical_prompt_sha256"],
        "step_prompt_sha256": row["step_prompt_sha256"],
        "response_token_ids": row["response_token_ids"],
        "response_mask": row["response_mask"],
        "search_count": row["search_count"],
        "finish": row["finish"],
        "steps": [
            {
                "step_index": step["step_index"],
                "reasoning_text": step["reasoning_text"],
                "candidate_query": step["candidate_query"],
                "checkpoint_token_ids": step["checkpoint_token_ids"],
                "reasoning_raw_token_ids": step["reasoning_raw_token_ids"],
                "query_raw_token_ids": step["query_raw_token_ids"],
                "action": step["action"],
                "answer_token_ids": step.get("answer_token_ids"),
                "answer_closed": step.get("answer_closed"),
            }
            for step in row["step_records"]
        ],
    }


def digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, b = load(args.run_a), load(args.run_b)
    ids_match = set(a) == set(b)
    ids = sorted(set(a) & set(b))
    comparisons = []
    for sample_id in ids:
        va, vb = trajectory_view(a[sample_id]), trajectory_view(b[sample_id])
        comparisons.append(
            {"sample_id": sample_id, "exact_match": va == vb, "sha256_a": digest(va), "sha256_b": digest(vb)}
        )
    passed = ids_match and len(ids) == 8 and all(row["exact_match"] for row in comparisons)
    summary = {
        "gate": "STEP_S0_REPLAY_PASS" if passed else "STEP_S0_REPLAY_FAIL",
        "rows_a": len(a),
        "rows_b": len(b),
        "sample_ids_match": ids_match,
        "exact_trajectory_match_rate": sum(row["exact_match"] for row in comparisons) / max(1, len(comparisons)),
        "comparisons": comparisons,
        "original_test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "comparisons"}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
