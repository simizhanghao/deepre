#!/usr/bin/env python3
"""Audit the frozen Phase-25 S0 rollout contract without reading Val/Test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    steps = [step for row in rows for step in row["step_records"]]
    continued = [(row, index) for row in rows for index, step in enumerate(row["step_records"]) if step["action"] == "continue"]
    query_valid = [
        bool(step["candidate_query"].strip())
        and len(step["candidate_query"]) <= 512
        and "<" not in step["candidate_query"]
        and "\n" not in step["candidate_query"]
        for step in steps
    ]
    searched_queries: dict[str, list[str]] = {}
    for row in rows:
        searched_queries[row["sample_id"]] = [
            step["candidate_query"] for step in row["step_records"] if step["action"] == "search"
        ]
    duplicate_executions = sum(
        len(queries) - len(set(queries)) for queries in searched_queries.values()
    )
    denominator = max(1, len(rows))
    step_denominator = max(1, len(steps))
    summary = {
        "stage": "phase25_s0",
        "rows": len(rows),
        "checkpoints": len(steps),
        "parser_valid_rate": sum(row["metrics"]["parser_valid"] for row in rows) / denominator,
        "checkpoint_close_rate": sum(step["matched_close"] == "</think>" for step in steps) / step_denominator,
        "query_field_valid_rate": sum(query_valid) / step_denominator,
        "finish_rate": sum(row["finish"] for row in rows) / denominator,
        "response_clip_rate": sum(row["metrics"]["response_clipped"] > 0 for row in rows) / denominator,
        "tool_violations": sum(row["metrics"]["tool_violations"] for row in rows),
        "final_answer_reserve_violations": sum(row["metrics"]["final_answer_reserve_violations"] for row in rows),
        "duplicate_search_executions": duplicate_executions,
        "continue_count": len(continued),
        "continue_reaches_next_checkpoint_rate": (
            sum(index + 1 < len(row["step_records"]) for row, index in continued) / max(1, len(continued))
        ),
        "reasoning_fallback_rate": sum(step.get("reasoning_fallback", False) for step in steps) / step_denominator,
        "query_fallback_rate": sum(step.get("query_fallback", False) for step in steps) / step_denominator,
        "search_count": sum(row["search_count"] for row in rows),
        "original_test_read": False,
    }
    passed = (
        summary["rows"] == args.expected
        and summary["parser_valid_rate"] == 1.0
        and summary["checkpoint_close_rate"] == 1.0
        and summary["query_field_valid_rate"] >= 0.98
        and summary["finish_rate"] >= 0.98
        and summary["response_clip_rate"] < 0.05
        and summary["tool_violations"] == 0
        and summary["final_answer_reserve_violations"] == 0
        and summary["duplicate_search_executions"] == 0
        and summary["continue_reaches_next_checkpoint_rate"] == 1.0
    )
    summary["gate"] = "STEP_S0_CONTRACT_PASS" if passed else "STEP_S0_CONTRACT_FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
