#!/usr/bin/env python3
"""Summarize one frozen CUR-1 paired capture without opening sealed test."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081201)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[str(row["sample_id"])][str(row["cur_forced_arm"])].append(row)

    errors: list[str] = []
    qrows: list[dict] = []
    rollout_counts = Counter()
    for sample_id, arms in sorted(groups.items()):
        if set(arms) != {"search", "internal"}:
            errors.append(f"{sample_id}: arms={sorted(arms)}")
            continue
        if len(arms["search"]) != len(arms["internal"]):
            errors.append(f"{sample_id}: unbalanced arms")
            continue
        rollout_counts[len(arms["search"])] += 1
        hashes = {row["canonical_prompt_sha256"] for arm in arms.values() for row in arm}
        if len(hashes) != 1:
            errors.append(f"{sample_id}: prompt hash mismatch")
        f1_search = mean(arms["search"], "answer_f1")
        f1_internal = mean(arms["internal"], "answer_f1")
        em_search = mean(arms["search"], "answer_em")
        em_internal = mean(arms["internal"], "answer_em")
        qrows.append(
            {
                "sample_id": sample_id,
                "mean_f1_search": f1_search,
                "mean_f1_internal": f1_internal,
                "delta_f1": f1_search - f1_internal,
                "mean_em_search": em_search,
                "mean_em_internal": em_internal,
                "delta_em": em_search - em_internal,
            }
        )

    delta = np.asarray([row["delta_f1"] for row in qrows], dtype=float)
    delta_em = np.asarray([row["delta_em"] for row in qrows], dtype=float)
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(delta), size=(10000, len(delta)))
    bootstrap = delta[indices].mean(axis=1)

    arm_summary = {}
    for arm in ("search", "internal"):
        arm_rows = [row for row in rows if row["cur_forced_arm"] == arm]
        arm_summary[arm] = {
            "rows": len(arm_rows),
            "mean_f1": mean(arm_rows, "answer_f1"),
            "mean_em": mean(arm_rows, "answer_em"),
            "finish_rate": mean(arm_rows, "finish"),
            "format_valid_rate": mean(arm_rows, "format"),
            "action_valid_rate": mean(arm_rows, "cur_forced_action_valid"),
            "policy_failure_rate": mean(arm_rows, "cur_policy_failure"),
            "mean_search_count": mean(arm_rows, "search_count"),
            "mean_observation_tokens": mean(arm_rows, "observation_tokens"),
            "mean_response_tokens": mean(arm_rows, "response_tokens"),
        }

    summary = {
        "gate": "CUR1_PAIRED_CAPTURE_PASS" if not errors else "CUR1_PAIRED_CAPTURE_FAIL",
        "rows": len(rows),
        "questions": len(groups),
        "rollouts_per_arm_distribution": {str(k): v for k, v in sorted(rollout_counts.items())},
        "arm_summary": arm_summary,
        "paired": {
            "mean_delta_f1_search_minus_internal": float(delta.mean()),
            "question_bootstrap_ci95_mean_delta_f1": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "median_delta_f1": float(np.median(delta)),
            "mean_delta_em_search_minus_internal": float(delta_em.mean()),
            "direction_counts": {
                "search_positive": int(np.sum(delta > 0)),
                "internal_positive": int(np.sum(delta < 0)),
                "tie": int(np.sum(delta == 0)),
            },
            "oracle_mean_f1_from_arm_means": float(
                np.mean([max(row["mean_f1_search"], row["mean_f1_internal"]) for row in qrows])
            ),
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
