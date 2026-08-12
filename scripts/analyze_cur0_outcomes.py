#!/usr/bin/env python3
"""Gate 0A: aggregate paired CUR forced-arm outcomes by question."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def bootstrap_delta(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    # Independent rollouts under two interventions; resample within each arm.
    sims = a[rng.integers(0, len(a), (10000, len(a)))].mean(1) - b[
        rng.integers(0, len(b), (10000, len(b)))
    ].mean(1)
    return tuple(float(x) for x in np.quantile(sims, [0.025, 0.975]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.input.read_text().splitlines() if x.strip()]
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["sample_id"])][str(row["cur_forced_arm"])].append(row)
    rng = np.random.default_rng(args.seed)
    qrows = []
    rollouts_per_arm = []
    for sid, arms in sorted(grouped.items()):
        if (
            set(arms) != {"search", "internal"}
            or len(arms["search"]) != len(arms["internal"])
            or len(arms["search"]) < 4
        ):
            raise SystemExit(f"invalid arms for {sid}: { {k: len(v) for k,v in arms.items()} }")
        rollouts_per_arm.append(len(arms["search"]))
        f1_s = np.asarray([x["answer_f1"] for x in arms["search"]], dtype=float)
        f1_i = np.asarray([x["answer_f1"] for x in arms["internal"]], dtype=float)
        em_s = np.asarray([x["answer_em"] for x in arms["search"]], dtype=float)
        em_i = np.asarray([x["answer_em"] for x in arms["internal"]], dtype=float)
        ci_f1 = bootstrap_delta(f1_s, f1_i, rng)
        ci_em = bootstrap_delta(em_s, em_i, rng)
        row = {
            "sample_id": sid,
            "mean_f1_search": float(f1_s.mean()),
            "mean_f1_internal": float(f1_i.mean()),
            "delta_f1": float(f1_s.mean() - f1_i.mean()),
            "delta_f1_ci95": list(ci_f1),
            "mean_em_search": float(em_s.mean()),
            "mean_em_internal": float(em_i.mean()),
            "delta_em": float(em_s.mean() - em_i.mean()),
            "delta_em_ci95": list(ci_em),
            "mean_search_count_do_search": float(np.mean([x["search_count"] for x in arms["search"]])),
            "mean_observation_tokens_do_search": float(np.mean([x["observation_tokens"] for x in arms["search"]])),
            "mean_response_tokens_search": float(np.mean([x["response_tokens"] for x in arms["search"]])),
            "mean_response_tokens_internal": float(np.mean([x["response_tokens"] for x in arms["internal"]])),
            "finish_rate_search": float(np.mean([x["finish"] for x in arms["search"]])),
            "finish_rate_internal": float(np.mean([x["finish"] for x in arms["internal"]])),
            "canonical_prompt_sha256": arms["search"][0]["canonical_prompt_sha256"],
        }
        row["high_confidence_direction"] = (
            "search" if ci_f1[0] > 0 else "internal" if ci_f1[1] < 0 else "borderline"
        )
        qrows.append(row)

    delta = np.asarray([x["delta_f1"] for x in qrows])
    delta_em = np.asarray([x["delta_em"] for x in qrows])
    summary = {
        "gate": "GATE_0A_PASS" if np.any(delta > 0) and np.any(delta < 0) else "GATE_0A_FAIL",
        "n_questions": len(qrows),
        "n_rollouts_per_arm_distribution": {
            str(n): int(sum(x == n for x in rollouts_per_arm)) for n in sorted(set(rollouts_per_arm))
        },
        "estimand": "E[F1|do(search)] - E[F1|do(internal)]",
        "mean_f1_search": float(np.mean([x["mean_f1_search"] for x in qrows])),
        "mean_f1_internal": float(np.mean([x["mean_f1_internal"] for x in qrows])),
        "mean_delta_f1": float(delta.mean()),
        "median_delta_f1": float(np.median(delta)),
        "delta_f1_quantiles": {str(q): float(np.quantile(delta, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
        "direction_counts": {
            "search_positive": int(np.sum(delta > 0)),
            "internal_negative": int(np.sum(delta < 0)),
            "exact_zero": int(np.sum(delta == 0)),
        },
        "high_confidence_counts": {
            key: sum(x["high_confidence_direction"] == key for x in qrows)
            for key in ("search", "internal", "borderline")
        },
        "mean_em_search": float(np.mean([x["mean_em_search"] for x in qrows])),
        "mean_em_internal": float(np.mean([x["mean_em_internal"] for x in qrows])),
        "mean_delta_em": float(delta_em.mean()),
        "mean_search_count_do_search": float(np.mean([x["mean_search_count_do_search"] for x in qrows])),
        "mean_observation_tokens_do_search": float(np.mean([x["mean_observation_tokens_do_search"] for x in qrows])),
        "finish_rate_search": float(np.mean([x["finish_rate_search"] for x in qrows])),
        "finish_rate_internal": float(np.mean([x["finish_rate_internal"] for x in qrows])),
        "n_borderline_for_n8": sum(x["high_confidence_direction"] == "borderline" for x in qrows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "question_outcomes.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in qrows)
    )
    (args.output_dir / "gate0a_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
