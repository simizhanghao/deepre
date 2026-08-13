#!/usr/bin/env python3
"""One-shot question-paired Fresh Val3 decision for the frozen Step Gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.metrics import exact_match, token_f1
from src.rl.rewards_cur import extract_answer


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def step_outcome(row, tokenizer, golds):
    text = tokenizer.decode(row["response_token_ids"], skip_special_tokens=False)
    prediction = extract_answer(text) or ""
    steps = row["step_records"]
    observation = sum(int(step.get("observation_tokens") or 0) for step in steps)
    raw = sum(len(step.get("reasoning_raw_token_ids") or []) + len(step.get("query_raw_token_ids") or []) for step in steps)
    raw += sum(max(0, len(step.get("answer_token_ids") or []) - int(step.get("answer_forced_prefix_token_count") or 0)) for step in steps)
    metrics = row.get("metrics") or {}
    probabilities = [float(x) for x in metrics.get("gate_probabilities") or []]
    return {
        "prediction": prediction,
        "f1": float(token_f1(prediction, golds)) if prediction else 0.0,
        "em": float(exact_match(prediction, golds)) if prediction else 0.0,
        "finish": int(row["finish"]),
        "search_calls": int(row["search_count"]),
        "response_tokens": int(row["response_tokens"]),
        "observation_tokens": observation,
        "raw_generation_tokens": raw,
        "token_cost": int(row["response_tokens"]),
        "generation_seconds": float(metrics.get("generate_sequences") or 0.0),
        "gate_search_decisions": int(metrics.get("gate_search_count") or 0),
        "gate_continue_decisions": int(metrics.get("gate_continue_count") or 0),
        "mean_gate_probability": float(np.mean(probabilities)) if probabilities else None,
    }


def mean_ci(values, rng, replicates=10000):
    values = np.asarray(values, dtype=np.float64)
    boot = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--root-outcomes", type=Path, required=True)
    ap.add_argument("--step-allsearch", type=Path, required=True)
    ap.add_argument("--step-gate", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    ids = [str(value) for value in manifest["question_ids"]]
    if len(ids) != 128:
        raise RuntimeError("Val3 must contain exactly 128 frozen questions")
    root_rows = read_jsonl(args.root_outcomes)
    root = {}
    for row in root_rows:
        arm = "no_search" if row["cur_forced_arm"] == "internal" else "old_always_search"
        root[(str(row["sample_id"]), arm)] = {
            "prediction": row["pred"], "f1": float(row["answer_f1"]), "em": float(row["answer_em"]),
            "finish": int(row["finish"]), "search_calls": int(row["search_count"]),
            "response_tokens": int(row["response_tokens"]), "observation_tokens": int(row["observation_tokens"]),
            "raw_generation_tokens": int(row["assistant_tokens"]), "token_cost": int(row["response_tokens"]),
            "generation_seconds": float(row.get("generation_seconds") or 0.0),
        }
    allsearch_rows = {str(row["sample_id"]): row for row in read_jsonl(args.step_allsearch)}
    gate_rows = {str(row["sample_id"]): row for row in read_jsonl(args.step_gate)}
    source = {str(row["extra_info"]["sample_id"]): row for row in pq.read_table(args.data_dir / "step_gate.parquet").to_pylist()}
    if len(root) != 256 or set(allsearch_rows) != set(ids) or set(gate_rows) != set(ids):
        raise RuntimeError(f"incomplete Val3 arms: root={len(root)} allsearch={len(allsearch_rows)} gate={len(gate_rows)}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    rows = []
    for sample_id in ids:
        golds = list(source[sample_id]["reward_model"]["ground_truth"]["target"])
        rows.append({
            "sample_id": sample_id,
            "no_search": root[(sample_id, "no_search")],
            "old_always_search": root[(sample_id, "old_always_search")],
            "step_allsearch": step_outcome(allsearch_rows[sample_id], tokenizer, golds),
            "step_gate": step_outcome(gate_rows[sample_id], tokenizer, golds),
        })
    arms = ("no_search", "old_always_search", "step_allsearch", "step_gate")
    metrics = ("f1", "em", "finish", "search_calls", "response_tokens", "observation_tokens", "raw_generation_tokens", "token_cost", "generation_seconds")
    rng = np.random.default_rng(2026081302)
    arm_summary = {
        arm: {metric: mean_ci([row[arm][metric] for row in rows], rng) for metric in metrics}
        for arm in arms
    }
    paired = {
        "gate_minus_step_allsearch_f1": mean_ci([row["step_gate"]["f1"] - row["step_allsearch"]["f1"] for row in rows], rng),
        "gate_minus_old_alwayssearch_f1": mean_ci([row["step_gate"]["f1"] - row["old_always_search"]["f1"] for row in rows], rng),
        "gate_minus_step_allsearch_calls": mean_ci([row["step_gate"]["search_calls"] - row["step_allsearch"]["search_calls"] for row in rows], rng),
        "gate_minus_old_alwayssearch_tokens": mean_ci([row["step_gate"]["token_cost"] - row["old_always_search"]["token_cost"] for row in rows], rng),
    }
    gate_calls = arm_summary["step_gate"]["search_calls"]["mean"]
    allsearch_calls = arm_summary["step_allsearch"]["search_calls"]["mean"]
    scientific = (
        gate_calls <= 0.75 * allsearch_calls
        and arm_summary["step_gate"]["f1"]["mean"] >= arm_summary["step_allsearch"]["f1"]["mean"] - 0.02
    )
    project = (
        arm_summary["step_gate"]["token_cost"]["mean"] < arm_summary["old_always_search"]["token_cost"]["mean"]
        and arm_summary["step_gate"]["f1"]["mean"] >= arm_summary["old_always_search"]["f1"]["mean"] - 0.02
    )
    summary = {
        "gate": "STEP_VAL3_PASS" if scientific and project else "STEP_VAL3_FAIL",
        "questions": 128,
        "arms": arm_summary,
        "paired_bootstrap": paired,
        "scientific_pass": scientific,
        "scientific_calls_ratio": gate_calls / max(allsearch_calls, 1e-12),
        "scientific_f1_delta": arm_summary["step_gate"]["f1"]["mean"] - arm_summary["step_allsearch"]["f1"]["mean"],
        "project_pass": project,
        "project_token_delta": arm_summary["step_gate"]["token_cost"]["mean"] - arm_summary["old_always_search"]["token_cost"]["mean"],
        "project_f1_delta": arm_summary["step_gate"]["f1"]["mean"] - arm_summary["old_always_search"]["f1"]["mean"],
        "original_test_read": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_question.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
