#!/usr/bin/env python3
"""Audit S1 causal pairs and compute local quality/cost/query Oracle evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.metrics import exact_match, normalize_answer, token_f1
from src.rl.rewards_cur import extract_answer


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def outcome(row: dict, tokenizer, golds: list[str]) -> dict:
    text = tokenizer.decode(row["response_token_ids"], skip_special_tokens=False)
    prediction = extract_answer(text) or ""
    steps = row["step_records"]
    observation_tokens = sum(int(step.get("observation_tokens") or 0) for step in steps)
    raw_generation_tokens = sum(
        len(step.get("reasoning_raw_token_ids") or []) + len(step.get("query_raw_token_ids") or [])
        for step in steps
    ) + sum(
        max(0, len(step.get("answer_token_ids") or []) - int(step.get("answer_forced_prefix_token_count") or 0))
        for step in steps
    )
    return {
        "prediction": prediction,
        "f1": float(token_f1(prediction, golds)) if prediction else 0.0,
        "em": float(exact_match(prediction, golds)) if prediction else 0.0,
        "search_calls": int(row["search_count"]),
        "response_tokens": int(row["response_tokens"]),
        "observation_tokens": observation_tokens,
        "raw_generation_tokens": raw_generation_tokens,
        "total_token_proxy": int(row["response_tokens"]),
        "finish": int(row["finish"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--base-records", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    records = [row for row in load_jsonl(args.records) if not row.get("step_padding")]
    base_records = load_jsonl(args.base_records)
    selections = {row["branch_id"]: row for row in load_jsonl(args.selections)}
    source = {
        str(row["extra_info"]["sample_id"]): row
        for row in pq.read_table(args.data).to_pylist()
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    grouped: dict[str, dict[str, dict]] = {}
    for row in records:
        grouped.setdefault(row["step_branch_id"], {})[row["step_branch_arm"]] = row
    if set(grouped) != set(selections):
        raise RuntimeError("branch record IDs do not equal frozen selections")

    pairs = []
    for branch_id in sorted(grouped):
        arms = grouped[branch_id]
        if set(arms) != {"search", "continue"}:
            raise RuntimeError(f"incomplete arms for {branch_id}: {sorted(arms)}")
        selection = selections[branch_id]
        target = int(selection["target_index"])
        target_steps = {}
        for arm, row in arms.items():
            matches = [step for step in row["step_records"] if int(step["step_index"]) == target]
            if len(matches) != 1:
                raise RuntimeError(f"missing target checkpoint: {branch_id}/{arm}")
            target_steps[arm] = matches[0]
        search_step, continue_step = target_steps["search"], target_steps["continue"]
        prefix_exact = (
            search_step["state_prefix_sha256"] == continue_step["state_prefix_sha256"]
            == selection["state_prefix_sha256"]
            and search_step["checkpoint_token_ids"] == continue_step["checkpoint_token_ids"]
            == selection["checkpoint_token_ids"]
            and search_step["candidate_query"] == continue_step["candidate_query"]
            == selection["candidate_query"]
        )
        action_exact = search_step["action"] == "search" and continue_step["action"] == "continue"
        source_row = source[selection["sample_id"]]
        golds = list(source_row["reward_model"]["ground_truth"]["target"])
        search_out = outcome(arms["search"], tokenizer, golds)
        continue_out = outcome(arms["continue"], tokenizer, golds)
        supporting_titles = {
            normalize_answer(fact["title"])
            for fact in source_row["reward_model"]["ground_truth"]["supporting_facts"]
        }
        documents = list((search_step.get("tool_metrics") or {}).get("documents") or [])
        retrieved_titles = {normalize_answer(document["title"]) for document in documents}
        support_hits = supporting_titles & retrieved_titles
        delta_f1 = search_out["f1"] - continue_out["f1"]
        delta_em = search_out["em"] - continue_out["em"]
        delta_calls = search_out["search_calls"] - continue_out["search_calls"]
        delta_tokens = search_out["total_token_proxy"] - continue_out["total_token_proxy"]
        pairs.append(
            {
                "branch_id": branch_id,
                "sample_id": selection["sample_id"],
                "target_index": target,
                "candidate_query": selection["candidate_query"],
                "prefix_exact": prefix_exact,
                "action_exact": action_exact,
                "duplicate_query": int(search_step["duplicate_query"]),
                "supporting_titles": sorted(supporting_titles),
                "retrieved_titles": sorted(retrieved_titles),
                "support_title_hits": sorted(support_hits),
                "supporting_title_recall": len(support_hits) / max(1, len(supporting_titles)),
                "search": search_out,
                "continue": continue_out,
                "delta_f1": delta_f1,
                "delta_em": delta_em,
                "delta_search_calls": delta_calls,
                "delta_token_proxy": delta_tokens,
                "preference": (
                    "search" if delta_f1 > 0.02 else
                    "continue" if delta_f1 < -0.02 else
                    ("continue" if continue_out["search_calls"] <= search_out["search_calls"] else "search")
                ),
                "preference_weight": max(abs(delta_f1), 0.02 if delta_calls != 0 else 0.0),
            }
        )

    # Local counterfactual frontier: start at SEARCH_NOW for every selected
    # state, then take the least-quality-loss CONTINUE substitutions until the
    # aggregate call count is reduced by at least 25%.
    fixed_calls = sum(pair["search"]["search_calls"] for pair in pairs)
    target_savings = 0.25 * fixed_calls
    candidates = [
        pair for pair in pairs if pair["search"]["search_calls"] > pair["continue"]["search_calls"]
    ]
    candidates.sort(
        key=lambda pair: (
            (pair["search"]["f1"] - pair["continue"]["f1"])
            / (pair["search"]["search_calls"] - pair["continue"]["search_calls"]),
            pair["branch_id"],
        )
    )
    selected_continue: set[str] = set()
    savings = 0
    for pair in candidates:
        if savings >= target_savings:
            break
        selected_continue.add(pair["branch_id"])
        savings += pair["search"]["search_calls"] - pair["continue"]["search_calls"]
    frontier_f1 = mean([
        pair["continue"]["f1"] if pair["branch_id"] in selected_continue else pair["search"]["f1"]
        for pair in pairs
    ])
    fixed_f1 = mean([pair["search"]["f1"] for pair in pairs])
    frontier_calls = sum(
        pair["continue"]["search_calls"] if pair["branch_id"] in selected_continue else pair["search"]["search_calls"]
        for pair in pairs
    )

    all_base_steps = [step for row in base_records for step in row["step_records"]]
    valid_base_queries = [
        not step["query_is_none"] and not step["duplicate_query"] for step in all_base_steps
    ]
    cost_saving = [
        pair["continue"]["f1"] >= pair["search"]["f1"] - 0.02
        and pair["continue"]["search_calls"] < pair["search"]["search_calls"]
        for pair in pairs
    ]
    prefix_rate = mean([float(pair["prefix_exact"] and pair["action_exact"]) for pair in pairs])
    cost_saving_rate = mean([float(value) for value in cost_saving])
    frontier_reduction = 1.0 - frontier_calls / max(1, fixed_calls)
    frontier_quality_pass = frontier_f1 >= fixed_f1 - 0.02
    headroom_pass = cost_saving_rate >= 0.25 and frontier_reduction >= 0.25 and frontier_quality_pass
    contract_pass = (
        prefix_rate == 1.0
        and all(pair["search"]["finish"] and pair["continue"]["finish"] for pair in pairs)
        and len(pairs) == len(selections)
    )
    summary = {
        "gate": (
            "STEP_ADAPTIVE_HEADROOM_PASS" if contract_pass and headroom_pass else
            "STEP_S1_CONTRACT_FAIL" if not contract_pass else
            "STEP_ADAPTIVE_HEADROOM_FAIL"
        ),
        "questions": len({pair["sample_id"] for pair in pairs}),
        "selected_states": len(pairs),
        "paired_rows": len(records),
        "prefix_and_action_exact_rate": prefix_rate,
        "base_valid_nonduplicate_query_rate": mean([float(value) for value in valid_base_queries]),
        "search_helpful_rate": mean([float(pair["delta_f1"] > 0) for pair in pairs]),
        "search_helpful_count": sum(pair["delta_f1"] > 0 for pair in pairs),
        "continue_safe_rate": mean([float(pair["delta_f1"] <= 0) for pair in pairs]),
        "cost_saving_continue_rate": cost_saving_rate,
        "mean_supporting_title_recall_at_5": mean([pair["supporting_title_recall"] for pair in pairs]),
        "any_supporting_title_hit_rate": mean([float(bool(pair["support_title_hits"])) for pair in pairs]),
        "mean_delta_f1_search_minus_continue": mean([pair["delta_f1"] for pair in pairs]),
        "mean_delta_em_search_minus_continue": mean([pair["delta_em"] for pair in pairs]),
        "mean_delta_search_calls": mean([pair["delta_search_calls"] for pair in pairs]),
        "mean_delta_token_proxy": mean([pair["delta_token_proxy"] for pair in pairs]),
        "local_frontier_25pct": {
            "fixed_search_f1": fixed_f1,
            "frontier_f1": frontier_f1,
            "quality_floor": fixed_f1 - 0.02,
            "fixed_search_calls": fixed_calls,
            "frontier_calls": frontier_calls,
            "retrieval_reduction": frontier_reduction,
            "quality_pass": frontier_quality_pass,
        },
        "preference_counts": {
            "search": sum(pair["preference"] == "search" for pair in pairs),
            "continue": sum(pair["preference"] == "continue" for pair in pairs),
        },
        "original_test_read": False,
        "val3_read": False,
    }
    args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
    with args.pairs_output.open("w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + "\n")
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not contract_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
