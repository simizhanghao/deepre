#!/usr/bin/env python3
"""Summarize frozen-20 VeXact Gate A1 from captured logits and verifier logs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--full-verifier-log", required=True)
    parser.add_argument("--fused-verifier-log", required=True)
    parser.add_argument(
        "--hf-reference",
        default=str(REPO / "results/16_audit_routing_exploration/worker_mismatch/hf_route_scores.jsonl"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def verifier_status(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    differences = [float(value) for value in re.findall(r"Maximum absolute difference:\s*([0-9.eE+-]+)", text)]
    return {
        "success": "success: True" in text and "Verification completed successfully!" in text,
        "maximum_absolute_differences": differences,
        "maximum_absolute_difference": max(differences) if differences else None,
        "path": str(path),
    }


def main() -> None:
    args = parse_args()
    if args.debug or args.max_samples != 20:
        raise SystemExit("Gate A1 summary requires --max-samples 20 without --debug")

    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    capture_dir = Path(args.capture_dir).resolve()
    hf_path = Path(args.hf_reference).resolve()
    full_log = Path(args.full_verifier_log).resolve()
    fused_log = Path(args.fused_verifier_log).resolve()
    expected_root = (REPO / "results/17_rollout_alignment/calibration").resolve()
    if output_dir != expected_root and expected_root not in output_dir.parents:
        raise SystemExit(f"output must be under {expected_root}")
    for path in (config_path, capture_dir, hf_path, full_log, fused_log):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    import torch

    logits_by_id = torch.load(capture_dir / "all_logits.pt", map_location="cpu")
    metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))
    capture_summary = json.loads((capture_dir / "capture_summary.json").read_text(encoding="utf-8"))
    hf_by_id = {str(row["sample_id"]): row for row in read_jsonl(hf_path)}
    if len(logits_by_id) != args.max_samples:
        raise SystemExit(f"wanted {args.max_samples} captures, found {len(logits_by_id)}")

    rows: list[dict[str, Any]] = []
    old_hf_search_deltas: list[float] = []
    old_hf_internal_deltas: list[float] = []
    for sample_id, logits_list in logits_by_id.items():
        if len(logits_list) != 1:
            raise SystemExit(f"expected one logit tensor for {sample_id}")
        log_probs = torch.log_softmax(logits_list[0].reshape(-1).float(), dim=-1)
        rollout_search = float(log_probs[27])
        rollout_internal = float(log_probs[4159])
        hf_row = hf_by_id[sample_id]
        delta_search = abs(rollout_search - float(hf_row["L_search_first_tok"]))
        delta_internal = abs(rollout_internal - float(hf_row["L_internal_first_tok"]))
        old_hf_search_deltas.append(delta_search)
        old_hf_internal_deltas.append(delta_internal)
        rows.append(
            {
                "sample_id": sample_id,
                "boundary": metadata[sample_id]["boundary"],
                "canonical_prompt_sha256": metadata[sample_id]["canonical_prompt_sha256"],
                "vexact_logp_search_tok0": rollout_search,
                "vexact_logp_internal_tok0": rollout_internal,
                "old_hf_logp_search_tok0": float(hf_row["L_search_first_tok"]),
                "old_hf_logp_internal_tok0": float(hf_row["L_internal_first_tok"]),
                "old_hf_abs_delta_search_tok0": delta_search,
                "old_hf_abs_delta_internal_tok0": delta_internal,
            }
        )

    full_status = verifier_status(full_log)
    fused_status = verifier_status(fused_log)
    exact_pass = (
        full_status["success"]
        and fused_status["success"]
        and full_status["maximum_absolute_difference"] is not None
        and fused_status["maximum_absolute_difference"] is not None
        and full_status["maximum_absolute_difference"] <= 1e-6
        and fused_status["maximum_absolute_difference"] <= 1e-6
    )
    p_internal = capture_summary["p_internal_NoSearch"]
    mixed_rate = capture_summary["mixed_action_group_rate"]
    other_count = sum(counts.get("other", 0) for counts in capture_summary["by_boundary"].values())
    support_pass = p_internal is not None and p_internal > 0.10 and mixed_rate is not None and mixed_rate > 0 and other_count == 0

    rows_path = output_dir / "vexact_gate_a1_route_scores.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "purpose": "vexact_frozen20_gate_a1",
        "gate": "PASS" if exact_pass and support_pass else "FAIL",
        "config": str(config_path),
        "seed": args.seed,
        "n_questions": len(rows),
        "n_rollouts": capture_summary["n_rollouts"],
        "authoritative_exact_contract": {
            "reference": "VeOmni batch-invariant actor-model forward",
            "candidate": "VeXact rollout",
            "full_logits": full_status,
            "fused_lce": fused_status,
            "pass": exact_pass,
        },
        "natural_sampling": {
            "temperature": capture_summary["temperature"],
            "top_p": capture_summary["top_p"],
            "by_boundary": capture_summary["by_boundary"],
            "p_internal_NoSearch": p_internal,
            "mixed_action_group_rate": mixed_rate,
            "other_count": other_count,
            "pass": support_pass,
        },
        "old_hf_continuity_diagnostic": {
            "median_abs_delta_search_tok0": statistics.median(old_hf_search_deltas),
            "p95_abs_delta_search_tok0": percentile(old_hf_search_deltas, 0.95),
            "median_abs_delta_internal_tok0": statistics.median(old_hf_internal_deltas),
            "p95_abs_delta_internal_tok0": percentile(old_hf_internal_deltas, 0.95),
            "is_authoritative_gate": False,
        },
        "route_scores_path": str(rows_path),
        "natural_samples_path": capture_summary["natural_samples_path"],
    }
    summary_path = output_dir / "vexact_gate_a1_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
