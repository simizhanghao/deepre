#!/usr/bin/env python3
"""Combine one Boundary@50 node's training, routing and exact-alignment evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def verifier(path: str) -> dict:
    text = Path(path).read_text()
    values = [float(x) for x in re.findall(r"Maximum absolute difference:\s*([0-9.eE+-]+)", text)]
    return {
        "success": "Verification completed successfully!" in text,
        "max_abs_delta": max(values) if values else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, choices=(10, 25, 50), required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--route-summary", required=True)
    p.add_argument("--full-log", required=True)
    p.add_argument("--fused-log", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--profile", choices=("grpo", "rfpp_baseline", "grpo_no_std"), default="grpo")
    args = p.parse_args()

    rows = [json.loads(x) for x in Path(args.metrics).read_text().splitlines() if x.strip()]
    step_rows = [x for x in rows if int(x["step"]) == args.step]
    if not step_rows:
        raise SystemExit(f"training metrics missing step {args.step}")
    metric = step_rows[-1]
    route = json.loads(Path(args.route_summary).read_text())
    full, fused = verifier(args.full_log), verifier(args.fused_log)
    alignment_pass = all(
        x["success"] and x["max_abs_delta"] is not None and x["max_abs_delta"] <= 1e-5
        for x in (full, fused)
    )

    get = lambda key: metric.get(key)
    mechanism = {
        "sr_need": get("boundary/search_rate_NeedSearch"),
        "sr_no": get("boundary/search_rate_NoSearch"),
        "delta_boundary": get("boundary/delta_boundary"),
        "osr": get("boundary/osr"),
        "usr": get("boundary/usr"),
        "p_internal_NoSearch": (
            None if get("boundary/search_rate_NoSearch") is None
            else 1.0 - get("boundary/search_rate_NoSearch")
        ),
        "mixed_action_group_rate": get("mixed/group_rate"),
        "mixed_NoSearch_rate": get("mixed/NoSearch_rate"),
        "mixed_NeedSearch_rate": get("mixed/NeedSearch_rate"),
        "delta_R_NS": get("mixed/delta_reward_NoSearch_internal_minus_search"),
        "delta_R_Need": get("mixed/delta_reward_NeedSearch_search_minus_internal"),
        "zero_std_group_rate": get("grpo/zero_std_group_rate"),
        "advantage_std": get("grpo/advantage_std"),
        "gradient_norm": get("actor/grad_norm"),
        "clip_ratio": get("actor/pg_clipfrac"),
        "kl": get("actor/kl_loss"),
        "importance_ratio": {
            q: get(f"rollout_corr_diag/importance_ratio_{q}")
            for q in ("p01", "p10", "p50", "p90", "p99")
        },
        "route_margin": route["mean_route_margin"],
    }
    trajectory = {
        "finish_rate": get("agent/finish_rate"),
        "response_clip_ratio": get("response_length/clip_ratio"),
        "final_answer_missing_rate": get("agent/final_answer_missing_rate"),
        "reserve_violations": get("agent/final_answer_reserve_violations"),
        "max_assistant_turn_tokens": get("agent/max_assistant_turn_tokens"),
        "max_observation_turn_tokens": get("agent/max_observation_turn_tokens"),
    }

    sr_need = mechanism["sr_need"]
    sr_no = mechanism["sr_no"]
    mixed = mechanism["mixed_action_group_rate"]
    delta = mechanism["delta_boundary"]
    ns_margin = mechanism["route_margin"].get("NoSearch")
    trajectory_pass = (
        (trajectory["finish_rate"] or 0) >= 0.95
        and (trajectory["response_clip_ratio"] or 0) < 0.05
        and (trajectory["final_answer_missing_rate"] or 0) <= 0.01
        and (trajectory["reserve_violations"] or 0) == 0
    )
    ratio_p99 = mechanism["importance_ratio"]["p99"]
    optimizer_health = (
        (mechanism["clip_ratio"] is None or mechanism["clip_ratio"] <= 0.20)
        and (ratio_p99 is None or 0.5 <= ratio_p99 <= 2.0)
        and (mechanism["kl"] is None or abs(mechanism["kl"]) < 0.05)
    )
    if not alignment_pass:
        verdict = "HARD_STOP_ALIGNMENT"
    elif args.profile in ("rfpp_baseline", "grpo_no_std") and args.step == 10:
        direction = ns_margin is not None and ns_margin < 0.863636
        preservation = mechanism["route_margin"].get("NeedSearch", float("-inf")) >= 1.272
        support = (mixed or 0) > 0 and (mechanism["p_internal_NoSearch"] or 0) > 0
        prefix = "RFPP_BASELINE" if args.profile == "rfpp_baseline" else "GRPO_NO_STD"
        verdict = (
            f"{prefix}_DIRECTION_PASS"
            if direction and preservation and support and trajectory_pass and optimizer_health
            else f"{prefix}_DIRECTION_FAIL"
        )
    elif args.profile == "rfpp_baseline" and args.step == 25:
        routing = (
            (sr_need or 0) >= 0.85 and sr_no is not None and sr_no <= 0.70
            and (delta or 0) >= 0.20 and (mixed or 0) > 0
        )
        verdict = (
            "RFPP_BASELINE_STEP25_PASS"
            if routing and trajectory_pass and optimizer_health else "RFPP_BASELINE_STEP25_FAIL"
        )
    elif args.profile == "rfpp_baseline" and args.step == 50:
        routing = (
            (sr_need or 0) >= 0.85 and sr_no is not None and sr_no <= 0.70
            and (delta or 0) >= 0.20
        )
        verdict = (
            "RFPP_BASELINE_ROUTING_PASS_PENDING_VAL200"
            if routing and trajectory_pass and optimizer_health else "RFPP_BASELINE_ROUTING_FAIL"
        )
    elif args.step == 10:
        collapse = sr_no is not None and sr_no >= 0.95 and (mixed or 0) < 0.05
        direction = (sr_no is not None and sr_no < 0.681818) or (
            ns_margin is not None and ns_margin < 0.863636
        )
        verdict = "STOP_POLICY_COLLAPSE" if collapse else (
            "CONTINUE_TO_25" if direction and (sr_need is None or sr_need >= 0.75) else "REVIEW_STEP10_DIRECTION"
        )
    elif args.step == 25:
        verdict = (
            "CONTINUE_TO_50"
            if delta is not None and delta >= 0.10 and (mixed or 0) > 0.15 and (sr_need or 0) >= 0.75
            else "REVIEW_STEP25_MECHANISM"
        )
    else:
        primary = (
            sr_need is not None and sr_need >= 0.85 and sr_no is not None and sr_no <= 0.70
            and delta is not None and delta >= 0.20
        )
        verdict = "BOUNDARY_ROUTING_GATE_PASS_PENDING_VAL200" if primary else "BOUNDARY_ROUTING_GATE_FAIL"

    summary = {
        "step": args.step,
        "verdict": verdict,
        "alignment": {"pass": alignment_pass, "full_logits": full, "fused_lce": fused},
        "trajectory": trajectory,
        "trajectory_gate_pass": trajectory_pass,
        "optimizer_health_pass": optimizer_health,
        "mechanism": mechanism,
        "answer_reward": get("reward/answer_reward/mean"),
        "evidence_f1": get("evidence/f1/mean"),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if verdict.startswith("HARD_STOP"):
        raise SystemExit(20)


if __name__ == "__main__":
    main()
