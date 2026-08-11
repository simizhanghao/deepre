#!/usr/bin/env python3
"""Audit whether saved trajectories support exact offline optimizer attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_exact_capture(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from types import SimpleNamespace
    from verl.trainer.ppo.core_algos import (
        compute_grpo_outcome_advantage,
        compute_reinforce_plus_plus_baseline_outcome_advantage,
        compute_reinforce_plus_plus_outcome_advantage,
    )

    capture_dir = Path(args.capture_dir).resolve()
    step_files = sorted(capture_dir.glob("step_*.npz"))
    if not step_files:
        raise SystemExit(f"no step NPZ files under {capture_dir}")

    chunks = [np.load(path, allow_pickle=False) for path in step_files]
    names = (
        "response_token_ids", "response_mask", "token_level_rewards",
        "old_log_probs", "rollout_log_probs", "online_advantages", "reward_components",
    )
    arrays = {}
    for name in names:
        values = [chunk[name] for chunk in chunks]
        if name == "reward_components":
            arrays[name] = np.concatenate(values, axis=0)
            continue
        width = max(value.shape[1] for value in values)
        padded = [
            np.pad(value, ((0, 0), (0, width - value.shape[1])), mode="constant")
            for value in values
        ]
        arrays[name] = np.concatenate(padded, axis=0)
    identities = [
        json.loads(raw)
        for chunk in chunks
        for raw in chunk["identity_json"].tolist()
    ]
    for chunk in chunks:
        chunk.close()

    n = len(identities)
    if any(value.shape[0] != n for value in arrays.values()):
        raise SystemExit("row count mismatch across captured tensors")
    root_dir = Path(args.root_logits_dir).resolve()
    logits_by_sample = torch.load(root_dir / "all_logits.pt", map_location="cpu", weights_only=True)

    root_logp_search = np.empty(n, dtype=np.float32)
    root_logp_internal = np.empty(n, dtype=np.float32)
    for i, row in enumerate(identities):
        sid = row["sample_id"]
        if sid not in logits_by_sample:
            raise SystemExit(f"missing root logits for sample_id={sid}")
        logits = logits_by_sample[sid][0].float().reshape(-1)
        lp = torch.log_softmax(logits, dim=-1)
        root_logp_search[i] = float(lp[27])
        root_logp_internal[i] = float(lp[4159])
    root_p_search = np.exp(root_logp_search)
    root_p_internal = np.exp(root_logp_internal)

    rewards = torch.from_numpy(arrays["token_level_rewards"]).float()
    mask = torch.from_numpy(arrays["response_mask"]).bool()
    uids = np.asarray([row["uid"] for row in identities])
    algo = SimpleNamespace(gamma=1.0)
    estimator_tensors = {
        "grpo": compute_grpo_outcome_advantage(
            rewards.clone(), mask, uids, norm_adv_by_std_in_grpo=True
        )[0],
        "grpo_no_std": compute_grpo_outcome_advantage(
            rewards.clone(), mask, uids, norm_adv_by_std_in_grpo=False
        )[0],
        "reinforce_plus_plus": compute_reinforce_plus_plus_outcome_advantage(
            rewards.clone(), mask, config=algo
        )[0],
        "reinforce_plus_plus_baseline": compute_reinforce_plus_plus_baseline_outcome_advantage(
            rewards.clone(), mask, uids, config=algo
        )[0],
    }

    group_routes: dict[str, set[str]] = defaultdict(set)
    group_sizes: Counter[str] = Counter()
    for row in identities:
        group_sizes[row["uid"]] += 1
        group_routes[row["uid"]].add(row["route_first"])
    mixed_ns = sum(
        routes.issuperset({"search", "internal"})
        and next(x["boundary"] for x in identities if x["uid"] == uid) == "NoSearch"
        for uid, routes in group_routes.items()
    )

    total_reward = arrays["token_level_rewards"].sum(axis=1)
    policy_tokens = arrays["response_mask"].sum(axis=1).astype(np.int32)
    routes = np.asarray([row["route_first"] for row in identities])
    boundaries = np.asarray([row["boundary"] for row in identities])
    valid_routes = np.isin(routes, ["search", "internal"])
    p_internal_ns = float(np.mean(routes[boundaries == "NoSearch"] == "internal")) if np.any(boundaries == "NoSearch") else None
    length_ratio = (
        float(policy_tokens[routes == "search"].mean() / policy_tokens[routes == "internal"].mean())
        if np.any(routes == "search") and np.any(routes == "internal") else None
    )

    estimator_reports: dict[str, Any] = {}
    for name, advantage in estimator_tensors.items():
        a0 = advantage[:, 0].detach().cpu().numpy().astype(np.float64)
        grad = a0 * ((routes == "search").astype(np.float64) - root_p_search.astype(np.float64))
        by_boundary = {label: float(grad[boundaries == label].sum()) for label in ("NoSearch", "NeedSearch", "Undetermined")}
        g_ns, g_need, g_u = (by_boundary[x] for x in ("NoSearch", "NeedSearch", "Undetermined"))
        competition = abs(g_need + g_u) / (abs(g_ns) + 1e-12)
        bucket: dict[str, Any] = {}
        for label in ("NeedSearch", "NoSearch", "Undetermined"):
            for route in ("search", "internal"):
                sel = (boundaries == label) & (routes == route)
                key = f"{label}/{route}"
                bucket[key] = {
                    "n": int(sel.sum()),
                    "mean_reward": float(total_reward[sel].mean()) if np.any(sel) else None,
                    "mean_advantage": float(a0[sel].mean()) if np.any(sel) else None,
                    "mean_policy_tokens": float(policy_tokens[sel].mean()) if np.any(sel) else None,
                    "root_gradient_mass": float(grad[sel].sum()) if np.any(sel) else 0.0,
                }
        estimator_reports[name] = {
            "G_NS": g_ns,
            "G_Need": g_need,
            "G_U": g_u,
            "gradient_competition_ratio": competition,
            "offline_gate": bool(g_ns < 0 and g_need > 0 and abs(g_ns) >= 0.25 * abs(g_need) and mixed_ns >= args.min_mixed_nosearch_groups),
            "boundary_route": bucket,
        }

    component_ok = bool(np.isfinite(arrays["reward_components"]).all())
    integrity_errors = []
    if args.expected_trajectories and n != args.expected_trajectories:
        integrity_errors.append(f"expected {args.expected_trajectories} rows, got {n}")
    bad_groups = {uid: size for uid, size in group_sizes.items() if size != args.n_rollouts}
    if bad_groups:
        integrity_errors.append(f"{len(bad_groups)} uid groups do not have size {args.n_rollouts}")
    if not bool(np.all(policy_tokens > 0)):
        integrity_errors.append("empty response_mask row")
    if not component_ok:
        integrity_errors.append("missing/nonfinite reward component")
    if not bool(np.isfinite(root_p_search).all() and np.isfinite(root_p_internal).all()):
        integrity_errors.append("invalid root probabilities")
    if not bool(np.all(arrays["response_token_ids"][:, 0][valid_routes] == np.where(routes[valid_routes] == "search", 27, 4159))):
        integrity_errors.append("route_first disagrees with root response token")

    final_npz = capture_dir / "attribution_capture.npz"
    np.savez_compressed(
        final_npz,
        **arrays,
        root_logp_search=root_logp_search,
        root_logp_internal=root_logp_internal,
        root_p_search=root_p_search,
        root_p_internal=root_p_internal,
        identity_json=np.asarray([json.dumps(row, sort_keys=True) for row in identities]),
    )
    report = {
        "purpose": "fixed_policy_attribution_capture",
        "gate": "CAPTURE_PASS" if not integrity_errors else "CAPTURE_FAIL",
        "n_trajectories": n,
        "n_groups": len(group_sizes),
        "n_rollouts": args.n_rollouts,
        "integrity_errors": integrity_errors,
        "group_size_histogram": dict(Counter(group_sizes.values())),
        "route_counts": dict(Counter(routes.tolist())),
        "boundary_counts": dict(Counter(boundaries.tolist())),
        "p_internal_NoSearch": p_internal_ns,
        "mixed_NoSearch_groups": mixed_ns,
        "policy_token_length_ratio_search_internal": length_ratio,
        "estimators": estimator_reports,
    }
    summary_path = capture_dir / "capture_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "purpose": "fixed_policy_attribution_capture",
        "model": "Evidence@400",
        "checkpoint_config_sha256": _sha256(Path(args.model_config)),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n_rollouts": args.n_rollouts,
        "forward_only": True,
        "backward": False,
        "optimizer_step": False,
        "scheduler_step": False,
        "checkpoint_written": False,
        "step_files": [path.name for path in step_files],
        "root_logits_dir": str(root_dir),
        "raw_npz": str(final_npz),
        "raw_npz_sha256": _sha256(final_npz),
    }
    (capture_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if integrity_errors:
        raise SystemExit(1)


FULL_REQUIRED = {
    "sample_id",
    "route_first",
    "total_reward",
    "answer_reward",
    "evidence_reward",
    "cost_reward",
    "format_reward",
    "policy_token_count",
    "root_p_search",
}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": percentile(values, 0.90),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir")
    p.add_argument("--root-logits-dir")
    p.add_argument("--model-config", default="outputs/rl/03_hf_evidence_step400/config.json")
    p.add_argument("--expected-trajectories", type=int, default=0)
    p.add_argument("--n-rollouts", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--min-mixed-nosearch-groups", type=int, default=0)
    p.add_argument("--trajectories")
    p.add_argument("--boundary-table")
    p.add_argument("--metrics")
    p.add_argument("--output")
    p.add_argument("--trajectories-per-step", type=int, default=64)
    args = p.parse_args()
    if args.capture_dir:
        if not args.root_logits_dir:
            p.error("--capture-dir requires --root-logits-dir")
        audit_exact_capture(args)
        return
    for name in ("trajectories", "boundary_table", "metrics", "output"):
        if not getattr(args, name):
            p.error(f"historical audit requires --{name.replace('_', '-')}")

    rows = [json.loads(x) for x in Path(args.trajectories).read_text().splitlines() if x.strip()]
    boundary_doc = json.loads(Path(args.boundary_table).read_text())
    boundary = boundary_doc.get("boundary", boundary_doc)
    metrics = [json.loads(x) for x in Path(args.metrics).read_text().splitlines() if x.strip()]
    metric_by_step = {int(x["step"]): x for x in metrics}

    schema = set.intersection(*(set(x) for x in rows)) if rows else set()
    missing = sorted(FULL_REQUIRED - schema)
    unknown_ids = sorted({str(x.get("sample_id")) for x in rows if str(x.get("sample_id")) not in boundary})

    matrix: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Counter[tuple[str, str]] = Counter()
    per_step: dict[int, Counter[str]] = defaultdict(Counter)
    parity_errors: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        step = i // args.trajectories_per_step + 1
        label = boundary.get(str(row.get("sample_id")), "MISSING")
        route = str(row.get("route_first", "missing"))
        per_step[step][label] += 1
        counts[(label, route)] += 1
        matrix[label][route].append(float(row.get("response_tokens", 0)))

    for step, observed in sorted(per_step.items()):
        metric = metric_by_step.get(step, {})
        for label in ("NoSearch", "NeedSearch", "Undetermined"):
            expected = metric.get(f"boundary/n_{label}")
            if expected is not None and int(expected) != observed[label]:
                parity_errors.append(
                    {"step": step, "label": label, "observed": observed[label], "metrics": int(expected)}
                )

    length_proxy = {
        label: {route: stats(values) for route, values in sorted(routes.items())}
        for label, routes in sorted(matrix.items())
    }
    search_spans = [float(x.get("response_tokens", 0)) for x in rows if x.get("route_first") == "search"]
    internal_spans = [float(x.get("response_tokens", 0)) for x in rows if x.get("route_first") == "internal"]
    span_ratio = (
        statistics.fmean(search_spans) / statistics.fmean(internal_spans)
        if search_spans and internal_spans and statistics.fmean(internal_spans) != 0
        else None
    )

    complete = not missing and not unknown_ids and not parity_errors and len(rows) > 0
    report = {
        "gate": "OPTIMIZER_OFFLINE_READY" if complete else "OPTIMIZER_OFFLINE_DATA_INCOMPLETE",
        "n_trajectories": len(rows),
        "n_steps_inferred": len(per_step),
        "schema": sorted(schema),
        "missing_required_fields": missing,
        "unknown_boundary_ids": unknown_ids,
        "step_boundary_count_parity_errors": parity_errors,
        "boundary_route_counts": {
            f"{label}/{route}": n for (label, route), n in sorted(counts.items())
        },
        "response_span_tokens_by_boundary_route": length_proxy,
        "response_span_search_internal_mean_ratio": span_ratio,
        "warnings": [
            "response_tokens is full response-span length and includes masked observations; it is not policy_token_count",
            "no advantage or root-gradient proxy is computed unless the exact per-trajectory fields are present",
        ],
        "offline_estimators_computed": False,
        "optimizer_offline_pass": None,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
