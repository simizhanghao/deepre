"""Phase 3C train metrics: answer/evidence/format + zero-std + search (from step 1)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def _as_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _reward_info(extra: Mapping[str, Any]) -> Mapping[str, Any]:
    info = extra.get("reward_extra_info") if isinstance(extra, Mapping) else None
    return info if isinstance(info, Mapping) else {}


def _agent_metrics(extra: Mapping[str, Any]) -> Mapping[str, Any]:
    m = extra.get("metrics") if isinstance(extra, Mapping) else None
    return m if isinstance(m, Mapping) else {}


def compute_phase3c_batch_metrics(
    extra_fields: Sequence[Mapping[str, Any]],
    *,
    uids: Optional[Sequence[Any]] = None,
    sequence_scores: Optional[Sequence[float]] = None,
    zero_std_eps: float = 1e-6,
) -> Dict[str, float]:
    n = len(extra_fields)
    if n == 0:
        return {}

    answer, evidence, fmt, total = [], [], [], []
    ev_p, ev_r, ev_f1, ev_nonempty, ev_valid = [], [], [], [], []
    finish, search_counts, dup_counts, obs_tokens = [], [], [], []
    max_hit, internal, search_route = [], [], []
    p_ints, eff_e_ws, eff_s_ws, search_inds = [], [], [], []
    boundaries: List[str] = []

    for extra in extra_fields:
        extra = extra or {}
        rinfo = _reward_info(extra)
        am = _agent_metrics(extra)

        for key, bucket, alts in (
            ("answer_reward", answer, ("em",)),
            ("evidence_reward", evidence, ("evidence_f1",)),
            ("format_reward", fmt, ("format",)),
            ("total_reward", total, ("score",)),
        ):
            val = rinfo.get(key)
            if val is None:
                for a in alts:
                    if a in rinfo:
                        val = rinfo[a]
                        break
            if val is not None:
                bucket.append(_as_float(val))

        for src, bucket in (
            ("evidence_precision", ev_p),
            ("evidence_recall", ev_r),
            ("evidence_f1", ev_f1),
            ("evidence_nonempty", ev_nonempty),
            ("evidence_valid", ev_valid),
        ):
            if src in rinfo:
                bucket.append(_as_float(rinfo[src]))

        fin = extra.get("finish", am.get("finish", 0))
        finish.append(1.0 if _as_float(fin) >= 0.5 else 0.0)

        sc = _as_float(extra.get("search_count", am.get("search_count", 0)))
        search_counts.append(sc)
        max_turns = _as_float(extra.get("max_search_turns", am.get("max_search_turns", 2)), 2.0)
        max_hit.append(1.0 if sc >= max_turns - 1e-9 else 0.0)
        dup_counts.append(_as_float(extra.get("duplicate_query_count", am.get("duplicate_query_count", 0))))
        obs_tokens.append(_as_float(extra.get("observation_tokens", am.get("observation_tokens", 0))))

        if sc > 0:
            search_route.append(1.0)
            internal.append(0.0)
        elif _as_float(am.get("used_internal", extra.get("used_internal", 0))) >= 0.5:
            search_route.append(0.0)
            internal.append(1.0)
        else:
            search_route.append(0.0)
            internal.append(0.0)

        # 3D2 capability fields (absent on 3C/3D1 → skipped)
        if "p_int" in rinfo:
            p_ints.append(_as_float(rinfo["p_int"]))
        if "eff_evidence_weight" in rinfo:
            eff_e_ws.append(_as_float(rinfo["eff_evidence_weight"]))
        if "eff_search_cost_weight" in rinfo:
            eff_s_ws.append(_as_float(rinfo["eff_search_cost_weight"]))
        if "search_indicator" in rinfo:
            search_inds.append(_as_float(rinfo["search_indicator"]))
        if "boundary" in rinfo and rinfo["boundary"] is not None:
            boundaries.append(str(rinfo["boundary"]))

    out: Dict[str, float] = {}

    def _mean_std(name: str, vals: List[float]) -> None:
        if not vals:
            return
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{name}/mean"] = float(arr.mean())
        out[f"{name}/std"] = float(arr.std())
        out[f"{name}/max"] = float(arr.max())
        out[f"{name}/min"] = float(arr.min())

    _mean_std("reward/answer_reward", answer)
    _mean_std("reward/evidence_reward", evidence)
    _mean_std("reward/format_reward", fmt)
    _mean_std("reward/total_reward", total)
    _mean_std("evidence/precision", ev_p)
    _mean_std("evidence/recall", ev_r)
    _mean_std("evidence/f1", ev_f1)
    if ev_nonempty:
        out["evidence/nonempty_rate"] = float(np.mean(ev_nonempty))
    if ev_valid:
        out["evidence/valid_rate"] = float(np.mean(ev_valid))
    if finish:
        out["agent/finish_rate"] = float(np.mean(finish))
        out["agent/format_valid_rate"] = float(np.mean(fmt)) if fmt else float(np.mean(finish))
    _mean_std("agent/search_count", search_counts)
    _mean_std("search/count", search_counts)
    if max_hit:
        out["agent/max_search_hit_rate"] = float(np.mean(max_hit))
        out["search/max_turn_hit_rate"] = float(np.mean(max_hit))
    _mean_std("agent/duplicate_query_count", dup_counts)
    _mean_std("search/duplicate", dup_counts)
    _mean_std("agent/observation_tokens", obs_tokens)
    if search_route:
        out["agent/search_rate"] = float(np.mean(search_route))
    if internal:
        out["agent/internal_rate"] = float(np.mean(internal))
    if p_ints:
        _mean_std("capability/p_int", p_ints)
        out["capability/p_int_mean"] = float(np.mean(p_ints))
    if eff_e_ws:
        _mean_std("capability/eff_evidence_weight", eff_e_ws)
    if eff_s_ws:
        _mean_std("capability/eff_search_cost_weight", eff_s_ws)
    if search_inds:
        out["capability/search_indicator_rate"] = float(np.mean(search_inds))

    # Δ_route = P(search|p_int≤0.25) − P(search|p_int≥0.75)
    if p_ints and search_route and len(p_ints) == len(search_route):
        low = [s for p, s in zip(p_ints, search_route) if p <= 0.25 + 1e-9]
        high = [s for p, s in zip(p_ints, search_route) if p >= 0.75 - 1e-9]
        buckets = {
            "0.00": [],
            "0.25": [],
            "0.50": [],
            "0.75": [],
            "1.00": [],
        }
        for p, s in zip(p_ints, search_route):
            key = f"{round(p * 4) / 4:.2f}"
            if key in buckets:
                buckets[key].append(s)
        for bk, vals in buckets.items():
            if vals:
                out[f"capability/search_rate_p{bk}"] = float(np.mean(vals))
                out[f"capability/n_p{bk}"] = float(len(vals))
        if low and high:
            out["capability/search_rate_low_pint"] = float(np.mean(low))
            out["capability/search_rate_high_pint"] = float(np.mean(high))
            out["capability/delta_route"] = float(np.mean(low) - np.mean(high))

    # 3D2b: Δ_boundary = P(search|NeedSearch) − P(search|NoSearch)
    if boundaries and search_route and len(boundaries) == len(search_route):
        by_lab: dict[str, list[float]] = defaultdict(list)
        for lab, s in zip(boundaries, search_route):
            by_lab[lab].append(s)
        for lab, vals in by_lab.items():
            if vals:
                out[f"boundary/search_rate_{lab}"] = float(np.mean(vals))
                out[f"boundary/n_{lab}"] = float(len(vals))
        need = by_lab.get("NeedSearch") or []
        nos = by_lab.get("NoSearch") or []
        if need and nos:
            out["boundary/delta_boundary"] = float(np.mean(need) - np.mean(nos))
        n_b = float(len(boundaries))
        out["boundary/frac_NoSearch"] = float(len(nos) / n_b)
        out["boundary/frac_NeedSearch"] = float(len(need) / n_b)
        out["boundary/frac_Undetermined"] = float(
            len(by_lab.get("Undetermined") or []) / n_b
        )

    # zero-std by question group (uid), not whole-batch std
    scores = list(sequence_scores) if sequence_scores is not None else list(total)
    if uids is not None and scores and len(uids) == len(scores):
        by_uid: dict[Any, list[float]] = defaultdict(list)
        for u, s in zip(uids, scores):
            by_uid[u].append(float(s))
        group_stds = []
        zero = 0
        for vals in by_uid.values():
            if len(vals) < 2:
                continue
            std = float(np.std(vals))
            group_stds.append(std)
            if std <= zero_std_eps:
                zero += 1
        if group_stds:
            out["grpo/zero_std_group_rate"] = zero / len(group_stds)
            out["grpo/group_reward_std/mean"] = float(np.mean(group_stds))
            out["grpo/num_groups"] = float(len(group_stds))

    out["phase3c/num_trajectories"] = float(n)
    return out


def summarize_console_line(metrics: Mapping[str, float]) -> str:
    labeled = [
        ("answer", "reward/answer_reward/mean"),
        ("evidence", "reward/evidence_reward/mean"),
        ("format", "reward/format_reward/mean"),
        ("total", "reward/total_reward/mean"),
        ("total_std", "reward/total_reward/std"),
        ("zero_std", "grpo/zero_std_group_rate"),
        ("finish", "agent/finish_rate"),
        ("search", "agent/search_count/mean"),
        ("search_rate", "agent/search_rate"),
        ("ev_f1", "evidence/f1/mean"),
        ("ev_nz", "evidence/nonempty_rate"),
        ("p_int", "capability/p_int_mean"),
        ("d_route", "capability/delta_route"),
        ("d_bnd", "boundary/delta_boundary"),
        ("sr_need", "boundary/search_rate_NeedSearch"),
        ("sr_no", "boundary/search_rate_NoSearch"),
        ("kl", "actor/kl_loss"),
    ]
    return " | ".join(f"{lab}={metrics[k]:.4g}" for lab, k in labeled if k in metrics)
