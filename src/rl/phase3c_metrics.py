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
        ("kl", "actor/kl_loss"),
    ]
    return " | ".join(f"{lab}={metrics[k]:.4g}" for lab, k in labeled if k in metrics)
