"""Phase 3D2 Capability-Aware reward.

R = R_A
  + λ_e (1 - p_int) R_E
  + λ_f R_F
  - λ_s p_int 1[N_search > 0]

p_int(q) comes from a frozen periodic capability table (tool-free n-rollout EM rate),
loaded via ECA_PINT_TABLE (JSON). Not recomputed inside each GRPO reward call.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from src.rl.reward_breakdown import combine_rewards
from src.rl.rewards_3c import (
    _as_gold_list,
    _gold_sf_keys,
    _search_count,
    _weights,
    evidence_f1_score,
    extract_answer,
    format_valid,
)
from src.eval.metrics import exact_match

_PINT_CACHE_KEY = None
_PINT_CACHE: Dict[str, float] = {}
_PINT_META: Dict[str, Any] = {}
_LOGGED_HASH = False


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _strict() -> bool:
    """Train launchers set ECA_PINT_STRICT=1 — missing sample_id is fatal."""
    return os.environ.get("ECA_PINT_STRICT", "0").strip() not in ("0", "false", "False", "")


def _default_pint() -> float:
    """Debug/selftest only when ECA_PINT_STRICT=0."""
    return _clamp01(float(os.environ.get("ECA_PINT_DEFAULT", "0.0")))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pint_table(path: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Accept {sid: float} or {sid: {p_int: float, ...}} or {\"p_int\": {...}}."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    meta: Dict[str, Any] = {}
    if isinstance(raw, dict) and isinstance(raw.get("p_int"), dict):
        meta = {k: raw[k] for k in ("mean_p_int", "histogram", "prompt_mode", "n") if k in raw}
        raw = raw["p_int"]
    out: Dict[str, float] = {}
    if not isinstance(raw, dict):
        raise ValueError(f"ECA_PINT_TABLE must be a JSON object: {path}")
    for k, v in raw.items():
        if isinstance(v, dict):
            if "p_int" not in v:
                continue
            out[str(k)] = _clamp01(float(v["p_int"]))
        else:
            out[str(k)] = _clamp01(float(v))
    try:
        resolved = str(p.resolve())
        digest = _sha256_file(p.resolve())
    except OSError:
        resolved = path
        digest = ""
    meta.update(
        {
            "path": resolved,
            "sha256": digest,
            "n_entries": len(out),
            "mean_p_int": sum(out.values()) / max(len(out), 1),
        }
    )
    return out, meta


def get_pint_table() -> Dict[str, float]:
    global _PINT_CACHE_KEY, _PINT_CACHE, _PINT_META, _LOGGED_HASH
    path = os.environ.get("ECA_PINT_TABLE", "").strip()
    if not path:
        if _strict():
            raise RuntimeError("ECA_PINT_STRICT=1 but ECA_PINT_TABLE is unset")
        return {}
    # Resolve symlinks so refresh via symlink swap is visible.
    try:
        key = str(Path(path).resolve()) + f"::{Path(path).stat().st_mtime_ns}"
    except OSError:
        key = path
    if key != _PINT_CACHE_KEY:
        _PINT_CACHE, _PINT_META = _load_pint_table(path)
        _PINT_CACHE_KEY = key
        _LOGGED_HASH = False
    if not _LOGGED_HASH:
        print(
            f"[3D2] loaded ECA_PINT_TABLE path={_PINT_META.get('path')} "
            f"sha256={_PINT_META.get('sha256')} n={_PINT_META.get('n_entries')} "
            f"mean_p_int={_PINT_META.get('mean_p_int'):.4f} strict={int(_strict())}",
            flush=True,
        )
        _LOGGED_HASH = True
    return _PINT_CACHE


def lookup_pint(sample_id: Any, extra_info: Dict[str, Any] | None = None) -> float:
    if isinstance(extra_info, dict) and extra_info.get("p_int") is not None:
        return _clamp01(float(extra_info["p_int"]))
    table = get_pint_table()
    if sample_id is None or str(sample_id).strip() == "":
        if _strict():
            raise KeyError("ECA_PINT_STRICT=1: missing sample_id in reward extra_info")
        return _default_pint()
    sid = str(sample_id)
    if sid in table:
        return table[sid]
    if _strict():
        raise KeyError(
            f"ECA_PINT_STRICT=1: sample_id={sid!r} not in ECA_PINT_TABLE "
            f"(sha256={_PINT_META.get('sha256', '?')})"
        )
    return _default_pint()


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: Dict[str, Any] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    del data_source, kwargs
    golds = _as_gold_list(ground_truth)
    pred = extract_answer(solution_str) or ""
    em = float(exact_match(pred, golds)) if pred else 0.0
    fmt = float(format_valid(solution_str))
    gold_keys = _gold_sf_keys(ground_truth, extra_info)
    ev = evidence_f1_score(solution_str, gold_keys)
    n_search = _search_count(solution_str, extra_info)
    searched = 1.0 if n_search > 0 else 0.0

    sample_id = (extra_info or {}).get("sample_id")
    p_int = lookup_pint(sample_id, extra_info)

    base_w = _weights(extra_info)
    # Gate evidence & cost; answer/format ungated.
    eff_e = float(base_w.evidence_weight) * (1.0 - p_int)
    eff_s = float(base_w.search_cost_weight) * p_int
    # Reuse combiner: cost channel = 1[N_s>0], weighted by eff_s.
    from src.rl.reward_breakdown import RewardWeights

    w = RewardWeights(
        answer_weight=base_w.answer_weight,
        evidence_weight=eff_e,
        format_weight=base_w.format_weight,
        search_cost_weight=eff_s,
        duplicate_weight=base_w.duplicate_weight,
    )
    br = combine_rewards(
        answer=em,
        evidence=ev["evidence_f1"],
        format_r=fmt,
        cost=searched,
        weights=w,
    )
    return {
        "score": br.total,
        "total_reward": br.total,
        "em": em,
        "answer_reward": br.answer_reward,
        "format": fmt,
        "format_reward": br.format_reward,
        "evidence_reward": br.evidence_reward,
        "evidence_f1": ev["evidence_f1"],
        "evidence_precision": ev["evidence_precision"],
        "evidence_recall": ev["evidence_recall"],
        "evidence_nonempty": ev["evidence_nonempty"],
        "evidence_valid": ev["evidence_valid"],
        "n_pred_evidence": ev["n_pred_evidence"],
        "n_gold_evidence": ev["n_gold_evidence"],
        "search_count": n_search,
        "search_indicator": searched,
        "cost_reward": br.cost_reward,
        "p_int": p_int,
        "eff_evidence_weight": eff_e,
        "eff_search_cost_weight": eff_s,
        "answer_weight": w.answer_weight,
        "evidence_weight": base_w.evidence_weight,
        "format_weight": w.format_weight,
        "search_cost_weight": base_w.search_cost_weight,
        "pred": pred,
        "gold": golds[0] if golds else "",
        "sample_id": sample_id,
    }
