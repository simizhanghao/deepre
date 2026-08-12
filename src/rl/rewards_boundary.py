"""Search-Boundary-Aware ECA reward.

Boundary S(q) ∈ {NoSearch, NeedSearch, Undetermined} from frozen table
(ECA_BOUNDARY_TABLE), built via dual search-disabled / search-enabled probes.

  NoSearch:      R = R_A + 0.1 R_F − α 1[N_s>0]     (Evidence OFF)
  NeedSearch:    R = R_A + λ_e R_E + 0.1 R_F         (no search cost)
  Undetermined:  R = R_A + λ_e R_E + 0.1 R_F         (no search cost)

α from ECA_SEARCH_COST_WEIGHT (default 0.30).
λ_e from ECA_EVIDENCE_WEIGHT (default 0.5).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from src.eval.metrics import exact_match
from src.rl.reward_breakdown import RewardWeights, combine_rewards
from src.rl.rewards_evidence import (
    _as_gold_list,
    _gold_sf_keys,
    _search_count,
    _weights,
    evidence_f1_score,
    extract_answer,
    format_valid,
)

_BOUNDARY_CACHE_KEY = None
_BOUNDARY_CACHE: Dict[str, str] = {}
_BOUNDARY_META: Dict[str, Any] = {}
_LOGGED_HASH = False

VALID_LABELS = ("NoSearch", "NeedSearch", "Undetermined")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _strict() -> bool:
    return os.environ.get("ECA_BOUNDARY_STRICT", "0").strip() not in (
        "0",
        "false",
        "False",
        "",
    )


def _default_label() -> str:
    lab = os.environ.get("ECA_BOUNDARY_DEFAULT", "Undetermined").strip()
    return lab if lab in VALID_LABELS else "Undetermined"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_label(v: Any) -> str:
    if isinstance(v, dict):
        v = v.get("boundary") or v.get("S") or v.get("label")
    s = str(v).strip()
    aliases = {
        "nosearch": "NoSearch",
        "no_search": "NoSearch",
        "needsearch": "NeedSearch",
        "need_search": "NeedSearch",
        "undetermined": "Undetermined",
    }
    key = s.replace(" ", "").replace("-", "_").lower()
    if key in aliases:
        return aliases[key]
    if s in VALID_LABELS:
        return s
    raise ValueError(f"unknown boundary label: {v!r}")


def _load_boundary_table(path: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    meta: Dict[str, Any] = {}
    mapping = raw
    if isinstance(raw, dict) and isinstance(raw.get("boundary"), dict):
        meta = {
            k: raw[k]
            for k in ("histogram", "delta", "n", "prompt_mode", "model_path")
            if k in raw
        }
        mapping = raw["boundary"]
    out: Dict[str, str] = {}
    if not isinstance(mapping, dict):
        raise ValueError(f"ECA_BOUNDARY_TABLE must be a JSON object: {path}")
    for k, v in mapping.items():
        out[str(k)] = _normalize_label(v)
    try:
        resolved = str(p.resolve())
        digest = _sha256_file(p.resolve())
    except OSError:
        resolved = path
        digest = ""
    hist: Dict[str, int] = {lab: 0 for lab in VALID_LABELS}
    for lab in out.values():
        hist[lab] = hist.get(lab, 0) + 1
    meta.update(
        {
            "path": resolved,
            "sha256": digest,
            "n_entries": len(out),
            "histogram": hist,
        }
    )
    return out, meta


def get_boundary_table() -> Dict[str, str]:
    global _BOUNDARY_CACHE_KEY, _BOUNDARY_CACHE, _BOUNDARY_META, _LOGGED_HASH
    path = os.environ.get("ECA_BOUNDARY_TABLE", "").strip()
    if not path:
        if _strict():
            raise RuntimeError("ECA_BOUNDARY_STRICT=1 but ECA_BOUNDARY_TABLE is unset")
        return {}
    try:
        key = str(Path(path).resolve()) + f"::{Path(path).stat().st_mtime_ns}"
    except OSError:
        key = path
    if key != _BOUNDARY_CACHE_KEY:
        _BOUNDARY_CACHE, _BOUNDARY_META = _load_boundary_table(path)
        _BOUNDARY_CACHE_KEY = key
        _LOGGED_HASH = False
    if not _LOGGED_HASH:
        print(
            f"[boundary] loaded ECA_BOUNDARY_TABLE path={_BOUNDARY_META.get('path')} "
            f"sha256={_BOUNDARY_META.get('sha256')} n={_BOUNDARY_META.get('n_entries')} "
            f"hist={_BOUNDARY_META.get('histogram')} strict={int(_strict())}",
            flush=True,
        )
        _LOGGED_HASH = True
    return _BOUNDARY_CACHE


def lookup_boundary(sample_id: Any, extra_info: Dict[str, Any] | None = None) -> str:
    if isinstance(extra_info, dict) and extra_info.get("boundary") is not None:
        return _normalize_label(extra_info["boundary"])
    table = get_boundary_table()
    if sample_id is None or str(sample_id).strip() == "":
        if _strict():
            raise KeyError("ECA_BOUNDARY_STRICT=1: missing sample_id in reward extra_info")
        return _default_label()
    sid = str(sample_id)
    if sid in table:
        return table[sid]
    if _strict():
        raise KeyError(
            f"ECA_BOUNDARY_STRICT=1: sample_id={sid!r} not in ECA_BOUNDARY_TABLE "
            f"(sha256={_BOUNDARY_META.get('sha256', '?')})"
        )
    return _default_label()


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
    boundary = lookup_boundary(sample_id, extra_info)

    base_w = _weights(extra_info)
    root_pivot = os.environ.get("ECA_ROOT_PIVOT", "0").strip().lower() in {"1", "true", "yes"}
    if root_pivot:
        # Routing utility is isolated in the root-token loss. Task credit keeps
        # Answer/Evidence/Format for both classes and never broadcasts cost.
        eff_e = float(base_w.evidence_weight)
        eff_s = 0.0
    elif boundary == "NoSearch":
        eff_e = 0.0
        eff_s = float(base_w.search_cost_weight)
    else:
        eff_e = float(base_w.evidence_weight)
        eff_s = 0.0

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
        "boundary": boundary,
        "eff_evidence_weight": eff_e,
        "eff_search_cost_weight": eff_s,
        "answer_weight": w.answer_weight,
        "evidence_weight": base_w.evidence_weight,
        "format_weight": w.format_weight,
        "search_cost_weight": base_w.search_cost_weight,
        "root_pivot_task_reward": float(root_pivot),
        "pred": pred,
        "gold": golds[0] if golds else "",
        "sample_id": sample_id,
    }
