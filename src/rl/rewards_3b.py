"""Phase 3B reward: R = EM + 0.1 * Format. Evidence/cost OFF."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Union

from src.eval.metrics import exact_match, normalize_answer

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_INTERNAL_RE = re.compile(r"<internal>(.*?)</internal>", re.DOTALL | re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL | re.IGNORECASE)


def extract_answer(solution_str: str) -> str | None:
    matches = list(_ANSWER_RE.finditer(solution_str or ""))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _as_gold_list(ground_truth: Any) -> List[str]:
    if ground_truth is None:
        return []
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target", ground_truth.get("gold_answers"))
        return _as_gold_list(target)
    if isinstance(ground_truth, str):
        return [ground_truth]
    if isinstance(ground_truth, Sequence):
        return [str(x) for x in ground_truth]
    return [str(ground_truth)]


def format_valid(solution_str: str) -> float:
    """Lightweight structural format check for 3B (0/1).

    Requires a non-empty <answer>. Soft checks: if both <search> and <internal>
    appear before first answer, format=0 (protocol violation).
    """
    text = solution_str or ""
    ans = extract_answer(text)
    if not ans:
        return 0.0
    # Truncate to first answer for routing conflict check.
    cut = text[: _ANSWER_RE.search(text).end()]  # type: ignore[union-attr]
    has_search = bool(_SEARCH_RE.search(cut))
    has_internal = bool(_INTERNAL_RE.search(cut))
    if has_search and has_internal:
        return 0.0
    return 1.0


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: Dict[str, Any] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """veRL custom reward entrypoint.

    Returns dict so NaiveRewardManager logs EM/format components.
    score = EM + 0.1 * format
    """
    del data_source, kwargs  # unused; keep signature compatible
    golds = _as_gold_list(ground_truth)
    pred = extract_answer(solution_str) or ""
    em = float(exact_match(pred, golds)) if pred else 0.0
    fmt = float(format_valid(solution_str))
    score = em + 0.1 * fmt
    # Dual names: veRL NaiveRewardManager keeps all dict keys in reward_extra_info;
    # answer_*/format_*/total_* are the Phase-3B2 TensorBoard names.
    info = {
        "score": score,
        "em": em,
        "format": fmt,
        "answer_reward": em,
        "format_reward": fmt,
        "total_reward": score,
        "pred": pred,
        "gold": golds[0] if golds else "",
        "has_evidence_tag": 1.0 if _EVIDENCE_RE.search(solution_str or "") else 0.0,
        "sample_id": (extra_info or {}).get("sample_id"),
    }
    return info
