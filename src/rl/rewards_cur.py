"""CUR outcome recorder: answer F1/EM and costs, without route utility labels.

The scalar score is answer F1 only because veRL requires a reward.  CUR never
uses this score as a mixed label: paired F1, EM and cost outcomes are persisted
separately and deployment-time lambda is applied downstream.
"""

from __future__ import annotations

from typing import Any, Dict

from src.eval.metrics import exact_match, token_f1
from src.rl.rewards_evidence import _as_gold_list, format_valid


def extract_answer(solution_str: str) -> str | None:
    """Take the final answer opening, robust to unmatched tags in user nudges."""
    text = solution_str or ""
    lower = text.lower()
    start = lower.rfind("<answer>")
    if start < 0:
        return None
    start += len("<answer>")
    end = lower.find("</answer>", start)
    if end < 0:
        return None
    answer = text[start:end].strip()
    return answer or None


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    del data_source, kwargs
    info = extra_info or {}
    golds = _as_gold_list(ground_truth)
    pred = extract_answer(solution_str) or ""
    em = float(exact_match(pred, golds)) if pred else 0.0
    f1 = float(token_f1(pred, golds)) if pred else 0.0
    fmt = float(format_valid(solution_str))
    search_count = float(info.get("search_count") or 0)
    return {
        "score": f1,
        "total_reward": f1,
        "answer_f1": f1,
        "answer_em": em,
        "em": em,
        "format": fmt,
        "search_count": search_count,
        "pred": pred,
        "gold": golds[0] if golds else "",
        "sample_id": info.get("sample_id"),
        "cur_forced_arm": info.get("cur_forced_arm"),
    }
