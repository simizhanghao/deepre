"""Phase 2E2: Kimi teacher for grounded <think> only (hard multi-hop).

Locks gold evidence + gold answer; teacher may only fill the reasoning bridge.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.eval.metrics import normalize_answer
from src.sft.prototype_builder import resolve_evidence_refs, whitespace_norm

PROMPT_VERSION = "grounded_reasoning_v1"
DEFAULT_TEACHER_MODEL = "Kimi-K2.6-CT-FP8KV"

SYSTEM_PROMPT = """You are generating supervised reasoning data for a multi-hop question-answering model.

Use ONLY the provided supporting evidence.

Requirements:
1. Derive the provided gold answer from the evidence.
2. Explicitly connect the entities or relations needed across the evidence.
3. Do not introduce facts not present in the evidence.
4. Do not question or change the gold answer.
5. Do not repeat the evidence verbatim unless necessary.
6. Keep the reasoning concise, normally 2-4 sentences.
7. Do not mention that this is a gold answer or training example.
8. Output exactly one <think>...</think> block and nothing else.
9. For comparison questions, explicitly identify the two attributes being compared and perform the comparison."""

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_FORBIDDEN_TAGS = ("<answer", "<evidence", "<search", "<internal", "<observation")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']{2,}|[0-9]{3,4}")
_PROPER_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")


def format_teacher_user_prompt(
    sample: Dict[str, Any],
    refs: Sequence[Dict[str, Any]],
    gold_answer: str,
) -> str:
    blocks = []
    for i, ref in enumerate(refs, 1):
        blocks.append(
            f"[Evidence {i}]\n"
            f"Title: {ref['title']}\n"
            f"Sentence: {whitespace_norm(ref['text'])}"
        )
    qtype = (sample.get("metadata") or {}).get("type") or "bridge"
    extra = ""
    if str(qtype).lower() == "comparison":
        extra = (
            "\nThis is a comparison question: identify the compared attributes "
            "and state the comparison outcome.\n"
        )
    return (
        f"Question:\n{sample['question']}\n\n"
        f"Gold answer:\n{gold_answer}\n\n"
        f"Gold supporting evidence:\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
        + "Task:\nWrite a short reasoning bridge that derives the gold answer "
        "using ONLY the supplied evidence.\n"
        + extra
        + "\nOutput exactly one <think>...</think> block."
    )


def extract_think(raw: str) -> Optional[str]:
    matches = list(_THINK_RE.finditer(raw or ""))
    if len(matches) != 1:
        return None
    body = matches[0].group(1).strip()
    return body or None


def _vocab(texts: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for t in texts:
        for w in _WORD_RE.findall(t or ""):
            out.add(w.lower())
    return out


def validate_teacher_think(
    raw_output: str,
    *,
    gold_answer: str,
    question: str,
    refs: Sequence[Dict[str, Any]],
    min_words: int = 18,
    max_words: int = 180,
) -> Dict[str, Any]:
    """Deterministic checks. Returns validation dict + extracted think body."""
    raw = (raw_output or "").strip()
    lower = raw.lower()
    format_valid = True
    errors: List[str] = []

    think = extract_think(raw)
    if think is None:
        format_valid = False
        errors.append("need exactly one non-empty <think>...</think>")
    # nothing outside the think block (allow whitespace)
    if think is not None:
        stripped = _THINK_RE.sub("", raw).strip()
        if stripped:
            format_valid = False
            errors.append("extra text outside <think> block")
    for tag in _FORBIDDEN_TAGS:
        if tag in lower:
            format_valid = False
            errors.append(f"forbidden tag fragment: {tag}")

    answer_consistent = False
    grounding_valid = False
    n_words = 0
    length_valid = False
    novel_props: List[str] = []

    if think is not None:
        n_words = len(think.split())
        length_valid = min_words <= n_words <= max_words
        if not length_valid:
            errors.append(f"word_count={n_words} not in [{min_words},{max_words}]")

        gold_n = normalize_answer(gold_answer)
        think_n = normalize_answer(think)
        # Prefer gold span in reasoning; also accept final sentence containing gold.
        answer_consistent = bool(gold_n) and gold_n in think_n
        if not answer_consistent:
            errors.append("gold answer not found in normalized think text")

        allowed = _vocab(
            [question, gold_answer]
            + [r.get("title", "") for r in refs]
            + [r.get("text", "") for r in refs]
        )
        # Common glue words / connective lexicon — ignore for grounding.
        stop = {
            "the",
            "and",
            "that",
            "this",
            "with",
            "from",
            "therefore",
            "thus",
            "hence",
            "because",
            "since",
            "which",
            "where",
            "when",
            "who",
            "whom",
            "whose",
            "what",
            "first",
            "second",
            "third",
            "evidence",
            "fact",
            "facts",
            "states",
            "state",
            "identifies",
            "identify",
            "requested",
            "answer",
            "born",
            "birthplace",
            "director",
            "directed",
            "comparing",
            "comparison",
            "attribute",
            "attributes",
            "larger",
            "smaller",
            "earlier",
            "later",
            "older",
            "younger",
            "more",
            "less",
            "both",
            "between",
            "across",
            "using",
            "only",
            "provided",
            "supporting",
        }
        allowed |= stop
        for prop in _PROPER_RE.findall(think):
            toks = [t.lower() for t in prop.split()]
            # novel if none of the tokens appear in allowed vocab
            if toks and all(t not in allowed for t in toks):
                # allow if full phrase normalized is inside evidence blob
                blob = " ".join(
                    whitespace_norm(x).lower()
                    for x in [question, gold_answer]
                    + [r.get("title", "") for r in refs]
                    + [r.get("text", "") for r in refs]
                )
                if prop.lower() not in blob:
                    novel_props.append(prop)
        grounding_valid = len(novel_props) == 0
        if not grounding_valid:
            errors.append(f"ungrounded proper nouns: {novel_props[:5]}")

    ok = format_valid and answer_consistent and grounding_valid and length_valid
    return {
        "format_valid": format_valid,
        "answer_consistent": answer_consistent,
        "grounding_valid": grounding_valid,
        "length_valid": length_valid,
        "n_words": n_words,
        "novel_proper_nouns": novel_props,
        "errors": errors,
        "accepted": ok,
        "think": think,
    }


def mine_hard_candidates(
    *,
    samples_by_id: Dict[str, Dict[str, Any]],
    direct: Dict[str, Dict[str, Any]],
    base_oracle: Dict[str, Dict[str, Any]],
    sft_oracle: Dict[str, Dict[str, Any]],
    seed: int = 42,
    n_persistent: int = 320,
    n_other: int = 80,
) -> Tuple[List[str], Dict[str, Any]]:
    """Prefer persistent C-like hard; fill remainder from other v0-oracle-wrong."""
    import random

    def d_ok(sid: str) -> bool:
        r = direct.get(sid) or {}
        return bool(r.get("direct_correct")) or float(r.get("exact_match") or 0) >= 1.0 - 1e-9

    def o_ok(table: Dict[str, Dict[str, Any]], sid: str) -> bool:
        r = table.get(sid) or {}
        em = r.get("exact_match")
        if em is None and isinstance(r.get("metrics"), dict):
            em = r["metrics"].get("exact_match")
        return float(em or 0) >= 1.0 - 1e-9

    persistent: List[str] = []
    other_hard: List[str] = []
    for sid, sample in samples_by_id.items():
        if sid not in sft_oracle or o_ok(sft_oracle, sid):
            continue
        try:
            refs = resolve_evidence_refs(sample)
        except Exception:
            continue
        if len(refs) < 2:
            continue
        if (not d_ok(sid)) and (not o_ok(base_oracle, sid)):
            persistent.append(sid)
        else:
            other_hard.append(sid)

    rng = random.Random(seed)
    rng.shuffle(persistent)
    rng.shuffle(other_hard)
    chosen = persistent[:n_persistent]
    need = max(0, n_persistent + n_other - len(chosen))
    chosen.extend(other_hard[:need])
    # if persistent shortfall, top up more from other
    if len(chosen) < n_persistent + n_other:
        rest = [s for s in other_hard if s not in chosen]
        chosen.extend(rest[: (n_persistent + n_other - len(chosen))])

    stats = {
        "n_persistent_available": len(persistent),
        "n_other_hard_available": len(other_hard),
        "n_chosen": len(chosen),
        "n_chosen_persistent": sum(1 for s in chosen if s in set(persistent)),
        "n_chosen_other": sum(1 for s in chosen if s not in set(persistent)),
        "n_persistent_target": n_persistent,
        "n_other_target": n_other,
    }
    return chosen, stats


def oracle_em_map_from_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sid = r["sample_id"]
        em = float((r.get("metrics") or {}).get("exact_match", 0) or 0)
        out[sid] = {"sample_id": sid, "exact_match": em, "metrics": r.get("metrics")}
    return out
