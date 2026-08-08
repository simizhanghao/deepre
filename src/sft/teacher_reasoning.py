"""Phase 2E2: Kimi teacher for grounded rationale (JSON), not XML protocol.

Contract:
  Teacher → structured {"reasoning": "..."}  (content only)
  Code    → semantic validate + quality score
  Builder → deterministic <think>...</think> wrap
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.eval.metrics import normalize_answer
from src.sft.prototype_builder import resolve_evidence_refs, whitespace_norm

PROMPT_VERSION = "grounded_rationale_json_v2"
DEFAULT_TEACHER_MODEL = "Kimi-K2.6-CT-FP8KV"

# Structured-output schema (OpenAI-compatible json_schema / json_object).
REASONING_JSON_SCHEMA: Dict[str, Any] = {
    "name": "teacher_reasoning",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Short grounded rationale bridging the gold evidence to "
                    "the gold answer (2-4 sentences)."
                ),
            }
        },
        "required": ["reasoning"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You write short grounded rationales for multi-hop QA supervision.

You are given a question, gold supporting evidence, and the gold answer.
Your job is ONLY to write the reasoning bridge from evidence → answer.

Rules:
1. Use ONLY facts present in the supplied evidence (and the question).
2. Explicitly connect entities/relations across the evidence pieces.
3. The rationale must lead to the provided gold answer (include it naturally).
4. Do not invent facts, titles, people, places, or dates.
5. Do not mention training, gold labels, prompts, or that you are an AI.
6. Do not output XML/HTML tags of any kind.
7. Keep it concise: normally 2-4 sentences (about 30-100 words).
8. For comparison questions: name the compared attributes and the outcome.

Return a JSON object with exactly one field:
{"reasoning": "<your rationale text>"}
"""

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']{2,}|[0-9]{3,4}")
_PROPER_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_META_PATTERNS = (
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bthe user asks\b",
    r"\bwe need (to )?(answer|ensure|inspect|identify)\b",
    r"\blet's (inspect|check|see|think)\b",
    r"\btraining (example|data|sample)\b",
    r"\bgold (answer|evidence)\b",
    r"\bi (will|need to|should|must)\b",
    r"\bmy (task|goal|job) is\b",
    r"\bstep[- ]by[- ]step\b",
    r"\b<think\b",
    r"\b</think\b",
    r"\b<answer\b",
    r"\b<evidence\b",
)
_META_RE = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)

_CONNECTIVE_RE = re.compile(
    r"\b(therefore|thus|hence|because|since|so|which means|"
    r"this (implies|shows|indicates|connects)|"
    r"comparing|compared|whereas|while|both)\b",
    re.IGNORECASE,
)

_GLUE_STOP = {
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
    "according",
    "indicates",
    "implies",
    "shows",
    "connects",
    "establishes",
    "mentions",
    "refers",
}


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
        + '\nReturn JSON only: {"reasoning": "..."}'
    )


def wrap_think(reasoning: str) -> str:
    """Deterministic ECA protocol wrap — never ask the LLM to do this."""
    body = (reasoning or "").strip()
    return f"<think>\n{body}\n</think>"


def parse_teacher_json(raw: str) -> Tuple[Optional[str], List[str]]:
    """Parse teacher content into reasoning text. Returns (reasoning, errors)."""
    errors: List[str] = []
    text = (raw or "").strip()
    if not text:
        return None, ["empty teacher content"]

    # Strip common fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    obj: Any = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # salvage first {...} object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError as exc:
                return None, [f"json_decode_error: {exc}"]
        else:
            return None, ["content is not a JSON object"]

    if not isinstance(obj, dict):
        return None, ["json root is not an object"]
    if "reasoning" not in obj:
        return None, ["missing field: reasoning"]
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, str):
        return None, ["reasoning must be a string"]
    reasoning = reasoning.strip()
    if not reasoning:
        return None, ["reasoning is empty"]
    if errors:
        return reasoning, errors
    return reasoning, []


def _vocab(texts: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for t in texts:
        for w in _WORD_RE.findall(t or ""):
            out.add(w.lower())
    return out


def _content_tokens(text: str) -> Set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")} - _GLUE_STOP


def _evidence_coverage(
    reasoning: str, refs: Sequence[Dict[str, Any]]
) -> Tuple[List[bool], List[float]]:
    """Per-evidence: whether rationale shares content tokens with that evidence."""
    r_toks = _content_tokens(reasoning)
    used: List[bool] = []
    ratios: List[float] = []
    for ref in refs:
        e_toks = _content_tokens(
            f"{ref.get('title', '')} {ref.get('text', '')}"
        )
        if not e_toks:
            used.append(False)
            ratios.append(0.0)
            continue
        overlap = r_toks & e_toks
        # Require at least 2 content tokens, or 1 strong overlap if evidence tiny.
        need = 2 if len(e_toks) >= 4 else 1
        ok = len(overlap) >= need
        used.append(ok)
        ratios.append(round(len(overlap) / max(len(e_toks), 1), 4))
    return used, ratios


def score_teacher_reasoning(
    reasoning: str,
    *,
    gold_answer: str,
    question: str,
    refs: Sequence[Dict[str, Any]],
    min_words: int = 20,
    max_words: int = 150,
) -> Dict[str, Any]:
    """5-point quality score + semantic flags (no XML checks)."""
    errors: List[str] = []
    text = (reasoning or "").strip()
    n_words = len(text.split()) if text else 0

    # 1) answer consistency
    gold_n = normalize_answer(gold_answer)
    think_n = normalize_answer(text)
    answer_consistent = bool(gold_n) and gold_n in think_n
    if not answer_consistent:
        errors.append("gold answer not found in normalized reasoning")

    # 2) grounding (novel proper nouns)
    allowed = _vocab(
        [question, gold_answer]
        + [r.get("title", "") for r in refs]
        + [r.get("text", "") for r in refs]
    )
    allowed |= _GLUE_STOP
    blob = " ".join(
        whitespace_norm(x).lower()
        for x in [question, gold_answer]
        + [r.get("title", "") for r in refs]
        + [r.get("text", "") for r in refs]
    )
    novel_props: List[str] = []
    for prop in _PROPER_RE.findall(text):
        toks = [t.lower() for t in prop.split()]
        if toks and all(t not in allowed for t in toks):
            if prop.lower() not in blob:
                novel_props.append(prop)
    grounding_valid = len(novel_props) == 0
    if not grounding_valid:
        errors.append(f"ungrounded proper nouns: {novel_props[:5]}")

    # 3-4) evidence lexical coverage
    ev_used, ev_ratios = _evidence_coverage(text, refs)
    while len(ev_used) < 2:
        ev_used.append(False)
        ev_ratios.append(0.0)
    evidence1_used = bool(ev_used[0]) if refs else False
    evidence2_used = bool(ev_used[1]) if len(refs) > 1 else False
    if len(refs) >= 2 and not (evidence1_used and evidence2_used):
        errors.append(
            f"evidence_coverage incomplete: used={ev_used[:2]} ratios={ev_ratios[:2]}"
        )

    # 5) length
    length_valid = min_words <= n_words <= max_words
    if not length_valid:
        errors.append(f"word_count={n_words} not in [{min_words},{max_words}]")

    # meta / protocol pollution
    meta_hit = _META_RE.search(text)
    meta_clean = meta_hit is None
    if not meta_clean:
        errors.append(f"meta_or_protocol_phrase: {meta_hit.group(0)!r}")

    bridge_ok = bool(_CONNECTIVE_RE.search(text)) or (
        evidence1_used and evidence2_used
    )
    if not bridge_ok:
        errors.append("weak bridge: no connective and incomplete evidence use")

    score = int(answer_consistent) + int(grounding_valid) + int(evidence1_used)
    score += int(evidence2_used) + int(length_valid)
    # soft penalties (do not remove points already counted; gate via accepted)
    return {
        "answer_consistent": answer_consistent,
        "grounding_valid": grounding_valid,
        "evidence1_used": evidence1_used,
        "evidence2_used": evidence2_used,
        "evidence_coverage": ev_used[: len(refs)],
        "evidence_overlap_ratios": ev_ratios[: len(refs)],
        "length_valid": length_valid,
        "meta_clean": meta_clean,
        "bridge_ok": bridge_ok,
        "n_words": n_words,
        "novel_proper_nouns": novel_props,
        "quality_score": score,
        "errors": errors,
    }


def validate_teacher_reasoning(
    raw_output: str,
    *,
    gold_answer: str,
    question: str,
    refs: Sequence[Dict[str, Any]],
    min_words: int = 20,
    max_words: int = 150,
    min_accept_score: int = 4,
) -> Dict[str, Any]:
    """Layer1 parse JSON → Layer2 semantic score. XML format is NOT a gate."""
    reasoning, parse_errors = parse_teacher_json(raw_output)
    parse_ok = reasoning is not None and not parse_errors

    if not parse_ok:
        return {
            "parse_ok": False,
            "format_valid": False,  # legacy alias: parseable structured output
            "answer_consistent": False,
            "grounding_valid": False,
            "evidence1_used": False,
            "evidence2_used": False,
            "length_valid": False,
            "meta_clean": False,
            "bridge_ok": False,
            "n_words": 0,
            "novel_proper_nouns": [],
            "quality_score": 0,
            "errors": parse_errors or ["parse_failed"],
            "accepted": False,
            "think": None,
            "reasoning": None,
            "think_wrapped": None,
        }

    scored = score_teacher_reasoning(
        reasoning or "",
        gold_answer=gold_answer,
        question=question,
        refs=refs,
        min_words=min_words,
        max_words=max_words,
    )
    errors = list(scored["errors"])
    multi_hop_ok = len(refs) < 2 or (
        scored["evidence1_used"] and scored["evidence2_used"]
    )
    accepted = (
        scored["answer_consistent"]
        and scored["grounding_valid"]
        and scored["meta_clean"]
        and multi_hop_ok
        and scored["quality_score"] >= min_accept_score
    )
    if scored["quality_score"] < min_accept_score:
        errors.append(
            f"quality_score={scored['quality_score']}<{min_accept_score}"
        )
    if not multi_hop_ok:
        errors.append("multi_hop requires evidence1_used and evidence2_used")

    wrapped = wrap_think(reasoning or "")
    return {
        "parse_ok": True,
        "format_valid": True,  # structured parse succeeded; XML is code-owned
        "answer_consistent": scored["answer_consistent"],
        "grounding_valid": scored["grounding_valid"],
        "evidence1_used": scored["evidence1_used"],
        "evidence2_used": scored["evidence2_used"],
        "evidence_coverage": scored["evidence_coverage"],
        "evidence_overlap_ratios": scored["evidence_overlap_ratios"],
        "length_valid": scored["length_valid"],
        "meta_clean": scored["meta_clean"],
        "bridge_ok": scored["bridge_ok"],
        "n_words": scored["n_words"],
        "novel_proper_nouns": scored["novel_proper_nouns"],
        "quality_score": scored["quality_score"],
        "errors": errors,
        "accepted": accepted,
        "think": reasoning,  # bare body for coldstart builder
        "reasoning": reasoning,
        "think_wrapped": wrapped,
    }


# Back-compat alias used by older call sites / docs.
def validate_teacher_think(
    raw_output: str,
    *,
    gold_answer: str,
    question: str,
    refs: Sequence[Dict[str, Any]],
    min_words: int = 20,
    max_words: int = 150,
) -> Dict[str, Any]:
    """Deprecated name: validates JSON rationale (not XML <think>)."""
    return validate_teacher_reasoning(
        raw_output,
        gold_answer=gold_answer,
        question=question,
        refs=refs,
        min_words=min_words,
        max_words=max_words,
    )


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
