"""Phase 2E2: coldstart_v1 mixture (routing + BM25 hard-neg + teacher think)."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.sft.coldstart_builder import assert_train_only, load_frozen_ids
from src.sft.prototype_builder import (
    AGENT_SYSTEM_PROMPT,
    base_row,
    build_internal,
    build_search_format,
    format_documents_for_user,
    format_evidence_block,
    gold_answer_of,
    gold_titles_covered,
    make_sft_id,
    resolve_evidence_refs,
    validate_sft_row,
)

BUILDER_NAME = "phase2e2_coldstart_builder_v1"

DEFAULT_TARGETS_V1 = {
    "internal": 950,
    "search_required": 850,
    "evidence_bm25": 1150,
    "evidence_reasoning": 1200,
    "search_format": 400,
}


def _direct_ok(label: Dict[str, Any]) -> bool:
    return bool(label.get("direct_correct")) or float(
        label.get("exact_match") or 0
    ) >= 1.0 - 1e-9


def _oracle_ok(row: Dict[str, Any]) -> bool:
    em = row.get("exact_match")
    if em is None and isinstance(row.get("metrics"), dict):
        em = row["metrics"].get("exact_match")
    return float(em or 0) >= 1.0 - 1e-9


def build_internal_v1(
    sample: Dict[str, Any], seed: int, label_row: Dict[str, Any]
) -> Dict[str, Any]:
    row = build_internal(sample, taxonomy_label="direct_correct", seed=seed)
    row["provenance"]["builder"] = BUILDER_NAME
    row["provenance"]["direct_label"] = {
        "exact_match": label_row.get("exact_match"),
        "token_f1": label_row.get("token_f1"),
        "prediction": label_row.get("prediction"),
    }
    row["metadata"]["phase"] = "2E2"
    row["metadata"]["mix_tag"] = "internal_direct_correct_v1"
    row["metadata"]["source_split"] = "train"
    row["sft_id"] = make_sft_id(sample["sample_id"], "internal", "direct_v1")
    return row


def build_search_required_v1(
    sample: Dict[str, Any],
    seed: int,
    retrieval_row: Dict[str, Any],
) -> Dict[str, Any]:
    row = build_search_format(sample, "search_required", seed, retrieval_row)
    row["category"] = "search_format"
    row["provenance"]["builder"] = BUILDER_NAME
    row["provenance"]["routing_label"] = "direct_wrong_oracle_correct"
    row["metadata"]["phase"] = "2E2"
    row["metadata"]["mix_tag"] = "search_required_v1"
    row["metadata"]["source_split"] = "train"
    row["sft_id"] = make_sft_id(sample["sample_id"], "search_format", "req_v1")
    return row


def build_search_format_v1(
    sample: Dict[str, Any],
    seed: int,
    retrieval_row: Dict[str, Any],
) -> Dict[str, Any]:
    row = build_search_format(sample, "train", seed, retrieval_row)
    row["provenance"]["builder"] = BUILDER_NAME
    row["metadata"]["phase"] = "2E2"
    row["metadata"]["mix_tag"] = "search_format_protocol_v1"
    row["metadata"]["source_split"] = "train"
    row["sft_id"] = make_sft_id(sample["sample_id"], "search_format", "cand_v1")
    return row


def build_evidence_bm25_v1(
    sample: Dict[str, Any],
    seed: int,
    retrieval_row: Dict[str, Any],
) -> Dict[str, Any]:
    refs = resolve_evidence_refs(sample)
    docs = list(retrieval_row.get("documents") or [])
    if not docs:
        raise ValueError(f"{sample['sample_id']}: empty BM25 docs")
    gold = gold_answer_of(sample)
    evidence_block = format_evidence_block(refs)
    target = (
        f"<evidence>\n{evidence_block}\n</evidence>\n"
        f"<answer>\n{gold}\n</answer>"
    )
    user = (
        f"Question: {sample['question']}\n\n"
        f"Documents:\n{format_documents_for_user(docs)}"
    )
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    retr = dict(retrieval_row.get("retriever") or {})
    retr.setdefault("name", "bm25s")
    retr.setdefault("scope", "candidate")
    row = base_row(
        sample,
        category="evidence",
        taxonomy_label="train",
        messages=messages,
        target=target,
        evidence_refs=refs,
        documents=docs,
        provenance={
            "supporting_facts": list(sample.get("supporting_facts") or []),
            "builder": BUILDER_NAME,
            "teacher_id": None,
            "retriever": retr,
            "reasoning_source": None,
            "context_view": "bm25_hardneg",
        },
        metadata={
            "phase": "2E2",
            "mix_tag": "evidence_bm25_hardneg_v1",
            "seed": seed,
            "level": (sample.get("metadata") or {}).get("level"),
            "type": (sample.get("metadata") or {}).get("type"),
            "observation_in_target": False,
            "source_split": "train",
            "context_view": "bm25_hardneg",
            "n_input_docs": len(docs),
        },
    )
    row["sft_id"] = make_sft_id(sample["sample_id"], "evidence", "bm25_v1")
    return row


def build_evidence_reasoning_teacher_v1(
    sample: Dict[str, Any],
    seed: int,
    teacher_row: Dict[str, Any],
    *,
    retrieval_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    refs = list(teacher_row.get("evidence_refs") or resolve_evidence_refs(sample))
    think = (teacher_row.get("think") or "").strip()
    if not think:
        raise ValueError(f"{sample['sample_id']}: empty teacher think")
    gold = gold_answer_of(sample)
    evidence_block = format_evidence_block(refs)
    # Prefer BM25 hard-neg context when available and covers gold titles.
    docs: List[Dict[str, Any]]
    context_view = "oracle_teacher"
    retr: Dict[str, Any] = {"name": "oracle", "scope": "oracle_supporting_docs"}
    if retrieval_row and gold_titles_covered(sample, retrieval_row):
        docs = list(retrieval_row.get("documents") or [])
        context_view = "bm25_hardneg_teacher"
        retr = dict(retrieval_row.get("retriever") or {})
        retr.setdefault("name", "bm25s")
        retr.setdefault("scope", "candidate")
    else:
        from src.sft.prototype_builder import oracle_documents

        docs = oracle_documents(sample)

    target = (
        f"<evidence>\n{evidence_block}\n</evidence>\n"
        f"<think>\n{think}\n</think>\n"
        f"<answer>\n{gold}\n</answer>"
    )
    user = (
        f"Question: {sample['question']}\n\n"
        f"Documents:\n{format_documents_for_user(docs)}"
    )
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    tv = teacher_row.get("teacher_validation") or {}
    row = base_row(
        sample,
        category="evidence_reasoning",
        taxonomy_label="hard",
        messages=messages,
        target=target,
        evidence_refs=refs,
        documents=docs,
        provenance={
            "supporting_facts": list(sample.get("supporting_facts") or []),
            "builder": BUILDER_NAME,
            "teacher_id": teacher_row.get("teacher_model"),
            "teacher_model": teacher_row.get("teacher_model"),
            "teacher_prompt_version": teacher_row.get("teacher_prompt_version"),
            "retriever": retr,
            "reasoning_source": "kimi2.6",
            "context_view": context_view,
            "teacher_validation": {
                "format_valid": tv.get("format_valid"),
                "answer_consistent": tv.get("answer_consistent"),
                "grounding_valid": tv.get("grounding_valid"),
                "accepted": tv.get("accepted"),
            },
        },
        metadata={
            "phase": "2E2",
            "mix_tag": "evidence_reasoning_kimi_hard_v1",
            "seed": seed,
            "level": (sample.get("metadata") or {}).get("level"),
            "type": (sample.get("metadata") or {}).get("type"),
            "observation_in_target": False,
            "source_split": "train",
            "context_view": context_view,
            "teacher_pool": teacher_row.get("pool"),
        },
    )
    row["sft_id"] = make_sft_id(sample["sample_id"], "evidence_reasoning", "kimi_v1")
    return row


def build_evidence_reasoning_template_fill(
    sample: Dict[str, Any],
    seed: int,
    retrieval_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Non-teacher fill for remaining evidence_reasoning quota (still uses <think>)."""
    from src.sft.prototype_builder import build_reasoning, oracle_documents

    refs = resolve_evidence_refs(sample)
    gold = gold_answer_of(sample)
    reasoning = build_reasoning(refs, gold)
    evidence_block = format_evidence_block(refs)
    if retrieval_row and gold_titles_covered(sample, retrieval_row):
        docs = list(retrieval_row.get("documents") or [])
        context_view = "bm25_hardneg"
        retr = dict(retrieval_row.get("retriever") or {})
        retr.setdefault("name", "bm25s")
        retr.setdefault("scope", "candidate")
    else:
        docs = oracle_documents(sample)
        context_view = "oracle"
        retr = {"name": "oracle", "scope": "oracle_supporting_docs"}
    target = (
        f"<evidence>\n{evidence_block}\n</evidence>\n"
        f"<think>\n{reasoning}\n</think>\n"
        f"<answer>\n{gold}\n</answer>"
    )
    user = (
        f"Question: {sample['question']}\n\n"
        f"Documents:\n{format_documents_for_user(docs)}"
    )
    row = base_row(
        sample,
        category="evidence_reasoning",
        taxonomy_label="train",
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        target=target,
        evidence_refs=refs,
        documents=docs,
        provenance={
            "supporting_facts": list(sample.get("supporting_facts") or []),
            "builder": BUILDER_NAME,
            "teacher_id": None,
            "retriever": retr,
            "reasoning_source": "template_v0",
            "context_view": context_view,
        },
        metadata={
            "phase": "2E2",
            "mix_tag": f"evidence_reasoning_{context_view}_template_v1",
            "seed": seed,
            "level": (sample.get("metadata") or {}).get("level"),
            "type": (sample.get("metadata") or {}).get("type"),
            "observation_in_target": False,
            "source_split": "train",
            "context_view": context_view,
        },
    )
    row["sft_id"] = make_sft_id(
        sample["sample_id"], "evidence_reasoning", f"{context_view}_tmpl_v1"
    )
    return row


def assign_coldstart_v1(
    train_samples: Sequence[Dict[str, Any]],
    *,
    frozen_ids: Set[str],
    direct_labels: Dict[str, Dict[str, Any]],
    base_oracle: Dict[str, Dict[str, Any]],
    retrieval: Dict[str, Dict[str, Any]],
    teacher_accepted: Dict[str, Dict[str, Any]],
    seed: int = 42,
    targets: Optional[Dict[str, int]] = None,
    n_teacher_reasoning: int = 400,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    targets = dict(targets or DEFAULT_TARGETS_V1)
    assert_train_only(train_samples, frozen_ids)

    rng = random.Random(seed)
    by_id = {s["sample_id"]: s for s in train_samples}
    all_ids = sorted(by_id.keys())
    rng.shuffle(all_ids)

    internal_ids = [sid for sid in all_ids if _direct_ok(direct_labels.get(sid, {}))]
    rng.shuffle(internal_ids)

    search_req_ids = [
        sid
        for sid in all_ids
        if (not _direct_ok(direct_labels.get(sid, {})))
        and _oracle_ok(base_oracle.get(sid, {}))
        and sid in retrieval
        and gold_titles_covered(by_id[sid], retrieval[sid])
    ]
    rng.shuffle(search_req_ids)

    bm25_evidence_ids = [
        sid
        for sid in all_ids
        if sid in retrieval and (retrieval[sid].get("documents") or [])
    ]
    rng.shuffle(bm25_evidence_ids)

    search_fmt_ids = [
        sid
        for sid in all_ids
        if sid in retrieval and gold_titles_covered(by_id[sid], retrieval[sid])
    ]
    rng.shuffle(search_fmt_ids)

    teacher_ids = [sid for sid in teacher_accepted if sid in by_id]
    rng.shuffle(teacher_ids)

    used: Set[str] = set()
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    def try_add(row: Dict[str, Any]) -> bool:
        errs = validate_sft_row(row)
        if errs:
            rejected.append({"sft_id": row.get("sft_id"), "errors": errs})
            return False
        accepted.append(row)
        return True

    def count_mix(tag: str) -> int:
        return sum(1 for r in accepted if (r.get("metadata") or {}).get("mix_tag") == tag)

    def count_cat(cat: str) -> int:
        return sum(1 for r in accepted if r["category"] == cat)

    # P0 internal
    for sid in internal_ids:
        if count_cat("internal") >= targets["internal"]:
            break
        if sid in used:
            continue
        if try_add(build_internal_v1(by_id[sid], seed, direct_labels[sid])):
            used.add(sid)

    # P0 search-required
    for sid in search_req_ids:
        if count_mix("search_required_v1") >= targets["search_required"]:
            break
        if sid in used:
            continue
        try:
            row = build_search_required_v1(by_id[sid], seed, retrieval[sid])
        except Exception as exc:  # noqa: BLE001
            rejected.append({"sft_id": f"{sid}__search_req", "errors": [str(exc)]})
            continue
        if try_add(row):
            used.add(sid)

    # P2 teacher reasoning first (subset of evidence_reasoning)
    n_teacher = 0
    for sid in teacher_ids:
        if n_teacher >= n_teacher_reasoning:
            break
        if count_cat("evidence_reasoning") >= targets["evidence_reasoning"]:
            break
        if sid in used:
            continue
        trow = teacher_accepted[sid]
        if not (trow.get("teacher_validation") or {}).get("accepted"):
            continue
        try:
            row = build_evidence_reasoning_teacher_v1(
                by_id[sid],
                seed,
                trow,
                retrieval_row=retrieval.get(sid),
            )
        except Exception as exc:  # noqa: BLE001
            rejected.append({"sft_id": f"{sid}__kimi", "errors": [str(exc)]})
            continue
        if try_add(row):
            used.add(sid)
            n_teacher += 1

    # P1 BM25 hard-neg evidence (no reasoning)
    for sid in bm25_evidence_ids:
        if count_mix("evidence_bm25_hardneg_v1") >= targets["evidence_bm25"]:
            break
        if sid in used:
            continue
        try:
            row = build_evidence_bm25_v1(by_id[sid], seed, retrieval[sid])
        except Exception as exc:  # noqa: BLE001
            rejected.append({"sft_id": f"{sid}__bm25_ev", "errors": [str(exc)]})
            continue
        if try_add(row):
            used.add(sid)

    # Fill remaining evidence_reasoning with template (non-teacher)
    for sid in all_ids:
        if count_cat("evidence_reasoning") >= targets["evidence_reasoning"]:
            break
        if sid in used:
            continue
        try:
            row = build_evidence_reasoning_template_fill(
                by_id[sid], seed, retrieval.get(sid)
            )
        except Exception as exc:  # noqa: BLE001
            rejected.append({"sft_id": f"{sid}__er_tmpl", "errors": [str(exc)]})
            continue
        if try_add(row):
            used.add(sid)

    # Protocol search_format top-up
    for sid in search_fmt_ids:
        if count_mix("search_format_protocol_v1") >= targets["search_format"]:
            break
        if sid in used:
            continue
        try:
            row = build_search_format_v1(by_id[sid], seed, retrieval[sid])
        except Exception as exc:  # noqa: BLE001
            rejected.append({"sft_id": f"{sid}__sf", "errors": [str(exc)]})
            continue
        if try_add(row):
            used.add(sid)

    accepted.sort(key=lambda r: (r["category"], r["sft_id"]))
    built_cat = Counter(r["category"] for r in accepted)
    built_mix = Counter((r.get("metadata") or {}).get("mix_tag") for r in accepted)
    stats = {
        "targets": targets,
        "built_categories": dict(built_cat),
        "built_mix_tags": dict(built_mix),
        "n_teacher_reasoning": n_teacher,
        "n_teacher_available": len(teacher_ids),
        "n_internal_pool": len(internal_ids),
        "n_search_required_pool": len(search_req_ids),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "n_unique_sample_ids": len({r["sample_id"] for r in accepted}),
        "builder": BUILDER_NAME,
    }
    return accepted, rejected, stats


__all__ = [
    "DEFAULT_TARGETS_V1",
    "BUILDER_NAME",
    "assign_coldstart_v1",
    "load_frozen_ids",
]
