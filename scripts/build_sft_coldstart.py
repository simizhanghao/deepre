"""Phase 2C: build coldstart_v0 (~3k) from HotpotQA train + audit.

Prereqs:
  1) python scripts/prepare_hotpotqa_train.py --max-samples 8000
  2) GPU: python scripts/label_direct_train.py --max-samples 4000 ...
  3) CPU: python scripts/retrieve_candidate_bm25.py --eval-file <train_pool> \\
          --max-samples 8000 --top-k 5

Usage:
  python scripts/build_sft_coldstart.py \\
    --train-file data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl \\
    --direct-labels results/phase2c_direct_label_*/labels.jsonl \\
    --retrieval-cache results/retrieval_candidate_bm25_*/retrieval_results.jsonl \\
    --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.sft.coldstart_builder import (  # noqa: E402
    DEFAULT_TARGETS,
    assign_coldstart,
    load_frozen_ids,
)
from src.sft.prototype_builder import (  # noqa: E402
    index_by_sample_id,
    load_jsonl,
    validate_sft_row,
)

DEFAULT_FROZEN = REPO_ROOT / "data" / "eval" / "hotpotqa_200_ids.txt"
DEFAULT_MODEL = "/data1/hcc/.hf_home/Qwen2.5-3B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Phase 2C coldstart_v0 SFT JSONL.")
    p.add_argument("--train-file", type=str, required=True)
    p.add_argument("--direct-labels", type=str, required=True)
    p.add_argument("--retrieval-cache", type=str, required=True)
    p.add_argument("--frozen-val-ids", type=str, default=str(DEFAULT_FROZEN))
    p.add_argument("--output-jsonl", type=str, default="data/sft/coldstart_v0.jsonl")
    p.add_argument(
        "--audit-dir",
        type=str,
        default="results/phase2c_coldstart_v0",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-distractors", type=int, default=3)
    p.add_argument("--noisy-fraction", type=float, default=0.5)
    p.add_argument("--n-evidence-reasoning", type=int, default=1500)
    p.add_argument("--n-evidence", type=int, default=600)
    p.add_argument("--n-internal", type=int, default=450)
    p.add_argument("--n-search-format", type=int, default=450)
    p.add_argument(
        "--tokenizer-path",
        type=str,
        default=DEFAULT_MODEL,
        help="For prompt/target token stats (CPU). Empty string to skip.",
    )
    p.add_argument("--spot-check", type=int, default=12)
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(vals: List[int], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def token_stats(
    rows: Sequence[Dict[str, Any]], tokenizer_path: str
) -> Dict[str, Any]:
    if not tokenizer_path:
        return {"skipped": True}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompt_lens: List[int] = []
    target_lens: List[int] = []
    total_lens: List[int] = []
    for row in rows:
        # messages only (system+user); target separate
        try:
            prompt_text = tok.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = "\n".join(m["content"] for m in row["messages"])
        pt = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
        tt = len(tok(row["target"], add_special_tokens=False)["input_ids"])
        prompt_lens.append(pt)
        target_lens.append(tt)
        total_lens.append(pt + tt)

    def pack(name: str, vals: List[int]) -> Dict[str, float]:
        return {
            "p50": round(percentile(vals, 0.50), 1),
            "p95": round(percentile(vals, 0.95), 1),
            "max": float(max(vals) if vals else 0),
            "mean": round(sum(vals) / max(len(vals), 1), 1),
        }

    return {
        "tokenizer_path": tokenizer_path,
        "n": len(rows),
        "prompt_tokens": pack("prompt", prompt_lens),
        "target_tokens": pack("target", target_lens),
        "total_tokens": pack("total", total_lens),
    }


def audit_rows(
    rows: Sequence[Dict[str, Any]],
    frozen_ids: set,
    rejected: Sequence[Dict[str, Any]],
    build_stats: Dict[str, Any],
    tokens: Dict[str, Any],
) -> Dict[str, Any]:
    sids = [r["sample_id"] for r in rows]
    targets = [r["target"] for r in rows]
    dup_targets = len(targets) - len(set(targets))
    overlap = sorted(set(sids) & frozen_ids)
    non_train = [s for s in sids if "_train_" not in s]

    # revalidate
    n_valid = 0
    evidence_ok = 0
    answer_ok = 0
    for r in rows:
        errs = validate_sft_row(r)
        if not errs:
            n_valid += 1
        if r["category"] != "internal":
            if r.get("evidence_refs"):
                evidence_ok += 1
        if r.get("gold_answer"):
            answer_ok += 1

    by_cat = Counter(r["category"] for r in rows)
    n = max(len(rows), 1)
    reasoning_marked = sum(
        1
        for r in rows
        if r["category"] == "evidence_reasoning"
        and (r.get("provenance") or {}).get("reasoning_source") == "template_v0"
    )
    n_er = by_cat.get("evidence_reasoning", 0)

    return {
        "total_samples": len(rows),
        "source_split": "train",
        "overlap_with_frozen_validation_200": len(overlap),
        "overlap_examples": overlap[:5],
        "non_train_sample_ids": len(non_train),
        "deterministic_validation_pass_rate": round(n_valid / n, 4),
        "n_validation_pass": n_valid,
        "evidence_refs_nonempty_rate": round(
            evidence_ok / max(sum(1 for r in rows if r["category"] != "internal"), 1),
            4,
        ),
        "gold_answer_present_rate": round(answer_ok / n, 4),
        "by_category": dict(by_cat),
        "category_rates": {k: round(v / n, 4) for k, v in by_cat.items()},
        "exact_duplicate_targets": dup_targets,
        "unique_sft_ids": len({r["sft_id"] for r in rows}),
        "unique_sample_ids": len(set(sids)),
        "template_v0_reasoning_marked": reasoning_marked,
        "template_v0_reasoning_rate": round(reasoning_marked / max(n_er, 1), 4),
        "n_rejected_at_build": len(rejected),
        "build_stats": build_stats,
        "token_stats": tokens,
        "gates": {
            "train_only": len(non_train) == 0,
            "zero_val200_overlap": len(overlap) == 0,
            "validation_near_100": (n_valid / n) >= 0.999,
            "approx_3k": 2500 <= len(rows) <= 3500,
            "search_format_le_20pct": by_cat.get("search_format", 0) / n <= 0.20 + 1e-9,
            "internal_in_10_20": 0.10 <= by_cat.get("internal", 0) / n <= 0.20 + 1e-9
            or by_cat.get("internal", 0) > 0,  # soft if shortfall documented
        },
    }


def render_report(audit: Dict[str, Any], spot: List[Dict[str, Any]]) -> str:
    lines = [
        "# Phase 2C Cold-start v0 Audit",
        "",
        f"- total: **{audit['total_samples']}**",
        f"- source_split: `{audit['source_split']}`",
        f"- overlap with frozen validation-200: **{audit['overlap_with_frozen_validation_200']}**",
        f"- deterministic validation pass: **{audit['deterministic_validation_pass_rate']:.2%}**",
        f"- duplicate targets: **{audit['exact_duplicate_targets']}**",
        f"- template_v0 reasoning marked: **{audit['template_v0_reasoning_marked']}** "
        f"({audit['template_v0_reasoning_rate']:.1%} of evidence_reasoning)",
        "",
        "## Gates",
        "",
    ]
    for k, v in audit["gates"].items():
        lines.append(f"- `{k}`: {'PASS' if v else 'FAIL'}")
    lines.extend(["", "## Category mixture", "", "| Category | Count | Rate |", "|---|---:|---:|"])
    for cat in ("evidence_reasoning", "evidence", "internal", "search_format"):
        c = audit["by_category"].get(cat, 0)
        r = audit["category_rates"].get(cat, 0.0)
        lines.append(f"| {cat} | {c} | {r:.1%} |")

    bs = audit.get("build_stats") or {}
    lines.extend(
        [
            "",
            "## Build stats",
            "",
            f"- targets: `{bs.get('targets')}`",
            f"- shortfall: `{bs.get('shortfall')}`",
            f"- direct_correct available: **{bs.get('n_direct_correct_available')}**",
            f"- search eligible: **{bs.get('n_search_eligible')}**",
            f"- context views: `{bs.get('context_views')}`",
            "",
            "## Token stats (for SFT cutoff)",
            "",
        ]
    )
    ts = audit.get("token_stats") or {}
    if ts.get("skipped"):
        lines.append("_tokenizer stats skipped_")
    else:
        for key in ("prompt_tokens", "target_tokens", "total_tokens"):
            pack = ts.get(key) or {}
            lines.append(
                f"- **{key}**: p50={pack.get('p50')} p95={pack.get('p95')} "
                f"max={pack.get('max')} mean={pack.get('mean')}"
            )

    lines.extend(["", "## Spot checks", ""])
    for row in spot:
        lines.append(
            f"### `{row['sft_id']}` ({row['category']} / "
            f"{(row.get('metadata') or {}).get('context_view', '-')})"
        )
        lines.append("")
        lines.append("```text")
        lines.append(row["target"][:1000] + ("…" if len(row["target"]) > 1000 else ""))
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## Human checklist (30–50 rows)",
            "",
            "- [ ] All sample_id contain `_train_`",
            "- [ ] No validation-200 ids",
            "- [ ] Noisy rows still emit only gold evidence",
            "- [ ] Internal rows are Direct-correct sourced",
            "- [ ] search_format scope=candidate",
            "",
            "Next: Phase 2D first Qwen2.5-3B Cold-start SFT (after spot-check).",
            "",
        ]
    )
    return "\n".join(lines)


def pick_spot(rows: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r)
    picked: List[Dict[str, Any]] = []
    # prefer one clean + one noisy for evidence types
    for cat in ("internal", "evidence", "evidence_reasoning", "search_format"):
        items = by.get(cat) or []
        if not items:
            continue
        picked.append(items[0])
        noisy = next(
            (
                x
                for x in items
                if (x.get("metadata") or {}).get("context_view") == "noisy"
            ),
            None,
        )
        if noisy and noisy not in picked:
            picked.append(noisy)
    for r in rows:
        if len(picked) >= k:
            break
        if r not in picked:
            picked.append(r)
    return picked[:k]


def main() -> None:
    args = parse_args()
    train_path = resolve(args.train_file)
    labels_path = resolve(args.direct_labels)
    retr_path = resolve(args.retrieval_cache)
    frozen_path = resolve(args.frozen_val_ids)
    out_jsonl = resolve(args.output_jsonl)
    audit_dir = resolve(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    for p in (train_path, labels_path, retr_path, frozen_path):
        if not p.is_file():
            raise SystemExit(f"missing: {p}")

    train_rows = load_jsonl(str(train_path))
    label_rows = load_jsonl(str(labels_path))
    retr_rows = load_jsonl(str(retr_path))
    frozen_ids = load_frozen_ids(str(frozen_path))
    direct_labels = {r["sample_id"]: r for r in label_rows}
    retrieval = index_by_sample_id(retr_rows)

    targets = {
        "evidence_reasoning": args.n_evidence_reasoning,
        "evidence": args.n_evidence,
        "internal": args.n_internal,
        "search_format": args.n_search_format,
    }

    accepted, rejected, build_stats = assign_coldstart(
        train_rows,
        frozen_ids=frozen_ids,
        direct_labels=direct_labels,
        retrieval=retrieval,
        seed=args.seed,
        targets=targets,
        max_distractors=args.max_distractors,
        noisy_fraction=args.noisy_fraction,
    )

    write_jsonl(out_jsonl, accepted)
    if rejected:
        write_jsonl(audit_dir / "rejected.jsonl", rejected)

    print("computing token stats...")
    tokens = token_stats(accepted, args.tokenizer_path or "")
    audit = audit_rows(accepted, frozen_ids, rejected, build_stats, tokens)
    audit["inputs"] = {
        "train_file": str(train_path),
        "direct_labels": str(labels_path),
        "retrieval_cache": str(retr_path),
        "frozen_val_ids": str(frozen_path),
        "seed": args.seed,
        "targets": targets,
        "noisy_fraction": args.noisy_fraction,
        "max_distractors": args.max_distractors,
    }
    (audit_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    spot = pick_spot(accepted, args.spot_check)
    (audit_dir / "report.md").write_text(
        render_report(audit, spot), encoding="utf-8"
    )
    # also keep a small spot jsonl
    write_jsonl(audit_dir / "spot_checks.jsonl", spot)

    print(f"wrote {out_jsonl} n={len(accepted)}")
    print(f"wrote {audit_dir / 'audit.json'}")
    print(f"wrote {audit_dir / 'report.md'}")
    print("by_category:", audit["by_category"])
    print("overlap_val200:", audit["overlap_with_frozen_validation_200"])
    print("gates:", audit["gates"])
    if audit["overlap_with_frozen_validation_200"] != 0:
        raise SystemExit("FAIL: overlap with frozen validation-200")
    if audit["deterministic_validation_pass_rate"] < 0.999:
        raise SystemExit("FAIL: validation pass rate < 99.9%")


if __name__ == "__main__":
    main()
