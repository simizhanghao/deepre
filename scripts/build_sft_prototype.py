"""Phase 2B: build cold-start SFT prototype JSONL (200–500 rows).

Usage (repo root):
    python scripts/build_sft_prototype.py

Reads:
    data/eval/hotpotqa_200.jsonl
    results/compare_baselines_n200_*/per_sample.jsonl
    results/retrieval_candidate_bm25_n200_*/retrieval_results.jsonl

Writes:
    data/sft/prototype_v0.jsonl
    data/sft/prototype_v0_audit.json
    data/sft/prototype_v0_report.md
    data/sft/prototype_v0_rejected.jsonl  (if any)

No GPU. No teacher LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sft.prototype_builder import (  # noqa: E402
    assign_and_build,
    index_by_sample_id,
    load_jsonl,
    summarize,
    validate_sft_row,
)

DEFAULT_EVAL = REPO_ROOT / "data" / "eval" / "hotpotqa_200.jsonl"
DEFAULT_TAXONOMY = (
    REPO_ROOT
    / "results"
    / "compare_baselines_n200_20260807_155225"
    / "per_sample.jsonl"
)
DEFAULT_RETRIEVAL = (
    REPO_ROOT
    / "results"
    / "retrieval_candidate_bm25_n200_20260807_154802"
    / "retrieval_results.jsonl"
)
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "sft"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Phase 2B SFT prototype JSONL.")
    p.add_argument("--eval-file", type=str, default=str(DEFAULT_EVAL))
    p.add_argument("--taxonomy-file", type=str, default=str(DEFAULT_TAXONOMY))
    p.add_argument("--retrieval-cache", type=str, default=str(DEFAULT_RETRIEVAL))
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--name", type=str, default="prototype_v0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-search-format", type=int, default=60)
    p.add_argument("--max-c-evidence-dual", type=int, default=40)
    p.add_argument(
        "--no-c-evidence-dual",
        action="store_true",
        help="Do not emit plain-evidence dual views for C samples.",
    )
    p.add_argument(
        "--spot-check",
        type=int,
        default=8,
        help="How many sample targets to embed in the markdown report.",
    )
    return p.parse_args()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_spot_checks(rows: List[Dict], k: int) -> List[Dict]:
    """One per category when possible, then fill."""
    by_cat: Dict[str, List[Dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    picked: List[Dict] = []
    for cat in ("internal", "evidence", "evidence_reasoning", "search_format"):
        if by_cat.get(cat):
            picked.append(by_cat[cat][0])
    for r in rows:
        if len(picked) >= k:
            break
        if r not in picked:
            picked.append(r)
    return picked[:k]


def render_report(
    summary: Dict,
    accepted: List[Dict],
    rejected: List[Dict],
    args: argparse.Namespace,
    spot: List[Dict],
) -> str:
    lines = [
        "# Phase 2B SFT Prototype Audit",
        "",
        f"- name: `{args.name}`",
        f"- builder: `{summary['builder']}`",
        f"- seed: `{args.seed}`",
        f"- eval: `{args.eval_file}`",
        f"- taxonomy: `{args.taxonomy_file}`",
        f"- retrieval: `{args.retrieval_cache}`",
        f"- evidence text equality: `{summary['evidence_text_equality']}`",
        f"- reasoning: `{summary['reasoning_source']}` (no LLM teacher)",
        f"- search query: `{summary['search_query_source']}`",
        "",
        "## Counts",
        "",
        f"- accepted: **{summary['n_accepted']}**",
        f"- rejected: **{summary['n_rejected']}**",
        "",
        "### By category",
        "",
        "| Category | Count | Rate |",
        "|----------|------:|-----:|",
    ]
    for cat in ("internal", "evidence", "evidence_reasoning", "search_format"):
        c = summary["by_category"].get(cat, 0)
        rate = summary["category_rates"].get(cat, 0.0)
        lines.append(f"| {cat} | {c} | {rate:.1%} |")
    lines.extend(
        [
            "",
            "### By taxonomy label (of accepted rows)",
            "",
            "| Label | Count |",
            "|------:|------:|",
        ]
    )
    for lab, c in sorted(summary["by_taxonomy_label"].items()):
        lines.append(f"| {lab} | {c} |")

    er = summary["by_category"].get("evidence", 0) + summary["by_category"].get(
        "evidence_reasoning", 0
    )
    n = max(summary["n_accepted"], 1)
    lines.extend(
        [
            "",
            "## Mixture check (provisional)",
            "",
            f"- evidence + evidence_reasoning: **{er / n:.1%}** (target ~60%)",
            f"- internal: **{summary['by_category'].get('internal', 0) / n:.1%}** "
            f"(target ~15–20%)",
            f"- search_format: **{summary['by_category'].get('search_format', 0) / n:.1%}** "
            f"(target ~15–20%)",
            "",
            "## Validation",
            "",
            "- All accepted rows passed `validate_sft_row` "
            "(tags, answer=gold, evidence provenance, template legality).",
            "- Base-model wrong outputs were never used as targets.",
            "- Candidate-BM25 search_format only when all gold titles ∈ Top-K.",
            "",
        ]
    )
    if rejected:
        lines.append("## Rejected (first 20)")
        lines.append("")
        for item in rejected[:20]:
            lines.append(f"- `{item.get('sft_id')}`: {item.get('errors')}")
        lines.append("")

    lines.append("## Spot checks")
    lines.append("")
    for row in spot:
        lines.append(f"### `{row['sft_id']}` ({row['category']} / {row['taxonomy_label']})")
        lines.append("")
        lines.append("User (truncated):")
        lines.append("")
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        lines.append("```text")
        lines.append(user[:800] + ("…" if len(user) > 800 else ""))
        lines.append("```")
        lines.append("")
        lines.append("Target:")
        lines.append("")
        lines.append("```text")
        tgt = row["target"]
        lines.append(tgt[:1200] + ("…" if len(tgt) > 1200 else ""))
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## Human audit checklist",
            "",
            "- [ ] Internal rows: no documents; short routing tag; gold answer",
            "- [ ] Evidence rows: provenance ids match HotpotQA sentences",
            "- [ ] Reasoning rows: short, no new facts, answer-consistent",
            "- [ ] Search-format: observation is candidate scope; not claimed full-corpus",
            "- [ ] No Base wrong answers as targets",
            "",
            "## Next",
            "",
            "After human spot-check passes → Phase 2C scale to 2k–5k.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    eval_path = resolve_path(args.eval_file)
    tax_path = resolve_path(args.taxonomy_file)
    retr_path = resolve_path(args.retrieval_cache)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in (eval_path, tax_path, retr_path):
        if not path.is_file():
            raise SystemExit(f"missing input file: {path}")

    eval_rows = load_jsonl(str(eval_path))
    tax_rows = load_jsonl(str(tax_path))
    retr_rows = load_jsonl(str(retr_path))
    taxonomy = {r["sample_id"]: r["label"] for r in tax_rows}
    retrieval = index_by_sample_id(retr_rows)

    accepted, rejected = assign_and_build(
        eval_rows,
        taxonomy,
        retrieval,
        seed=args.seed,
        max_search_format=args.max_search_format,
        include_c_evidence_dual=not args.no_c_evidence_dual,
        max_c_evidence_dual=args.max_c_evidence_dual,
    )

    # Final re-validate pass
    hard_fail = []
    for row in accepted:
        errs = validate_sft_row(row)
        if errs:
            hard_fail.append({"sft_id": row["sft_id"], "errors": errs})
    if hard_fail:
        raise SystemExit(f"internal validation failed on accepted set: {hard_fail[:3]}")

    summary = summarize(accepted, rejected)
    summary["inputs"] = {
        "eval_file": str(eval_path),
        "taxonomy_file": str(tax_path),
        "retrieval_cache": str(retr_path),
        "seed": args.seed,
        "max_search_format": args.max_search_format,
        "max_c_evidence_dual": 0
        if args.no_c_evidence_dual
        else args.max_c_evidence_dual,
        "n_eval": len(eval_rows),
        "n_taxonomy": len(taxonomy),
        "n_retrieval": len(retrieval),
    }

    jsonl_path = out_dir / f"{args.name}.jsonl"
    audit_path = out_dir / f"{args.name}_audit.json"
    report_path = out_dir / f"{args.name}_report.md"
    rejected_path = out_dir / f"{args.name}_rejected.jsonl"

    write_jsonl(jsonl_path, accepted)
    if rejected:
        write_jsonl(rejected_path, rejected)
    audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    spot = pick_spot_checks(accepted, args.spot_check)
    report_path.write_text(
        render_report(summary, accepted, rejected, args, spot), encoding="utf-8"
    )

    print(f"wrote {jsonl_path}  n={len(accepted)}")
    print(f"wrote {audit_path}")
    print(f"wrote {report_path}")
    if rejected:
        print(f"wrote {rejected_path}  n_rejected={len(rejected)}")
    print("by_category:", summary["by_category"])
    if not (200 <= len(accepted) <= 500):
        print(
            f"WARNING: accepted={len(accepted)} outside prototype range 200–500",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
