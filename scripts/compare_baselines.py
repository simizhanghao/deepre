"""Phase 1F: join Direct / Oracle / Candidate-BM25 by sample_id + failure taxonomy.

Reads three metrics.json (same HotpotQA subset), classifies each sample, writes:

  results/{run}/taxonomy_summary.json
  results/{run}/per_sample.jsonl
  results/{run}/per_sample.tsv
  results/{run}/report.md

Taxonomy (EM-based; correct = exact_match == 1.0):

  A  Direct❌ Oracle✅ BM25✅   retrieval helps
  B  Direct❌ Oracle✅ BM25❌   retrieval misses / noise vs oracle
  C  Direct❌ Oracle❌ *        evidence/reasoning gap (oracle also fails)
  D  Direct✅ Oracle✅ BM25✅   search likely unnecessary (cost)
  E  Direct✅ * BM25❌          retrieval hurts vs direct
  O  other combinations

Usage:
  python scripts/compare_baselines.py \
    --direct results/baseline_direct_n50_20260807_151233/metrics.json \
    --oracle results/baseline_oracle_n50_20260807_151255/metrics.json \
    --bm25 results/baseline_candidate_bm25_n50_20260807_153012/metrics.json \
    --retrieval-summary results/retrieval_candidate_bm25_n50_20260807_152841/summary.json \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Direct/Oracle/BM25 baselines.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results"))
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--direct", type=str, required=True, help="Direct metrics.json")
    p.add_argument("--oracle", type=str, required=True, help="Oracle metrics.json")
    p.add_argument("--bm25", type=str, required=True, help="Candidate-BM25 metrics.json")
    p.add_argument(
        "--retrieval-summary",
        type=str,
        default=None,
        help="Optional Candidate-BM25 retrieval summary.json",
    )
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_metrics(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        out[r["sample_id"]] = r
    return out


def is_correct(row: Dict[str, Any]) -> bool:
    return float(row.get("metrics", {}).get("exact_match", 0.0)) == 1.0


def classify(d_ok: bool, o_ok: bool, b_ok: bool) -> str:
    if (not d_ok) and o_ok and b_ok:
        return "A"
    if (not d_ok) and o_ok and (not b_ok):
        return "B"
    if (not d_ok) and (not o_ok):
        return "C"
    if d_ok and o_ok and b_ok:
        return "D"
    if d_ok and (not b_ok):
        return "E"
    return "O"


LABEL_MEANING = {
    "A": "Direct❌ Oracle✅ BM25✅ — retrieval helps",
    "B": "Direct❌ Oracle✅ BM25❌ — retrieval miss / noise",
    "C": "Direct❌ Oracle❌ — evidence/reasoning gap",
    "D": "Direct✅ Oracle✅ BM25✅ — search likely unnecessary (cost)",
    "E": "Direct✅ BM25❌ — retrieval hurts",
    "O": "other pattern",
}


def mean_em(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1.0 if r[key] else 0.0 for r in rows) / len(rows)


def main() -> None:
    args = parse_args()
    direct = load_metrics(resolve(args.direct))
    oracle = load_metrics(resolve(args.oracle))
    bm25 = load_metrics(resolve(args.bm25))

    ids = sorted(set(direct) & set(oracle) & set(bm25))
    if not ids:
        raise SystemExit("no overlapping sample_ids across the three metrics files")
    # Preserve Direct file order when possible
    ordered = [r["sample_id"] for r in json.loads(resolve(args.direct).read_text())]
    ids = [i for i in ordered if i in ids]
    if args.max_samples and args.max_samples > 0:
        ids = ids[: args.max_samples]

    per_sample: List[Dict[str, Any]] = []
    for sid in ids:
        d, o, b = direct[sid], oracle[sid], bm25[sid]
        d_ok, o_ok, b_ok = is_correct(d), is_correct(o), is_correct(b)
        label = classify(d_ok, o_ok, b_ok)
        row = {
            "sample_id": sid,
            "label": label,
            "label_meaning": LABEL_MEANING[label],
            "direct_em": 1.0 if d_ok else 0.0,
            "oracle_em": 1.0 if o_ok else 0.0,
            "bm25_em": 1.0 if b_ok else 0.0,
            "direct_f1": d.get("metrics", {}).get("token_f1"),
            "oracle_f1": o.get("metrics", {}).get("token_f1"),
            "bm25_f1": b.get("metrics", {}).get("token_f1"),
            "direct_pred": d.get("prediction"),
            "oracle_pred": o.get("prediction"),
            "bm25_pred": b.get("prediction"),
            "gold_answers": d.get("gold_answers") or o.get("gold_answers"),
            "direct_prompt_tokens": d.get("prompt_tokens"),
            "oracle_prompt_tokens": o.get("prompt_tokens"),
            "bm25_prompt_tokens": b.get("prompt_tokens"),
        }
        per_sample.append(row)
        if args.debug:
            print(f"{label} {sid} D={d_ok} O={o_ok} B={b_ok}")

    counts = Counter(r["label"] for r in per_sample)
    n = len(per_sample)
    taxonomy_rates = {k: round(counts.get(k, 0) / n, 4) for k in "ABCDEO"}

    retrieval_summary = None
    if args.retrieval_summary:
        retrieval_summary = json.loads(
            resolve(args.retrieval_summary).read_text(encoding="utf-8")
        )

    aggregate = {
        "num_samples": n,
        "mean_em": {
            "direct": round(mean_em(per_sample, "direct_em"), 4),
            "oracle": round(mean_em(per_sample, "oracle_em"), 4),
            "candidate_bm25": round(mean_em(per_sample, "bm25_em"), 4),
        },
        "deltas": {
            "oracle_minus_direct": round(
                mean_em(per_sample, "oracle_em") - mean_em(per_sample, "direct_em"), 4
            ),
            "bm25_minus_direct": round(
                mean_em(per_sample, "bm25_em") - mean_em(per_sample, "direct_em"), 4
            ),
            "oracle_minus_bm25": round(
                mean_em(per_sample, "oracle_em") - mean_em(per_sample, "bm25_em"), 4
            ),
        },
        "taxonomy_counts": {k: counts.get(k, 0) for k in "ABCDEO"},
        "taxonomy_rates": taxonomy_rates,
        "mean_prompt_tokens": {
            "direct": round(
                sum(r["direct_prompt_tokens"] or 0 for r in per_sample) / n, 1
            ),
            "oracle": round(
                sum(r["oracle_prompt_tokens"] or 0 for r in per_sample) / n, 1
            ),
            "candidate_bm25": round(
                sum(r["bm25_prompt_tokens"] or 0 for r in per_sample) / n, 1
            ),
        },
        "retrieval_summary": retrieval_summary,
        "inputs": {
            "direct": str(resolve(args.direct)),
            "oracle": str(resolve(args.oracle)),
            "bm25": str(resolve(args.bm25)),
        },
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"compare_baselines_n{n}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "taxonomy_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (run_dir / "per_sample.jsonl").open("w", encoding="utf-8") as f:
        for r in per_sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (run_dir / "per_sample.tsv").open("w", encoding="utf-8") as f:
        headers = [
            "sample_id",
            "label",
            "direct_em",
            "oracle_em",
            "bm25_em",
            "direct_pred",
            "oracle_pred",
            "bm25_pred",
            "gold_answers",
        ]
        f.write("\t".join(headers) + "\n")
        for r in per_sample:
            gold = " | ".join(r["gold_answers"] or [])
            f.write(
                "\t".join(
                    [
                        r["sample_id"],
                        r["label"],
                        str(int(r["direct_em"])),
                        str(int(r["oracle_em"])),
                        str(int(r["bm25_em"])),
                        str(r["direct_pred"]).replace("\t", " ").replace("\n", " "),
                        str(r["oracle_pred"]).replace("\t", " ").replace("\n", " "),
                        str(r["bm25_pred"]).replace("\t", " ").replace("\n", " "),
                        gold.replace("\t", " "),
                    ]
                )
                + "\n"
            )

    md = [
        "# Baseline Comparison & Failure Taxonomy",
        "",
        f"n = **{n}**",
        "",
        "## Aggregate EM",
        "",
        "| Method | EM |",
        "|--------|---:|",
        f"| Direct | {aggregate['mean_em']['direct']} |",
        f"| Oracle | {aggregate['mean_em']['oracle']} |",
        f"| Candidate-BM25 | {aggregate['mean_em']['candidate_bm25']} |",
        "",
        f"- Oracle − Direct = **{aggregate['deltas']['oracle_minus_direct']}**",
        f"- BM25 − Direct = **{aggregate['deltas']['bm25_minus_direct']}**",
        f"- Oracle − BM25 = **{aggregate['deltas']['oracle_minus_bm25']}**",
        "",
        "## Taxonomy",
        "",
        "| Label | Count | Rate | Meaning |",
        "|------:|------:|-----:|---------|",
    ]
    for k in "ABCDEO":
        md.append(
            f"| {k} | {counts.get(k, 0)} | {taxonomy_rates[k]:.1%} | {LABEL_MEANING[k]} |"
        )
    if retrieval_summary:
        md.extend(
            [
                "",
                "## Retrieval (Candidate-BM25)",
                "",
                f"- mean title Recall@5: "
                f"**{retrieval_summary.get('mean_title_recall@5')}**",
                f"- title hit_all@5: "
                f"**{retrieval_summary.get('title_hit_all_rate@5')}**",
            ]
        )
    md.extend(
        [
            "",
            "## Implications (auto)",
            "",
            f"- External knowledge value (A+B share of Direct-fail with Oracle-ok): "
            f"**{(counts.get('A', 0) + counts.get('B', 0)) / n:.1%}** of all samples "
            f"are Direct❌ Oracle✅.",
            f"- Retrieval gap among those: "
            f"B/(A+B) = "
            f"**{counts.get('B', 0) / max(counts.get('A', 0) + counts.get('B', 0), 1):.1%}** "
            f"(Oracle ok but BM25 fail).",
            f"- Reasoning/evidence gap (C): **{taxonomy_rates['C']:.1%}**",
            f"- Possible no-search cases (D): **{taxonomy_rates['D']:.1%}**",
            f"- Retrieval harm (E): **{taxonomy_rates['E']:.1%}**",
            "",
            "See `per_sample.tsv` for labeling / case review.",
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"[compare_baselines] wrote {run_dir}")


if __name__ == "__main__":
    main()
