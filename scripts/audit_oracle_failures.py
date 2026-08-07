"""Quick Oracle failure audit for Phase 1 (no model, no GPU).

Reads an Oracle baseline metrics.json + the HotpotQA eval JSONL, keeps rows
with exact_match == 0, and writes a labeling sheet with helper signals:

  - pred / gold
  - whether gold (normalized) appears in oracle supporting docs
  - whether gold is a substring of pred (format/extraction soft-hit)
  - supporting titles + gold sentences

Suggested labels (fill manually in audit.md / copy of TSV):
  A  semantic OK, EM miss (format / normalize)
  B  answer span in docs, model missed it (evidence selection)
  C  needs multi-hop; docs present but reasoning failed
  D  pipeline / oracle construction bug

Usage:
  python scripts/audit_oracle_failures.py \
    --metrics results/baseline_oracle_n50_20260807_151255/metrics.json \
    --eval-file data/eval/hotpotqa_50.jsonl \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import normalize_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle EM=0 failure audit sheet.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results"),
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all failures")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to Oracle run metrics.json",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        required=True,
        help="Matching HotpotQA eval JSONL",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["sample_id"]] = row
    return out


def oracle_doc_texts(sample: Dict[str, Any]) -> List[str]:
    titles: Set[str] = set()
    for sf in sample.get("supporting_facts") or []:
        titles.add(sf["title"])
    texts: List[str] = []
    for ctx in sample.get("contexts") or []:
        if ctx["title"] in titles:
            texts.append(ctx.get("text") or "")
    return texts


def gold_in_docs(golds: List[str], docs: List[str]) -> bool:
    blob = normalize_answer(" ".join(docs))
    for g in golds:
        ng = normalize_answer(g)
        if ng and ng in blob:
            return True
    return False


def gold_in_pred(golds: List[str], pred: str) -> bool:
    np = normalize_answer(pred)
    for g in golds:
        ng = normalize_answer(g)
        if ng and ng in np:
            return True
    return False


def pred_in_gold(golds: List[str], pred: str) -> bool:
    np = normalize_answer(pred)
    if not np:
        return False
    for g in golds:
        ng = normalize_answer(g)
        if np and np in ng:
            return True
    return False


def suggest_hint(
    *,
    gold_in_documents: bool,
    gold_substring_of_pred: bool,
    pred_substring_of_gold: bool,
    n_support_titles: int,
) -> str:
    if gold_substring_of_pred or pred_substring_of_gold:
        return "likely_A_format_or_partial"
    if gold_in_documents and n_support_titles >= 2:
        return "likely_B_or_C_evidence_or_multihop"
    if gold_in_documents:
        return "likely_B_evidence_miss"
    return "check_D_or_hard_reasoning"


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    eval_path = Path(args.eval_file)
    if not metrics_path.is_absolute():
        metrics_path = REPO_ROOT / metrics_path
    if not eval_path.is_absolute():
        eval_path = REPO_ROOT / eval_path

    metrics_rows: List[Dict[str, Any]] = load_json(metrics_path)
    eval_map = load_eval_map(eval_path)

    failures = [r for r in metrics_rows if r.get("metrics", {}).get("exact_match", 1) == 0.0]
    if args.max_samples and args.max_samples > 0:
        failures = failures[: args.max_samples]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"oracle_failure_audit_n{len(failures)}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: List[Dict[str, Any]] = []
    missing = 0
    for r in failures:
        sid = r["sample_id"]
        sample = eval_map.get(sid)
        if sample is None:
            missing += 1
            continue
        docs = oracle_doc_texts(sample)
        golds = list(r.get("gold_answers") or sample.get("gold_answers") or [])
        pred = r.get("prediction") or ""
        titles = []
        seen = set()
        for sf in sample.get("supporting_facts") or []:
            if sf["title"] not in seen:
                seen.add(sf["title"])
                titles.append(sf["title"])

        gin_docs = gold_in_docs(golds, docs)
        gin_pred = gold_in_pred(golds, pred)
        pin_gold = pred_in_gold(golds, pred)
        hint = suggest_hint(
            gold_in_documents=gin_docs,
            gold_substring_of_pred=gin_pred,
            pred_substring_of_gold=pin_gold,
            n_support_titles=len(titles),
        )
        row = {
            "sample_id": sid,
            "question": sample["question"],
            "prediction": pred,
            "gold_answers": golds,
            "token_f1": r.get("metrics", {}).get("token_f1"),
            "oracle_titles": titles,
            "gold_normalized_in_oracle_docs": gin_docs,
            "gold_normalized_in_prediction": gin_pred,
            "prediction_normalized_in_gold": pin_gold,
            "supporting_facts": sample.get("supporting_facts"),
            "auto_hint": hint,
            "label": "",  # fill A/B/C/D
            "notes": "",
        }
        audit_rows.append(row)

    # Write JSONL
    jsonl_path = run_dir / "audit.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write TSV for quick spreadsheet labeling
    tsv_path = run_dir / "audit.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        headers = [
            "sample_id",
            "label",
            "auto_hint",
            "token_f1",
            "gold_in_docs",
            "gold_in_pred",
            "pred_in_gold",
            "prediction",
            "gold_answers",
            "oracle_titles",
            "question",
            "notes",
        ]
        f.write("\t".join(headers) + "\n")
        for row in audit_rows:
            f.write(
                "\t".join(
                    [
                        row["sample_id"],
                        row["label"],
                        row["auto_hint"],
                        str(row["token_f1"]),
                        str(row["gold_normalized_in_oracle_docs"]),
                        str(row["gold_normalized_in_prediction"]),
                        str(row["prediction_normalized_in_gold"]),
                        row["prediction"].replace("\t", " ").replace("\n", " "),
                        " | ".join(row["gold_answers"]).replace("\t", " "),
                        " | ".join(row["oracle_titles"]).replace("\t", " "),
                        row["question"].replace("\t", " ").replace("\n", " "),
                        "",
                    ]
                )
                + "\n"
            )

    # Markdown checklist
    n = len(audit_rows)
    n_gold_in_docs = sum(1 for r in audit_rows if r["gold_normalized_in_oracle_docs"])
    n_soft = sum(
        1
        for r in audit_rows
        if r["gold_normalized_in_prediction"] or r["prediction_normalized_in_gold"]
    )
    n_hint_a = sum(1 for r in audit_rows if r["auto_hint"].startswith("likely_A"))
    n_hint_bc = sum(
        1 for r in audit_rows if "B_or_C" in r["auto_hint"] or r["auto_hint"].startswith("likely_B")
    )
    n_check_d = sum(1 for r in audit_rows if r["auto_hint"].startswith("check_D"))

    md_lines = [
        "# Oracle Failure Audit (auto sheet)",
        "",
        f"- metrics: `{metrics_path}`",
        f"- eval: `{eval_path}`",
        f"- EM=0 failures audited: **{n}** (missing eval rows: {missing})",
        f"- gold string found in oracle docs: **{n_gold_in_docs}/{n}**",
        f"- soft string match pred↔gold: **{n_soft}/{n}**",
        f"- auto_hint A-like: {n_hint_a}; B/C-like: {n_hint_bc}; check D: {n_check_d}",
        "",
        "## Label legend",
        "",
        "| Label | Meaning |",
        "|-------|---------|",
        "| A | Semantic OK / format / normalize EM miss |",
        "| B | Answer in docs; model missed evidence |",
        "| C | Multi-hop / reasoning failure |",
        "| D | Pipeline / oracle construction bug |",
        "",
        "## How to label (15–40 min)",
        "",
        "1. Open `audit.tsv` (or browse list below).",
        "2. Fill `label` with A/B/C/D; optional `notes`.",
        "3. Priority: first confirm almost no D; then count A vs B/C.",
        "",
        "## Failure list",
        "",
    ]
    for i, row in enumerate(audit_rows, 1):
        md_lines.extend(
            [
                f"### {i}. `{row['sample_id']}`",
                "",
                f"- **Q:** {row['question']}",
                f"- **pred:** `{row['prediction']}`",
                f"- **gold:** {row['gold_answers']}",
                f"- **F1:** {row['token_f1']}",
                f"- **oracle titles:** {row['oracle_titles']}",
                f"- **gold_in_docs:** {row['gold_normalized_in_oracle_docs']}",
                f"- **gold_in_pred / pred_in_gold:** "
                f"{row['gold_normalized_in_prediction']} / "
                f"{row['prediction_normalized_in_gold']}",
                f"- **auto_hint:** `{row['auto_hint']}`",
                f"- **label:** _pending_",
                "",
            ]
        )

    md_path = run_dir / "audit.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary = {
        "num_failures": n,
        "missing_eval": missing,
        "gold_in_oracle_docs": n_gold_in_docs,
        "soft_pred_gold_match": n_soft,
        "auto_hint_counts": {
            "A_like": n_hint_a,
            "B_or_C_like": n_hint_bc,
            "check_D": n_check_d,
        },
        "artifacts": {
            "audit_jsonl": str(jsonl_path),
            "audit_tsv": str(tsv_path),
            "audit_md": str(md_path),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[audit_oracle_failures] wrote {run_dir}")


if __name__ == "__main__":
    main()
