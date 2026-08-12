#!/usr/bin/env python3
"""Repair CUR answer extraction without altering immutable raw capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.metrics import exact_match, token_f1


def repair_pred(pred: str) -> tuple[str, bool]:
    # rewards_cur v0 captured from an earlier unmatched <answer> in a user
    # suffix. The true final opening remains inside that captured span.
    lower = pred.lower()
    pos = lower.rfind("<answer>")
    if pos < 0:
        return pred.strip(), False
    return pred[pos + len("<answer>") :].strip(), True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    repaired = 0
    changed_f1 = 0
    for line in args.input.read_text().splitlines():
        row = json.loads(line)
        pred, applied = repair_pred(str(row.get("pred", "")))
        gold = str(row.get("gold", ""))
        f1 = float(token_f1(pred, [gold])) if pred else 0.0
        em = float(exact_match(pred, [gold])) if pred else 0.0
        repaired += int(applied)
        changed_f1 += int(abs(f1 - float(row.get("answer_f1", 0.0))) > 1e-12)
        row.update({
            "pred": pred,
            "answer_f1": f1,
            "answer_em": em,
            "answer_extraction_repaired": applied,
            "raw_capture_file": str(args.input),
        })
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    summary = {
        "gate": "CUR_OUTCOME_REPAIR_PASS",
        "rows": len(rows),
        "repaired_rows": repaired,
        "rows_with_changed_f1": changed_f1,
        "raw_capture_immutable": str(args.input),
        "repair": "take suffix after final literal <answer> in captured pred",
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
