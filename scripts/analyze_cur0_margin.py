#!/usr/bin/env python3
"""Gate 0B: frozen root margin versus paired causal F1 benefit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    return float((sum(float(p > n) + .5 * float(p == n) for p in pos for n in neg)) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", type=Path, required=True)
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import torch

    qrows = {x["sample_id"]: x for x in map(json.loads, args.outcomes.read_text().splitlines())}
    logits = torch.load(args.capture_dir / "all_logits.pt", map_location="cpu")
    rows = []
    for sid, tensors in logits.items():
        lp = torch.log_softmax(tensors[0].reshape(-1).float(), -1)
        margin = float(lp[27] - lp[4159])
        rows.append({**qrows[sid], "root_margin_search_minus_internal": margin})
    margins = np.asarray([x["root_margin_search_minus_internal"] for x in rows])
    delta = np.asarray([x["delta_f1"] for x in rows])
    nonzero = delta != 0
    high = np.asarray([x["high_confidence_direction"] != "borderline" for x in rows])
    summary = {
        "gate": "GATE_0B_COMPLETE",
        "n": len(rows),
        "margin_definition": "logP(<search>)-logP(<internal>)",
        "spearman_margin_delta_f1": spearman(margins, delta),
        "direction_auroc_nonzero": auroc(margins[nonzero], (delta[nonzero] > 0).astype(int)),
        "n_nonzero": int(nonzero.sum()),
        "direction_auroc_high_confidence": auroc(margins[high], (delta[high] > 0).astype(int)),
        "n_high_confidence": int(high.sum()),
        "root_margin_mean": float(margins.mean()),
        "root_margin_positive_rate": float(np.mean(margins > 0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
