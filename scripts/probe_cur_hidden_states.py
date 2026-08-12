#!/usr/bin/env python3
"""Gate 0C: question-level 5-fold linear probes on fixed CUR layers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from analyze_cur0_margin import auroc, spearman


def make_folds(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    perm = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(perm, n_splits)
    return [
        (np.concatenate([part for j, part in enumerate(parts) if j != i]), parts[i])
        for i in range(n_splits)
    ]


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    mean = x_train.mean(0)
    scale = x_train.std(0)
    scale[scale < 1e-12] = 1.0
    z_train = (x_train - mean) / scale
    z_test = (x_test - mean) / scale
    y_mean = float(y_train.mean())
    dual = np.linalg.solve(
        z_train @ z_train.T + alpha * np.eye(len(z_train)), y_train - y_mean
    )
    return y_mean + z_test @ z_train.T @ dual


def select_alpha(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    candidates = (.01, .1, 1., 10., 100., 1000.)
    scores = []
    for alpha in candidates:
        errors = []
        for train, test in make_folds(len(y), 4, seed):
            pred = ridge_predict(x[train], y[train], x[test], alpha)
            errors.append(float(np.mean((pred - y[test]) ** 2)))
        scores.append(float(np.mean(errors)))
    return float(candidates[int(np.argmin(scores))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=Path, required=True)
    ap.add_argument("--outcomes", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args()
    hidden = np.load(args.hidden)
    outcomes = {x["sample_id"]: x for x in map(json.loads, args.outcomes.read_text().splitlines())}
    ids = [str(x) for x in hidden["sample_ids"]]
    y = np.asarray([outcomes[x]["delta_f1"] for x in ids])
    cost = np.asarray([outcomes[x]["mean_search_count_do_search"] for x in ids])
    high = np.asarray([outcomes[x]["high_confidence_direction"] != "borderline" for x in ids])
    folds = make_folds(len(y), 5, args.seed)
    results = {}
    prediction_rows = [{"sample_id": sid, "delta_f1": float(value)} for sid, value in zip(ids, y)]
    for layer in (18, 27, 36):
        x = hidden[f"layer{layer}"].astype(np.float64)
        pred = np.empty(len(y))
        alphas = []
        for fold_index, (train, test) in enumerate(folds):
            alpha = select_alpha(x[train], y[train], args.seed + 100 + fold_index)
            pred[test] = ridge_predict(x[train], y[train], x[test], alpha)
            alphas.append(alpha)
        metrics = {
            "spearman": spearman(pred, y),
            "mae": float(np.mean(np.abs(pred - y))),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "direction_auroc_high_confidence": auroc(pred[high], (y[high] > 0).astype(int)),
            "n_high_confidence": int(high.sum()),
            "fold_alphas": alphas,
        }
        results[f"layer{layer}"] = metrics
        for row, value in zip(prediction_rows, pred):
            row[f"pred_layer{layer}"] = float(value)
    cost_summary = {
        "mean": float(cost.mean()), "std": float(cost.std()),
        "predictable": bool(cost.std() > 0),
        "verdict": "constant_cost_no_model_needed" if cost.std() == 0 else "fit_cost_model",
    }
    primary = results["layer27"]
    baseline_pred = np.full_like(y, y.mean())
    baseline = {
        "mae": float(np.mean(np.abs(baseline_pred - y))),
        "rmse": float(np.sqrt(np.mean((baseline_pred - y) ** 2))),
    }
    summary = {
        "gate": "GATE_0C_COMPLETE",
        "protocol": "question-level shuffled 5-fold OOF; train-fold StandardScaler + nested NumPy closed-form RidgeCV",
        "n": len(y),
        "primary_layer": 27,
        "layers": results,
        "search_count_cost": cost_summary,
        "constant_mean_baseline": baseline,
        "primary_rmse_improvement_fraction": float(1 - primary["rmse"] / baseline["rmse"]),
        "decision": "REVIEW_AFTER_PREREGISTERED_N8_BORDERLINE_LABEL_REFINEMENT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    args.output.with_name("probe_predictions.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in prediction_rows)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
