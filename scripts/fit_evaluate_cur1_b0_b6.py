#!/usr/bin/env python3
"""Fit the frozen CUR-1 B0--B6 matrix and emit one Validation Unlock decision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

BUDGETS = (0.25, 0.50, 0.75)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SEEDS = (1, 2, 3)
COMPLEXITY = {"B1": 0, "B2": 1, "B4": 1, "B3": 2, "B5": 2, "B6": 3}
CONFIG = {
    "pca": {"dims": 64, "oversample": 16, "power_iterations": 3, "seed": 2026081201},
    "ridge": {"alphas": RIDGE_ALPHAS, "folds": 5, "seed": 2026081201},
    "mlp": {
        "hidden_dims": [64, 32],
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "max_epochs": 500,
        "early_stop_fraction": 0.20,
        "early_stop_patience": 40,
        "early_stop_min_delta": 1e-6,
        "seeds": SEEDS,
        "refit_full_train": True,
        "outputs": "sigmoid(mu_internal,mu_search)",
        "loss": "0.5*MSE_internal + 0.5*MSE_search",
    },
    "budgets": BUDGETS,
    "validation_unlock": {
        "recovery_50_min": 0.65,
        "f1_50_vs_always_search_tolerance": 0.02,
        "budgets_beating_strongest_noncandidate_min": 2,
        "occam_recovery50_tolerance": 0.02,
        "bootstrap_replicates": 10000,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=list).encode()).hexdigest()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def make_folds(n: int, count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    permutation = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(permutation, count)
    return [
        (np.concatenate([part for j, part in enumerate(parts) if j != index]), parts[index])
        for index in range(count)
    ]


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    mean, scale = fit_scaler(x)
    z = (x - mean) / scale
    y_mean = float(y.mean())
    dual = np.linalg.solve(z @ z.T + alpha * np.eye(len(z)), y - y_mean)
    coef = z.T @ dual
    return {"mean": mean, "scale": scale, "coef": coef, "intercept": y_mean}


def ridge_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return model["intercept"] + ((x - model["mean"]) / model["scale"]) @ model["coef"]


def select_ridge_alpha(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    folds = make_folds(len(y), CONFIG["ridge"]["folds"], CONFIG["ridge"]["seed"])
    for alpha in RIDGE_ALPHAS:
        mse = []
        for train_index, heldout_index in folds:
            model = ridge_fit(x[train_index], y[train_index], alpha)
            prediction = ridge_predict(model, x[heldout_index])
            mse.append(float(np.mean((prediction - y[heldout_index]) ** 2)))
        scores[str(alpha)] = float(np.mean(mse))
    selected = min(RIDGE_ALPHAS, key=lambda value: (scores[str(value)], value))
    return float(selected), scores


def randomized_pca(x: np.ndarray) -> dict[str, np.ndarray]:
    cfg = CONFIG["pca"]
    mean = x.mean(axis=0)
    centered = x - mean
    rank = cfg["dims"] + cfg["oversample"]
    rng = np.random.default_rng(cfg["seed"])
    omega = rng.standard_normal((centered.shape[1], rank))
    basis = centered @ omega
    for _ in range(cfg["power_iterations"]):
        basis = centered @ (centered.T @ basis)
        basis, _ = np.linalg.qr(basis, mode="reduced")
    basis, _ = np.linalg.qr(basis, mode="reduced")
    small = basis.T @ centered
    _, singular, vt = np.linalg.svd(small, full_matrices=False)
    components = vt[: cfg["dims"]]
    return {"mean": mean, "components": components, "singular_values": singular[: cfg["dims"]]}


def pca_transform(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return (x - model["mean"]) @ model["components"].T


def dynamics(features: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    h18, h27, h36 = features["layer18"], features["layer27"], features["layer36"]
    eps = 1e-8
    norms = [np.linalg.norm(state, axis=1) for state in (h18, h27, h36)]

    def cosine(left: np.ndarray, right: np.ndarray, nl: np.ndarray, nr: np.ndarray) -> np.ndarray:
        return np.sum(left * right, axis=1) / (nl * nr + eps)

    values = np.column_stack(
        [
            cosine(h18, h27, norms[0], norms[1]),
            cosine(h27, h36, norms[1], norms[2]),
            cosine(h18, h36, norms[0], norms[2]),
            np.linalg.norm(h27 - h18, axis=1) / (norms[0] + eps),
            np.linalg.norm(h36 - h27, axis=1) / (norms[1] + eps),
            norms[0],
            norms[1],
            norms[2],
            features["root_margin"],
        ]
    )
    names = [
        "cos_h18_h27", "cos_h27_h36", "cos_h18_h36",
        "relative_update_h18_h27", "relative_update_h27_h36",
        "norm_h18", "norm_h27", "norm_h36", "final_root_margin",
    ]
    return values, names


def load_outcomes(path: Path, sample_ids: list[str]) -> dict[str, np.ndarray]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for line in path.read_text().splitlines():
        row = json.loads(line)
        grouped[str(row["sample_id"])][str(row["cur_forced_arm"])].append(row)
    if set(grouped) != set(sample_ids):
        raise RuntimeError(f"outcome IDs differ: expected={len(sample_ids)} found={len(grouped)}")
    result: dict[str, list[float]] = defaultdict(list)
    for sample_id in sample_ids:
        arms = grouped[sample_id]
        if set(arms) != {"internal", "search"}:
            raise RuntimeError(f"{sample_id}: incomplete arms")
        for arm in ("internal", "search"):
            rows = arms[arm]
            result[f"f1_{arm}"].append(float(np.mean([row["answer_f1"] for row in rows])))
            result[f"em_{arm}"].append(float(np.mean([row["answer_em"] for row in rows])))
            result[f"response_{arm}"].append(float(np.mean([row["response_tokens"] for row in rows])))
            result[f"observation_{arm}"].append(float(np.mean([row["observation_tokens"] for row in rows])))
            result[f"search_count_{arm}"].append(float(np.mean([row["search_count"] for row in rows])))
    return {key: np.asarray(value, dtype=np.float64) for key, value in result.items()}


def train_mlp_ensemble(
    name: str,
    x_train: np.ndarray,
    y_internal: np.ndarray,
    y_search: np.ndarray,
    x_validation: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn

    cfg = CONFIG["mlp"]
    mean, scale = fit_scaler(x_train)
    z_train = ((x_train - mean) / scale).astype(np.float32)
    z_validation = ((x_validation - mean) / scale).astype(np.float32)
    targets = np.column_stack([y_internal, y_search]).astype(np.float32)
    holdout = make_folds(len(z_train), 5, 2026081201)[0]
    fit_index, stop_index = holdout
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class OutcomeMLP(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(width, cfg["hidden_dims"][0]),
                nn.GELU(),
                nn.Linear(cfg["hidden_dims"][0], cfg["hidden_dims"][1]),
                nn.GELU(),
                nn.Linear(cfg["hidden_dims"][1], 2),
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.network(values))

    def fit_for_epochs(seed: int, indices: np.ndarray, epochs: int) -> OutcomeMLP:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = OutcomeMLP(z_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        generator = np.random.default_rng(seed)
        model.train()
        for _ in range(epochs):
            order = generator.permutation(indices)
            for start in range(0, len(order), cfg["batch_size"]):
                batch = order[start : start + cfg["batch_size"]]
                inputs = torch.from_numpy(z_train[batch]).to(device)
                truth = torch.from_numpy(targets[batch]).to(device)
                prediction = model(inputs)
                loss = 0.5 * ((prediction[:, 0] - truth[:, 0]) ** 2).mean()
                loss = loss + 0.5 * ((prediction[:, 1] - truth[:, 1]) ** 2).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return model

    validation_predictions = []
    seed_records = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = OutcomeMLP(z_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        rng = np.random.default_rng(seed)
        best_loss = math.inf
        best_epoch = 0
        best_state = None
        stale = 0
        stop_x = torch.from_numpy(z_train[stop_index]).to(device)
        stop_y = torch.from_numpy(targets[stop_index]).to(device)
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.train()
            order = rng.permutation(fit_index)
            for start in range(0, len(order), cfg["batch_size"]):
                batch = order[start : start + cfg["batch_size"]]
                inputs = torch.from_numpy(z_train[batch]).to(device)
                truth = torch.from_numpy(targets[batch]).to(device)
                prediction = model(inputs)
                loss = 0.5 * ((prediction[:, 0] - truth[:, 0]) ** 2).mean()
                loss = loss + 0.5 * ((prediction[:, 1] - truth[:, 1]) ** 2).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                heldout = model(stop_x)
                heldout_loss = float(
                    0.5 * ((heldout[:, 0] - stop_y[:, 0]) ** 2).mean()
                    + 0.5 * ((heldout[:, 1] - stop_y[:, 1]) ** 2).mean()
                )
            if heldout_loss < best_loss - cfg["early_stop_min_delta"]:
                best_loss, best_epoch = heldout_loss, epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= cfg["early_stop_patience"]:
                    break
        if best_state is None:
            raise RuntimeError(f"{name} seed {seed}: early stop never produced a state")
        # Refit from the same initialization on all 640 questions for the
        # train-only selected epoch count.
        final_model = fit_for_epochs(seed, np.arange(len(z_train)), best_epoch)
        model_path = output_dir / f"seed{seed}.pt"
        torch.save(final_model.state_dict(), model_path)
        final_model.eval()
        with torch.inference_mode():
            prediction = final_model(torch.from_numpy(z_validation).to(device)).cpu().numpy()
        validation_predictions.append(prediction)
        seed_records.append(
            {
                "seed": seed,
                "best_epoch_train_internal_holdout": best_epoch,
                "best_holdout_loss": best_loss,
                "model_sha256": sha256(model_path),
            }
        )
    ensemble = np.mean(validation_predictions, axis=0)
    scaler_path = output_dir / "scaler.npz"
    np.savez(scaler_path, mean=mean, scale=scale)
    return ensemble[:, 0], ensemble[:, 1], {
        "input_dim": int(x_train.shape[1]),
        "scaler_sha256": sha256(scaler_path),
        "seeds": seed_records,
    }


def select_set(scores: np.ndarray, sample_ids: list[str], budget: float) -> np.ndarray:
    count = int(round(len(scores) * budget))
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), sample_ids[index]))
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:count]] = True
    return selected


def policy_arrays(selected: np.ndarray, outcomes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        metric: np.where(selected, outcomes[f"{metric}_search"], outcomes[f"{metric}_internal"])
        for metric in ("f1", "em", "response", "observation", "search_count")
    }


def expected_random_arrays(budget: float, outcomes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        metric: (1 - budget) * outcomes[f"{metric}_internal"] + budget * outcomes[f"{metric}_search"]
        for metric in ("f1", "em", "response", "observation", "search_count")
    }


def summarize_arrays(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    return {f"mean_{key}": float(value.mean()) for key, value in arrays.items()}


def bootstrap_difference(left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, Any]:
    difference = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(CONFIG["validation_unlock"]["bootstrap_replicates"], len(difference)))
    draws = difference[indices].mean(axis=1)
    return {
        "mean": float(difference.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--train-outcomes", type=Path, required=True)
    parser.add_argument("--validation-outcomes", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if "test" in " ".join(str(value).lower() for value in vars(args).values()):
        raise RuntimeError("sealed test path is forbidden in Validation selection")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_npz = np.load(args.train_features)
    validation_npz = np.load(args.validation_features)
    train = {key: train_npz[key].astype(np.float64) for key in ("layer18", "layer27", "layer36", "root_margin")}
    validation = {key: validation_npz[key].astype(np.float64) for key in ("layer18", "layer27", "layer36", "root_margin")}
    train_ids = [str(value) for value in train_npz["sample_ids"]]
    validation_ids = [str(value) for value in validation_npz["sample_ids"]]
    if len(train_ids) != 640 or len(validation_ids) != 128:
        raise RuntimeError("fresh Train/Validation sizes must be exactly 640/128")
    train_y = load_outcomes(args.train_outcomes, train_ids)
    validation_y = load_outcomes(args.validation_outcomes, validation_ids)
    delta_train = train_y["f1_search"] - train_y["f1_internal"]
    delta_validation = validation_y["f1_search"] - validation_y["f1_internal"]

    pca18 = randomized_pca(train["layer18"])
    pca27 = randomized_pca(train["layer27"])
    pca18_path, pca27_path = args.output_dir / "pca18.npz", args.output_dir / "pca27.npz"
    np.savez(pca18_path, **pca18)
    np.savez(pca27_path, **pca27)
    z18_train, z18_validation = pca_transform(pca18, train["layer18"]), pca_transform(pca18, validation["layer18"])
    z27_train, z27_validation = pca_transform(pca27, train["layer27"]), pca_transform(pca27, validation["layer27"])
    dynamic_train, dynamic_names = dynamics(train)
    dynamic_validation, _ = dynamics(validation)

    predictions: dict[str, np.ndarray] = {"B1": validation["root_margin"]}
    fitted: dict[str, Any] = {}
    for name, layer in (("B2", "layer27"), ("B4", "layer18")):
        alpha, cv = select_ridge_alpha(train[layer], delta_train)
        model = ridge_fit(train[layer], delta_train, alpha)
        predictions[name] = ridge_predict(model, validation[layer])
        model_path = args.output_dir / f"{name.lower()}_ridge.npz"
        np.savez(model_path, mean=model["mean"], scale=model["scale"], coef=model["coef"], intercept=model["intercept"], alpha=alpha)
        fitted[name] = {"alpha": alpha, "train_cv_mse": cv, "model_sha256": sha256(model_path)}

    mlp_inputs = {
        "B3": (z27_train, z27_validation),
        "B5": (z18_train, z18_validation),
        "B6": (np.column_stack([z18_train, dynamic_train]), np.column_stack([z18_validation, dynamic_validation])),
    }
    for name, (x_train, x_validation) in mlp_inputs.items():
        mu_internal, mu_search, record = train_mlp_ensemble(
            name, x_train, train_y["f1_internal"], train_y["f1_search"], x_validation,
            args.output_dir / name.lower(),
        )
        predictions[name] = mu_search - mu_internal
        fitted[name] = record

    diagnostics = {}
    for name, prediction in predictions.items():
        diagnostics[name] = {
            "spearman": spearman(prediction, delta_validation),
            "rmse": float(np.sqrt(np.mean((prediction - delta_validation) ** 2))),
            "mae": float(np.mean(np.abs(prediction - delta_validation))),
        }

    policy_values: dict[str, dict[str, Any]] = {name: {} for name in ["B0", *predictions]}
    policy_rows: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    oracle_values, random_values = {}, {}
    for budget in BUDGETS:
        key = str(budget)
        random_arrays = expected_random_arrays(budget, validation_y)
        oracle_set = select_set(delta_validation, validation_ids, budget)
        oracle_arrays = policy_arrays(oracle_set, validation_y)
        random_values[key] = summarize_arrays(random_arrays)
        oracle_values[key] = summarize_arrays(oracle_arrays)
        denominator = oracle_arrays["f1"].mean() - random_arrays["f1"].mean()
        policy_values["B0"][key] = {
            **summarize_arrays(random_arrays), "recovery": 0.0,
            "regret": float(oracle_arrays["f1"].mean() - random_arrays["f1"].mean()),
        }
        policy_rows[("B0", budget)] = random_arrays
        for name, prediction in predictions.items():
            selected = select_set(prediction, validation_ids, budget)
            arrays = policy_arrays(selected, validation_y)
            recovery = (arrays["f1"].mean() - random_arrays["f1"].mean()) / denominator
            policy_values[name][key] = {
                **summarize_arrays(arrays),
                "recovery": float(recovery),
                "regret": float(oracle_arrays["f1"].mean() - arrays["f1"].mean()),
                "selected_count": int(selected.sum()),
            }
            policy_rows[(name, budget)] = arrays

    # Primary Recovery@50, secondary mean recovery. Apply the frozen Occam
    # tie-break only when the fixed-policy paired bootstrap includes zero.
    candidates = list(predictions)
    candidates.sort(
        key=lambda name: (
            -policy_values[name]["0.5"]["recovery"],
            -float(np.mean([policy_values[name][str(b)]["recovery"] for b in BUDGETS])),
            COMPLEXITY[name], name,
        )
    )
    selected_name = candidates[0]
    occam = []
    for contender in candidates[1:]:
        gap = abs(policy_values[selected_name]["0.5"]["recovery"] - policy_values[contender]["0.5"]["recovery"])
        if gap >= CONFIG["validation_unlock"]["occam_recovery50_tolerance"]:
            continue
        paired = bootstrap_difference(
            policy_rows[(selected_name, 0.50)]["f1"], policy_rows[(contender, 0.50)]["f1"],
            2026081201 + int(contender[1:]),
        )
        stable = paired["ci95"][0] > 0 or paired["ci95"][1] < 0
        occam.append({"incumbent": selected_name, "contender": contender, "recovery_gap": gap, "paired": paired, "stable": stable})
        if not stable and COMPLEXITY[contender] < COMPLEXITY[selected_name]:
            selected_name = contender

    strongest = {}
    bootstrap_vs_strongest = {}
    wins = 0
    for budget in BUDGETS:
        key = str(budget)
        noncandidate = [name for name in policy_values if name != selected_name]
        strongest_name = max(noncandidate, key=lambda name: (policy_values[name][key]["mean_f1"], name))
        candidate_value = policy_values[selected_name][key]["mean_f1"]
        baseline_value = policy_values[strongest_name][key]["mean_f1"]
        beat = candidate_value > baseline_value
        wins += int(beat)
        strongest[key] = {"name": strongest_name, "mean_f1": baseline_value, "candidate_beats": beat}
        bootstrap_vs_strongest[key] = bootstrap_difference(
            policy_rows[(selected_name, budget)]["f1"], policy_rows[(strongest_name, budget)]["f1"],
            2026081300 + int(100 * budget),
        )

    always_search_f1 = float(validation_y["f1_search"].mean())
    checks = {
        "unlock_1_recovery50": policy_values[selected_name]["0.5"]["recovery"] >= 0.65,
        "unlock_2_quality_preservation": policy_values[selected_name]["0.5"]["mean_f1"] >= always_search_f1 - 0.02,
        "unlock_3_beats_strongest_at_2_of_3": wins >= 2,
    }
    unlock = all(checks.values())

    prediction_path = args.output_dir / "validation_predictions.jsonl"
    with prediction_path.open("w") as handle:
        for index, sample_id in enumerate(validation_ids):
            handle.write(json.dumps({
                "sample_id": sample_id,
                "delta_f1": float(delta_validation[index]),
                **{f"prediction_{name}": float(prediction[index]) for name, prediction in predictions.items()},
            }, sort_keys=True) + "\n")
    feature_schema = {"pca18": 64, "pca27": 64, "dynamics": dynamic_names, "root_margin": "final only"}
    artifact_hashes = {
        "train_features": sha256(args.train_features),
        "validation_features": sha256(args.validation_features),
        "train_outcomes": sha256(args.train_outcomes),
        "validation_outcomes": sha256(args.validation_outcomes),
        "train_split": sha256(args.train_split),
        "validation_split": sha256(args.validation_split),
        "pca18": sha256(pca18_path),
        "pca27": sha256(pca27_path),
        "predictions": sha256(prediction_path),
    }
    decision = {
        "gate": "VALIDATION_UNLOCK_PASS" if unlock else "VALIDATION_UNLOCK_FAIL",
        "test_opened": False,
        "candidate": selected_name,
        "selection_priority": ["Recovery@50", "mean Recovery@25/50/75", "Occam paired-bootstrap tie-break"],
        "config": CONFIG,
        "config_sha256": json_hash(CONFIG),
        "feature_schema": feature_schema,
        "feature_schema_sha256": json_hash(feature_schema),
        "diagnostics": diagnostics,
        "policy_values": policy_values,
        "random_values": random_values,
        "oracle_values": oracle_values,
        "always_search_f1": always_search_f1,
        "occam_comparisons": occam,
        "strongest_noncandidate": strongest,
        "paired_bootstrap_vs_strongest": bootstrap_vs_strongest,
        "unlock_checks": checks,
        "budgets_won": wins,
        "fitted_models": fitted,
        "artifact_hashes": artifact_hashes,
    }
    decision_path = args.output_dir / "validation_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    bundle = {
        "candidate": selected_name,
        "validation_gate": decision["gate"],
        "candidate_model_hashes": fitted.get(selected_name, {"frozen_baseline": selected_name}),
        "pca_hashes": {"pca18": artifact_hashes["pca18"], "pca27": artifact_hashes["pca27"]},
        "feature_schema_sha256": decision["feature_schema_sha256"],
        "training_config_sha256": decision["config_sha256"],
        "seed_list": list(SEEDS),
        "split_hashes": {"train": artifact_hashes["train_split"], "validation": artifact_hashes["validation_split"]},
        "validation_decision_sha256": sha256(decision_path),
        "test_lock_created": False,
    }
    (args.output_dir / "candidate_bundle_manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps({
        "gate": decision["gate"], "candidate": selected_name, "unlock_checks": checks,
        "budgets_won": wins, "recovery50": policy_values[selected_name]["0.5"]["recovery"],
        "f1_at_50": policy_values[selected_name]["0.5"]["mean_f1"], "always_search_f1": always_search_f1,
        "test_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
