#!/usr/bin/env python3
"""Fit and freeze DSSR K0--K3 using Train640 only; never read Val2/Test outcomes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from fit_evaluate_cur1_b0_b6 import (
    make_folds,
    pca_transform,
    randomized_pca,
    rankdata,
    train_mlp_ensemble,
)

BUDGETS = (0.25, 0.50, 0.75)
SEEDS = (1, 2, 3)
CONFIG = {
    "target": "max(0, F1_search_N1 - F1_probe_deterministic)",
    "outer_folds": 5,
    "outer_fold_seed": 2026081202,
    "inner_holdout_folds": 5,
    "inner_holdout_seed": 2026081202,
    "pca": {"dims": 64, "source": "Train-only post-answer L27"},
    "mlp": {
        "hidden_dims": [64, 32],
        "activation": "GELU",
        "output": "sigmoid scalar SkipRegret",
        "loss": "MSE",
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "max_epochs": 500,
        "patience": 40,
        "min_delta": 1e-6,
        "seeds": SEEDS,
    },
    "k1": "z(prefix20 entropy)-z(prefix20 top1/top2 margin)-z(prefix20 chosen-token logP)",
    "k2": ["answer_mean_logprob", "answer_p10_logprob", "answer_min_logprob", "answer_content_tokens", "closed_answer"],
    "k3": ["K0_B3_prior_oof_for_train", "K2", "post_L27_PCA64", "post_dynamics_5"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=list).encode()).hexdigest()


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def select(scores: np.ndarray, ids: list[str], budget: float) -> np.ndarray:
    count = int(round(len(scores) * budget))
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), ids[i]))
    chosen = np.zeros(len(scores), dtype=bool)
    chosen[order[:count]] = True
    return chosen


def ranking_diagnostics(scores: np.ndarray, ids: list[str], probe: np.ndarray, search: np.ndarray) -> dict:
    delta = search - probe
    values = {}
    for budget in BUDGETS:
        chosen = select(scores, ids, budget)
        oracle = select(delta, ids, budget)
        policy_f1 = float(np.where(chosen, search, probe).mean())
        oracle_f1 = float(np.where(oracle, search, probe).mean())
        random_f1 = float(((1 - budget) * probe + budget * search).mean())
        denom = oracle_f1 - random_f1
        values[str(budget)] = {
            "f1": policy_f1,
            "random_f1": random_f1,
            "oracle_f1": oracle_f1,
            "recovery": float((policy_f1 - random_f1) / denom) if abs(denom) > 1e-12 else 0.0,
        }
    return values


def scalar_mlp_ensemble(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    output_dir: Path | None,
) -> tuple[np.ndarray, dict]:
    import torch
    from torch import nn

    cfg = CONFIG["mlp"]
    mean, scale = fit_scaler(x_train)
    z_train = ((x_train - mean) / scale).astype(np.float32)
    z_predict = ((x_predict - mean) / scale).astype(np.float32)
    y = y_train.astype(np.float32)
    fit_index, stop_index = make_folds(len(y), CONFIG["inner_holdout_folds"], CONFIG["inner_holdout_seed"])[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class MLP(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
            )

        def forward(self, values):
            return torch.sigmoid(self.network(values)).squeeze(-1)

    def train_epochs(seed: int, indices: np.ndarray, epochs: int) -> MLP:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = MLP(z_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            model.train()
            order = rng.permutation(indices)
            for start in range(0, len(order), cfg["batch_size"]):
                batch = order[start : start + cfg["batch_size"]]
                prediction = model(torch.from_numpy(z_train[batch]).to(device))
                truth = torch.from_numpy(y[batch]).to(device)
                loss = ((prediction - truth) ** 2).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return model

    predictions, seed_rows = [], []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = MLP(z_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        rng = np.random.default_rng(seed)
        stop_x = torch.from_numpy(z_train[stop_index]).to(device)
        stop_y = torch.from_numpy(y[stop_index]).to(device)
        best_loss, best_epoch, stale = math.inf, 0, 0
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.train()
            order = rng.permutation(fit_index)
            for start in range(0, len(order), cfg["batch_size"]):
                batch = order[start : start + cfg["batch_size"]]
                pred = model(torch.from_numpy(z_train[batch]).to(device))
                truth = torch.from_numpy(y[batch]).to(device)
                loss = ((pred - truth) ** 2).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                heldout = float(((model(stop_x) - stop_y) ** 2).mean())
            if heldout < best_loss - cfg["min_delta"]:
                best_loss, best_epoch, stale = heldout, epoch, 0
            else:
                stale += 1
                if stale >= cfg["patience"]:
                    break
        final = train_epochs(seed, np.arange(len(y)), best_epoch)
        final.eval()
        with torch.inference_mode():
            predictions.append(final(torch.from_numpy(z_predict).to(device)).cpu().numpy())
        row = {"seed": seed, "selected_epoch": best_epoch, "inner_holdout_mse": best_loss}
        if output_dir is not None:
            model_path = output_dir / f"seed{seed}.pt"
            torch.save(final.state_dict(), model_path)
            row["model_sha256"] = sha256_file(model_path)
        seed_rows.append(row)
    record = {"input_dim": int(x_train.shape[1]), "seeds": seed_rows}
    if output_dir is not None:
        scaler_path = output_dir / "scaler.npz"
        np.savez(scaler_path, mean=mean, scale=scale)
        record["scaler_sha256"] = sha256_file(scaler_path)
    return np.mean(predictions, axis=0), record


def paired_outcomes(path: Path, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for line in path.read_text().splitlines():
        row = json.loads(line)
        grouped[str(row["sample_id"])][str(row["cur_forced_arm"])].append(float(row["answer_f1"]))
    internal, search = [], []
    for sample_id in ids:
        if len(grouped[sample_id]["internal"]) != 1 or len(grouped[sample_id]["search"]) != 1:
            raise RuntimeError(f"{sample_id}: expected exactly one paired outcome")
        internal.append(grouped[sample_id]["internal"][0])
        search.append(grouped[sample_id]["search"][0])
    return np.asarray(internal), np.asarray(search)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-features", type=Path, required=True)
    ap.add_argument("--paired-outcomes", type=Path, required=True)
    ap.add_argument("--probe-dir", type=Path, required=True)
    ap.add_argument("--static-b3-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if any("test" in str(value).lower() or "val2" in str(value).lower() for value in vars(args).values()):
        raise RuntimeError("Val2/Test paths are forbidden in Train-only model fitting")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    static_npz = np.load(args.static_features)
    probe_npz = np.load(args.probe_dir / "hidden_states.npz")
    ids = [str(x) for x in probe_npz["sample_ids"]]
    if ids != [str(x) for x in static_npz["sample_ids"]] or len(ids) != 640:
        raise RuntimeError("static/probe Train feature IDs differ")
    probes = [json.loads(line) for line in (args.probe_dir / "probes.jsonl").read_text().splitlines() if line]
    if [str(x["sample_id"]) for x in probes] != ids:
        raise RuntimeError("Probe JSONL order differs from hidden states")
    old_internal, search = paired_outcomes(args.paired_outcomes, ids)
    probe_f1 = np.asarray([float(x["answer_f1"]) for x in probes])
    target = np.maximum(0.0, search - probe_f1)

    confidence_names = list(CONFIG["k2"])
    confidence = np.column_stack(
        [
            [float(row["answer_mean_logprob"]) for row in probes],
            [float(row["answer_p10_logprob"]) for row in probes],
            [float(row["answer_min_logprob"]) for row in probes],
            [float(row["answer_content_tokens"]) for row in probes],
            [float(row["closed_answer"]) for row in probes],
        ]
    )
    prefix = np.column_stack(
        [
            [float(row["prefix20_mean_entropy"]) for row in probes],
            [float(row["prefix20_mean_margin"]) for row in probes],
            [float(row["prefix20_mean_logprob"]) for row in probes],
        ]
    )
    prefix_mean, prefix_scale = fit_scaler(prefix)
    prefix_z = (prefix - prefix_mean) / prefix_scale
    k1_score = prefix_z[:, 0] - prefix_z[:, 1] - prefix_z[:, 2]
    np.savez(args.output_dir / "k1_prefix_scaler.npz", mean=prefix_mean, scale=prefix_scale)

    # K0 Train scores are strictly out-of-fold to prevent stacking leakage in K3.
    static_h27 = static_npz["layer27"].astype(np.float64)
    k0_oof = np.zeros(len(ids), dtype=np.float64)
    folds = make_folds(len(ids), CONFIG["outer_folds"], CONFIG["outer_fold_seed"])
    for fold, (fit_idx, held_idx) in enumerate(folds):
        pca = randomized_pca(static_h27[fit_idx])
        x_fit, x_held = pca_transform(pca, static_h27[fit_idx]), pca_transform(pca, static_h27[held_idx])
        mu_internal, mu_search, _ = train_mlp_ensemble(
            f"K0_fold{fold}", x_fit, old_internal[fit_idx], search[fit_idx], x_held,
            args.output_dir / "oof_tmp" / f"k0_fold{fold}",
        )
        k0_oof[held_idx] = mu_search - mu_internal
        print(f"K0 OOF fold={fold + 1}/5", flush=True)

    post_h27 = probe_npz["layer27"].astype(np.float64)
    post_pca = randomized_pca(post_h27)
    post_z = pca_transform(post_pca, post_h27)
    post_pca_path = args.output_dir / "post_l27_pca64.npz"
    np.savez(post_pca_path, **post_pca)
    dynamics = np.column_stack(
        [
            [float(row["cosine_18_27"]) for row in probes],
            [float(row["cosine_27_36"]) for row in probes],
            [float(row["cosine_18_36"]) for row in probes],
            [float(row["relative_update_18_27"]) for row in probes],
            [float(row["relative_update_27_36"]) for row in probes],
        ]
    )
    k3_x = np.column_stack([k0_oof, confidence, post_z, dynamics])

    predictions = {"K0": k0_oof, "K1": k1_score}
    fitted: dict[str, Any] = {
        "K0": {"source": "frozen static B3", "train_feature": "strict 5-fold OOF prior"},
        "K1": {"training_free": True, "scaler_sha256": sha256_file(args.output_dir / "k1_prefix_scaler.npz")},
    }
    for name, values in (("K2", confidence), ("K3", k3_x)):
        oof = np.zeros(len(ids), dtype=np.float64)
        for fold, (fit_idx, held_idx) in enumerate(folds):
            pred, _ = scalar_mlp_ensemble(name, values[fit_idx], target[fit_idx], values[held_idx], None)
            oof[held_idx] = pred
        predictions[name] = oof
        _, record = scalar_mlp_ensemble(name, values, target, values[:1], args.output_dir / name.lower())
        fitted[name] = record
        print(f"{name} OOF/final fit complete", flush=True)

    np.savez(args.output_dir / "train_oof_predictions.npz", sample_ids=np.asarray(ids), target=target, **predictions)
    diagnostics = {
        name: {
            "spearman_skip_regret": spearman(score, target),
            "rmse_skip_regret": float(np.sqrt(np.mean((score - target) ** 2))),
            "policy": ranking_diagnostics(score, ids, probe_f1, search),
        }
        for name, score in predictions.items()
    }
    schema = {
        "K0": "frozen CUR-1 B3; OOF score only while training K3",
        "K1": CONFIG["k1"],
        "K2": confidence_names,
        "K3": {"k0_prior": 1, "confidence": confidence_names, "post_l27_pca": 64, "dynamics": 5},
    }
    artifact_hashes = {
        "static_features": sha256_file(args.static_features),
        "paired_outcomes": sha256_file(args.paired_outcomes),
        "probe_jsonl": sha256_file(args.probe_dir / "probes.jsonl"),
        "probe_hidden": sha256_file(args.probe_dir / "hidden_states.npz"),
        "static_b3_bundle": sha256_file(args.static_b3_dir / "candidate_bundle_manifest.json"),
        "post_pca": sha256_file(post_pca_path),
        "oof_predictions": sha256_file(args.output_dir / "train_oof_predictions.npz"),
    }
    decision = {
        "gate": "DSSR_K0_K3_TRAIN_FREEZE_PASS",
        "n": len(ids),
        "target_mean": float(target.mean()),
        "target_positive_rate": float(np.mean(target > 0)),
        "config": CONFIG,
        "config_sha256": json_hash(CONFIG),
        "feature_schema": schema,
        "feature_schema_sha256": json_hash(schema),
        "train_oof_diagnostics_not_a_selection_gate": diagnostics,
        "fitted_models": fitted,
        "artifact_hashes": artifact_hashes,
        "val2_outcomes_read": False,
        "test_read": False,
    }
    decision_path = args.output_dir / "train_freeze.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    bundle = {
        "gate": decision["gate"],
        "config_sha256": decision["config_sha256"],
        "feature_schema_sha256": decision["feature_schema_sha256"],
        "static_b3_dir": str(args.static_b3_dir),
        "static_b3_bundle_sha256": artifact_hashes["static_b3_bundle"],
        "post_l27_pca64_sha256": artifact_hashes["post_pca"],
        "k1_prefix_scaler_sha256": fitted["K1"]["scaler_sha256"],
        "k2": fitted["K2"],
        "k3": fitted["K3"],
        "train_freeze_sha256": sha256_file(decision_path),
        "val2_locked_for_next_stage": True,
        "test_sealed": True,
    }
    (args.output_dir / "frozen_bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps({"gate": decision["gate"], "target_mean": decision["target_mean"], "diagnostics": diagnostics, "test_read": False}, indent=2))


if __name__ == "__main__":
    main()
