#!/usr/bin/env python3
"""Fit and freeze the single conservative Step Preference Gate on S1 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SEEDS = (1, 2, 3)
OUTER_FOLDS = 5
FOLD_SEED = 2026081301
CONFIG = {
    "label": "lexicographic preference; SEARCH=1, CONTINUE=0",
    "loss": "max(abs(delta_F1),0.02) weighted BCE with SEARCH pos_weight=N_continue/N_search",
    "pca": {"dims": 64, "fit": "outer-train only during OOF; full Train for frozen refit"},
    "outer_folds": OUTER_FOLDS,
    "group": "sample_id/question",
    "hidden_dims": [64, 32],
    "activation": "GELU",
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "max_epochs": 400,
    "patience": 35,
    "min_delta": 1e-5,
    "seeds": SEEDS,
    "threshold": "minimum paired SearchCalls subject to >=95% positive Search-regret capture; token cost tie-break",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_folds(groups: np.ndarray, count: int, seed: int):
    unique = np.asarray(sorted(set(groups.tolist())))
    parts = np.array_split(np.random.default_rng(seed).permutation(unique), count)
    folds = []
    for held_groups in parts:
        held = np.isin(groups, held_groups)
        folds.append((np.flatnonzero(~held), np.flatnonzero(held)))
    return folds


def pca_fit(x: np.ndarray, dims: int = 64):
    mean = x.mean(axis=0)
    centered = x - mean
    rng = np.random.default_rng(FOLD_SEED)
    basis = centered @ rng.standard_normal((centered.shape[1], dims + 16))
    for _ in range(3):
        basis = centered @ (centered.T @ basis)
        basis, _ = np.linalg.qr(basis, mode="reduced")
    basis, _ = np.linalg.qr(basis, mode="reduced")
    _, singular, vt = np.linalg.svd(basis.T @ centered, full_matrices=False)
    return mean, vt[:dims], singular[:dims]


def scale_fit(x: np.ndarray):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def full_b3_scores(static_features: Path, model_dir: Path) -> tuple[list[str], np.ndarray]:
    import torch
    from torch import nn

    data = np.load(static_features)
    ids = [str(value) for value in data["sample_ids"]]
    pca = np.load(model_dir.parent / "pca27.npz")
    z = (data["layer27"].astype(np.float64) - pca["mean"]) @ pca["components"].T
    scaler = np.load(model_dir / "scaler.npz")
    z = ((z - scaler["mean"]) / scaler["scale"]).astype(np.float32)

    class B3(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2)
            )

        def forward(self, values):
            return torch.sigmoid(self.network(values))

    predictions = []
    for seed in SEEDS:
        model = B3()
        model.load_state_dict(torch.load(model_dir / f"seed{seed}.pt", map_location="cpu"))
        model.eval()
        with torch.inference_mode():
            outcome = model(torch.from_numpy(z)).numpy()
        predictions.append(outcome[:, 1] - outcome[:, 0])
    return ids, np.mean(predictions, axis=0)


def train_ensemble(x, y, weights, groups, predict_x, seed_epochs=None):
    import torch
    from torch import nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = float(np.sum(y == 0) / max(1, np.sum(y == 1)))

    class Gate(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
            )

        def forward(self, values):
            return self.network(values).squeeze(-1)

    def fit(seed, indices, epochs):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = Gate(x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        rng = np.random.default_rng(seed)
        for _ in range(max(1, epochs)):
            order = rng.permutation(indices)
            for start in range(0, len(order), CONFIG["batch_size"]):
                batch = order[start:start + CONFIG["batch_size"]]
                logits = model(torch.from_numpy(x[batch]).to(device))
                truth = torch.from_numpy(y[batch]).to(device)
                sample_weight = torch.from_numpy(weights[batch]).to(device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, truth, weight=sample_weight,
                    pos_weight=torch.tensor(pos_weight, device=device), reduction="mean",
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return model

    predictions, records, states = [], [], []
    inner_train, inner_stop = grouped_folds(groups, 5, FOLD_SEED + 1)[0]
    for seed_idx, seed in enumerate(SEEDS):
        if seed_epochs is None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = Gate(x.shape[1]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
            rng = np.random.default_rng(seed)
            best, best_epoch, stale = math.inf, 1, 0
            for epoch in range(1, CONFIG["max_epochs"] + 1):
                model.train()
                order = rng.permutation(inner_train)
                for start in range(0, len(order), CONFIG["batch_size"]):
                    batch = order[start:start + CONFIG["batch_size"]]
                    logits = model(torch.from_numpy(x[batch]).to(device))
                    truth = torch.from_numpy(y[batch]).to(device)
                    sw = torch.from_numpy(weights[batch]).to(device)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, truth, weight=sw,
                        pos_weight=torch.tensor(pos_weight, device=device), reduction="mean",
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.inference_mode():
                    logits = model(torch.from_numpy(x[inner_stop]).to(device))
                    truth = torch.from_numpy(y[inner_stop]).to(device)
                    sw = torch.from_numpy(weights[inner_stop]).to(device)
                    held = float(torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, truth, weight=sw,
                        pos_weight=torch.tensor(pos_weight, device=device), reduction="mean",
                    ))
                if held < best - CONFIG["min_delta"]:
                    best, best_epoch, stale = held, epoch, 0
                else:
                    stale += 1
                    if stale >= CONFIG["patience"]:
                        break
        else:
            best, best_epoch = float("nan"), int(seed_epochs[seed_idx])
        final = fit(seed, np.arange(len(y)), best_epoch)
        final.eval()
        with torch.inference_mode():
            predictions.append(torch.sigmoid(final(torch.from_numpy(predict_x).to(device))).cpu().numpy())
        records.append({"seed": seed, "selected_epoch": best_epoch, "inner_weighted_bce": best})
        states.append({key: value.detach().cpu() for key, value in final.state_dict().items()})
    return np.mean(predictions, axis=0), records, states


def select_threshold(prob, delta_f1, search_calls, continue_calls, search_tokens, continue_tokens):
    regret = np.maximum(delta_f1, 0.0)
    total_regret = float(regret.sum())
    candidates = sorted(set([0.0, 1.0 + 1e-8] + [float(value) for value in prob]))
    rows = []
    for threshold in candidates:
        search = prob >= threshold
        capture = float(regret[search].sum() / total_regret) if total_regret else 1.0
        calls = float(np.where(search, search_calls, continue_calls).sum())
        tokens = float(np.where(search, search_tokens, continue_tokens).sum())
        f1 = float(np.where(search, delta_f1 + 0.0, 0.0).sum())  # delta relative to Continue
        rows.append({"threshold": threshold, "capture": capture, "calls": calls, "tokens": tokens, "delta_f1_vs_continue_sum": f1, "search_rate": float(search.mean())})
    eligible = [row for row in rows if row["capture"] >= 0.95 - 1e-12]
    return min(eligible, key=lambda row: (row["calls"], row["tokens"], -row["threshold"])), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--root-oof", type=Path, required=True)
    ap.add_argument("--root-static-features", type=Path, required=True)
    ap.add_argument("--root-model-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if any(word in str(value).lower() for value in vars(args).values() for word in ("val3", "test")):
        raise RuntimeError("Val3/Test inputs are forbidden during Step-Gate fitting")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    features = np.load(args.features)
    groups = np.asarray([str(value) for value in features["sample_ids"]])
    hidden = features["layer27"].astype(np.float64)
    scalars = features["scalars"].astype(np.float64)
    y = features["label"].astype(np.float32)
    delta = features["delta_f1"].astype(np.float64)
    weights = np.maximum(np.abs(delta), 0.02).astype(np.float32)
    if len(y) != 1022 or int(y.sum()) != 74:
        raise RuntimeError(f"frozen label contract changed: n={len(y)} search={int(y.sum())}")

    root = np.load(args.root_oof)
    root_map = {str(sample_id): float(score) for sample_id, score in zip(root["sample_ids"], root["K0"])}
    root_oof = np.asarray([root_map[sample_id] for sample_id in groups], dtype=np.float64)[:, None]
    oof = np.zeros(len(y), dtype=np.float64)
    fold_rows, epochs_by_seed = [], [[] for _ in SEEDS]
    for fold, (train_idx, held_idx) in enumerate(grouped_folds(groups, OUTER_FOLDS, FOLD_SEED)):
        pca_mean, components, _ = pca_fit(hidden[train_idx])
        train_x = np.column_stack([(hidden[train_idx] - pca_mean) @ components.T, scalars[train_idx], root_oof[train_idx]])
        held_x = np.column_stack([(hidden[held_idx] - pca_mean) @ components.T, scalars[held_idx], root_oof[held_idx]])
        mean, scale = scale_fit(train_x)
        train_z = ((train_x - mean) / scale).astype(np.float32)
        held_z = ((held_x - mean) / scale).astype(np.float32)
        prediction, records, _ = train_ensemble(train_z, y[train_idx], weights[train_idx], groups[train_idx], held_z)
        oof[held_idx] = prediction
        for index, record in enumerate(records):
            epochs_by_seed[index].append(record["selected_epoch"])
        fold_rows.append({"fold": fold, "train_states": len(train_idx), "heldout_states": len(held_idx), "train_questions": len(set(groups[train_idx])), "heldout_questions": len(set(groups[held_idx])), "models": records})
        print(f"outer_fold={fold} complete", flush=True)

    selected, threshold_curve = select_threshold(
        oof, delta, features["search_calls"], features["continue_calls"],
        features["search_tokens"], features["continue_tokens"],
    )
    decision = oof >= selected["threshold"]
    policy_f1 = float(np.where(decision, features["search_f1"], features["continue_f1"]).mean())
    all_search_f1 = float(features["search_f1"].mean())
    all_search_calls = float(features["search_calls"].sum())

    full_ids, full_root_values = full_b3_scores(args.root_static_features, args.root_model_dir)
    full_root_map = dict(zip(full_ids, full_root_values))
    full_root = np.asarray([full_root_map[sample_id] for sample_id in groups], dtype=np.float64)[:, None]
    pca_mean, components, singular = pca_fit(hidden)
    full_x = np.column_stack([(hidden - pca_mean) @ components.T, scalars, full_root])
    mean, scale = scale_fit(full_x)
    full_z = ((full_x - mean) / scale).astype(np.float32)
    final_epochs = [int(np.median(values)) for values in epochs_by_seed]
    _, final_records, states = train_ensemble(full_z, y, weights, groups, full_z, seed_epochs=final_epochs)
    pca_path, scaler_path = args.output_dir / "pca_l27.npz", args.output_dir / "scaler.npz"
    np.savez(pca_path, mean=pca_mean, components=components, singular_values=singular)
    np.savez(scaler_path, mean=mean, scale=scale)
    model_hashes = {}
    for seed, state in zip(SEEDS, states):
        path = args.output_dir / f"seed{seed}.pt"
        torch.save(state, path)
        model_hashes[f"seed{seed}"] = sha256_file(path)
    np.savez(
        args.output_dir / "oof_predictions.npz",
        branch_ids=features["branch_ids"], sample_ids=groups, probability=oof,
        label=y, decision_search=decision, delta_f1=delta,
    )
    threshold_path = args.output_dir / "threshold.json"
    threshold_payload = {
        "threshold": selected["threshold"], "positive_regret_capture": selected["capture"],
        "search_rate": selected["search_rate"], "paired_calls": selected["calls"],
        "paired_calls_vs_all_search_ratio": selected["calls"] / all_search_calls,
        "paired_tokens": selected["tokens"], "oof_policy_f1": policy_f1,
        "all_search_f1": all_search_f1, "f1_delta_vs_all_search": policy_f1 - all_search_f1,
    }
    threshold_path.write_text(json.dumps(threshold_payload, indent=2) + "\n")
    (args.output_dir / "threshold_curve.json").write_text(json.dumps(threshold_curve, indent=2) + "\n")
    summary = {
        "gate": "STEP_PREFERENCE_GATE_TRAIN_FREEZE_PASS",
        "n_states": len(y), "n_questions": len(set(groups)),
        "labels": {"search": int(y.sum()), "continue": int(len(y) - y.sum())},
        "config": CONFIG, "folds": fold_rows, "final_models": final_records,
        "threshold": threshold_payload,
        "feature_dim": int(full_z.shape[1]),
        "artifact_sha256": {
            "features": sha256_file(args.features), "pca": sha256_file(pca_path),
            "scaler": sha256_file(scaler_path), "threshold": sha256_file(threshold_path),
            **model_hashes,
        },
        "root_feature": {"oof": "DSSR train_oof_predictions.K0", "final": "frozen CUR1 full-Train B3"},
        "val3_read": False, "test_read": False,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
