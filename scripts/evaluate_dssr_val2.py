#!/usr/bin/env python3
"""Apply the frozen DSSR K0--K3 bundle and emit the one Val2 Gate decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BUDGETS = (0.25, 0.50, 0.75)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select(scores: np.ndarray, ids: list[str], budget: float) -> np.ndarray:
    count = int(round(len(scores) * budget))
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), ids[i]))
    chosen = np.zeros(len(scores), dtype=bool)
    chosen[order[:count]] = True
    return chosen


def load_search(path: Path, ids: list[str]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        grouped[str(row["sample_id"])].append(row)
    result = defaultdict(list)
    for sample_id in ids:
        rows = grouped[sample_id]
        if len(rows) != 4 or any(row["cur_forced_arm"] != "search" for row in rows):
            raise RuntimeError(f"{sample_id}: expected Search N=4")
        for key, source in (("f1", "answer_f1"), ("response", "response_tokens"), ("observation", "observation_tokens")):
            result[key].append(float(np.mean([float(row[source]) for row in rows])))
    return {key: np.asarray(value) for key, value in result.items()}


def pca_transform(path: Path, values: np.ndarray) -> np.ndarray:
    model = np.load(path)
    return (values - model["mean"]) @ model["components"].T


def mlp_predictions(model_dir: Path, x: np.ndarray, output_dim: int) -> np.ndarray:
    import torch
    from torch import nn

    scaler = np.load(model_dir / "scaler.npz")
    z = ((x - scaler["mean"]) / scaler["scale"]).astype(np.float32)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(z.shape[1], 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, output_dim)
            )

        def forward(self, values):
            output = torch.sigmoid(self.network(values))
            return output.squeeze(-1) if output_dim == 1 else output

    predictions = []
    for seed in (1, 2, 3):
        model = MLP()
        model.load_state_dict(torch.load(model_dir / f"seed{seed}.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.inference_mode():
            predictions.append(model(torch.from_numpy(z)).numpy())
    return np.mean(predictions, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--probe-dir", type=Path, required=True)
    ap.add_argument("--search-outcomes", type=Path, required=True)
    ap.add_argument("--static-features", type=Path, required=True)
    ap.add_argument("--static-b3-dir", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--search-log", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if any("test" in str(value).lower() for value in vars(args).values()):
        raise RuntimeError("sealed Test path is forbidden in Val2 evaluation")

    ids = [line for line in args.ids.read_text().splitlines() if line]
    probes = [json.loads(line) for line in (args.probe_dir / "probes.jsonl").read_text().splitlines() if line]
    probe_hidden = np.load(args.probe_dir / "hidden_states.npz")
    static = np.load(args.static_features)
    if len(ids) != 128 or [str(x["sample_id"]) for x in probes] != ids:
        raise RuntimeError("Val2 Probe IDs/order mismatch")
    if [str(x) for x in probe_hidden["sample_ids"]] != ids or [str(x) for x in static["sample_ids"]] != ids:
        raise RuntimeError("Val2 feature IDs/order mismatch")
    search = load_search(args.search_outcomes, ids)
    probe_f1 = np.asarray([float(row["answer_f1"]) for row in probes])
    prompt_tokens = np.asarray([float(row["canonical_prompt_tokens"]) for row in probes])
    probe_tokens = np.asarray([float(row["response_tokens"]) for row in probes])
    delta = search["f1"] - probe_f1
    skip_regret = np.maximum(0.0, delta)

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
    prefix_scaler = np.load(args.models_dir / "k1_prefix_scaler.npz")
    prefix_z = (prefix - prefix_scaler["mean"]) / prefix_scaler["scale"]
    k1 = prefix_z[:, 0] - prefix_z[:, 1] - prefix_z[:, 2]

    static_z = pca_transform(args.static_b3_dir / "pca27.npz", static["layer27"].astype(np.float64))
    static_outcome = mlp_predictions(args.static_b3_dir / "b3", static_z, 2)
    k0 = static_outcome[:, 1] - static_outcome[:, 0]
    k2 = mlp_predictions(args.models_dir / "k2", confidence, 1)
    post_z = pca_transform(args.models_dir / "post_l27_pca64.npz", probe_hidden["layer27"].astype(np.float64))
    dynamics = np.column_stack(
        [
            [float(row["cosine_18_27"]) for row in probes],
            [float(row["cosine_27_36"]) for row in probes],
            [float(row["cosine_18_36"]) for row in probes],
            [float(row["relative_update_18_27"]) for row in probes],
            [float(row["relative_update_27_36"]) for row in probes],
        ]
    )
    k3 = mlp_predictions(args.models_dir / "k3", np.column_stack([k0, confidence, post_z, dynamics]), 1)
    scores = {"K0": k0, "K1": k1, "K2": k2, "K3": k3}

    always_cost = prompt_tokens + search["response"] + search["observation"]
    stop_cost = prompt_tokens + probe_tokens
    routed_search_cost = 2 * prompt_tokens + probe_tokens + search["response"] + search["observation"]
    policies: dict[str, dict[str, dict]] = {name: {} for name in scores}
    selections = {}
    random_values, oracle_values = {}, {}
    for budget in BUDGETS:
        key = str(budget)
        random_f1 = float(((1 - budget) * probe_f1 + budget * search["f1"]).mean())
        oracle_set = select(delta, ids, budget)
        oracle_f1 = float(np.where(oracle_set, search["f1"], probe_f1).mean())
        random_values[key] = random_f1
        oracle_values[key] = oracle_f1
        denom = oracle_f1 - random_f1
        for name, score in scores.items():
            chosen = select(score, ids, budget)
            selections[(name, budget)] = chosen
            f1 = float(np.where(chosen, search["f1"], probe_f1).mean())
            token_cost = float(np.where(chosen, routed_search_cost, stop_cost).mean())
            policies[name][key] = {
                "selected_search": int(chosen.sum()),
                "f1": f1,
                "recovery": float((f1 - random_f1) / denom) if abs(denom) > 1e-12 else 0.0,
                "regret_vs_oracle": oracle_f1 - f1,
                "token_cost": token_cost,
                "token_cost_ratio_vs_always_search": token_cost / float(always_cost.mean()),
                "response_tokens": float(np.where(chosen, probe_tokens + search["response"], probe_tokens).mean()),
                "observation_tokens": float(np.where(chosen, search["observation"], 0.0).mean()),
                "skip_risk_population_normalized": float(np.where(~chosen, skip_regret, 0.0).mean()),
            }

    gate4_wins = {}
    for budget in BUDGETS:
        key = str(budget)
        strongest = max((policies[name][key]["f1"] for name in ("K0", "K1", "K2")))
        gate4_wins[key] = {
            "k3_f1": policies["K3"][key]["f1"],
            "strongest_baseline_f1": strongest,
            "strict_win": policies["K3"][key]["f1"] > strongest,
        }
    always_f1 = float(search["f1"].mean())
    checks = {
        "gate1_recovery50": policies["K3"]["0.5"]["recovery"] >= 0.65,
        "gate2_f1_preservation50": policies["K3"]["0.5"]["f1"] >= always_f1 - 0.02,
        "gate3_token_cost50": policies["K3"]["0.5"]["token_cost_ratio_vs_always_search"] <= 0.65,
        "gate4_k3_beats_baselines_2_of_3": sum(int(x["strict_win"]) for x in gate4_wins.values()) >= 2,
    }
    gate_pass = all(checks.values())

    risk_curve = []
    for search_budget in np.linspace(0, 1, 21):
        chosen = select(k3, ids, float(search_budget))
        risk_curve.append(
            {
                "stop_coverage": float((~chosen).mean()),
                "search_budget": float(search_budget),
                "skip_risk_population_normalized": float(np.where(~chosen, skip_regret, 0.0).mean()),
                "skip_risk_conditional_on_stop": float(skip_regret[~chosen].mean()) if (~chosen).any() else 0.0,
            }
        )

    log_text = args.search_log.read_text(errors="replace")
    gen_seconds = sum(float(x) for x in re.findall(r"timing_s/gen:([0-9.]+)", log_text))
    step_seconds = sum(float(x) for x in re.findall(r"timing_s/step:([0-9.]+)", log_text))
    probe_summary = json.loads((args.probe_dir / "summary.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w") as handle:
        for i, sample_id in enumerate(ids):
            handle.write(json.dumps({
                "sample_id": sample_id,
                "probe_f1": float(probe_f1[i]),
                "search_f1_n4": float(search["f1"][i]),
                "skip_regret": float(skip_regret[i]),
                **{f"score_{name}": float(score[i]) for name, score in scores.items()},
            }, sort_keys=True) + "\n")
    decision = {
        "gate": "DSSR_VAL2_PASS" if gate_pass else "SELF_KNOWLEDGE_ROUTER_FAIL",
        "original_test_opened": False,
        "n": len(ids),
        "probe_f1": float(probe_f1.mean()),
        "always_search_f1": always_f1,
        "always_search_token_cost": float(always_cost.mean()),
        "policies": policies,
        "random_f1": random_values,
        "oracle_f1": oracle_values,
        "k3_vs_baselines": gate4_wins,
        "gate_checks": checks,
        "risk_coverage_k3": risk_curve,
        "measured_batch_wall_clock": {
            "probe_capture_seconds": probe_summary["generation_and_feature_wall_seconds"],
            "search_generation_seconds_sum_over_4_batches": gen_seconds,
            "search_step_seconds_sum_over_4_batches": step_seconds,
            "note": "measured acquisition latency; not FLOPs and not used as a Gate",
        },
        "frozen_artifact_sha256": {
            "bundle": sha256_file(args.models_dir / "frozen_bundle.json"),
            "val2_ids": sha256_file(args.ids),
            "probe": sha256_file(args.probe_dir / "probes.jsonl"),
            "probe_hidden": sha256_file(args.probe_dir / "hidden_states.npz"),
            "search": sha256_file(args.search_outcomes),
            "static_features": sha256_file(args.static_features),
            "predictions": sha256_file(predictions_path),
        },
        "test_read": False,
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({
        "gate": decision["gate"],
        "probe_f1": decision["probe_f1"],
        "always_search_f1": always_f1,
        "k3_at_50": policies["K3"]["0.5"],
        "gate_checks": checks,
        "k3_vs_baselines": gate4_wins,
        "original_test_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
