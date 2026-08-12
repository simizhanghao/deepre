#!/usr/bin/env python3
"""Verify trainer dispatch equals the official RF++-baseline function."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-npz", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl.protocol import DataProto
    from verl.trainer.ppo import core_algos
    from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories

    capture = Path(args.capture_npz).resolve()
    raw = np.load(capture, allow_pickle=False)
    rewards = torch.from_numpy(raw["token_level_rewards"]).float()
    mask = torch.from_numpy(raw["response_mask"]).bool()
    identities = [json.loads(x) for x in raw["identity_json"]]
    uids = np.asarray([row["uid"] for row in identities], dtype=object)
    config = OmegaConf.create({"gamma": 1.0})

    direct_adv, direct_returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
        token_level_rewards=rewards.clone(), response_mask=mask, index=uids, config=config
    )
    data = DataProto.from_dict(
        tensors={"token_level_rewards": rewards.clone(), "response_mask": mask},
        non_tensors={"uid": uids},
    )
    dispatched = compute_advantage_for_multi_trajectories(
        data,
        batch_keys=[f"parity_{i}_0" for i in range(len(identities))],
        adv_estimator=core_algos.AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        gamma=1.0,
        lam=1.0,
        num_repeat=4,
        config=config,
    )
    adv_delta = float((dispatched.batch["advantages"] - direct_adv).abs().max())
    ret_delta = float((dispatched.batch["returns"] - direct_returns).abs().max())
    resolved = core_algos.get_adv_estimator_fn(
        core_algos.AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
    )
    report = {
        "gate": "RFPP_ESTIMATOR_PARITY_PASS" if adv_delta == 0.0 and ret_delta == 0.0 else "RFPP_ESTIMATOR_PARITY_FAIL",
        "capture": str(capture),
        "n_trajectories": len(identities),
        "group_size_histogram": {
            str(int(size)): int(count)
            for size, count in zip(
                *np.unique(np.unique(uids, return_counts=True)[1], return_counts=True), strict=True
            )
        },
        "adv_estimator": "reinforce_plus_plus_baseline",
        "resolved_function": f"{resolved.__module__}.{resolved.__name__}",
        "max_abs_advantage_delta": adv_delta,
        "max_abs_returns_delta": ret_delta,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["gate"] != "RFPP_ESTIMATOR_PARITY_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
