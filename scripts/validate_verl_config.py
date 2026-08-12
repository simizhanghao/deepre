#!/usr/bin/env python3
"""Run veRL/Hydra's real config validation without starting Ray or GPUs."""

from __future__ import annotations

import os

from verl.trainer import main_ppo


def _validated_only(config, task_runner_class) -> None:  # noqa: ARG001
    expected_adv = os.environ.get("ECA_EXPECT_ADV_ESTIMATOR", "").strip()
    expected_loss = os.environ.get("ECA_EXPECT_LOSS_AGG_MODE", "").strip()
    expected_norm = os.environ.get("ECA_EXPECT_NORM_ADV_BY_STD", "").strip().lower()
    actual_adv = str(config.algorithm.adv_estimator)
    actual_loss = str(config.actor_rollout_ref.actor.loss_agg_mode)
    actual_norm = str(bool(config.algorithm.norm_adv_by_std_in_grpo)).lower()
    if expected_adv and actual_adv != expected_adv:
        raise RuntimeError(f"adv_estimator mismatch: expected={expected_adv} actual={actual_adv}")
    if expected_loss and actual_loss != expected_loss:
        raise RuntimeError(f"loss_agg_mode mismatch: expected={expected_loss} actual={actual_loss}")
    if expected_norm and actual_norm != expected_norm:
        raise RuntimeError(f"norm_adv_by_std mismatch: expected={expected_norm} actual={actual_norm}")
    print(f"RESOLVED_ADV_ESTIMATOR={actual_adv}", flush=True)
    print(f"RESOLVED_LOSS_AGG_MODE={actual_loss}", flush=True)
    print(f"RESOLVED_NORM_ADV_BY_STD={actual_norm}", flush=True)
    print("VERL_CONFIG_VALIDATION_PASS", flush=True)


main_ppo.run_ppo = _validated_only
main_ppo.main()
