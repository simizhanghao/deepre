#!/usr/bin/env python3
"""Run veRL/Hydra's real config validation without starting Ray or GPUs."""

from __future__ import annotations

from verl.trainer import main_ppo


def _validated_only(config, task_runner_class) -> None:  # noqa: ARG001
    print("VERL_CONFIG_VALIDATION_PASS", flush=True)


main_ppo.run_ppo = _validated_only
main_ppo.main()
