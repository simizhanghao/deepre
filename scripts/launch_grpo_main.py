#!/usr/bin/env python3
"""Same-process entry: apply Phase-3B metrics monkeypatch, then veRL main_ppo.

Hydra CLI overrides are forwarded unchanged:
  python scripts/launch_grpo_main.py algorithm.adv_estimator=grpo ...
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    scripts = REPO / "scripts"
    hydra_args = list(sys.argv[1:])  # keep overrides; do not leak into patch CLIs

    # File patch (idempotent) for sgl055 pause/continue API.
    sgl = _load("patch_verl_sgl055_compat", scripts / "patch_verl_sgl055_compat.py")
    saved = sys.argv
    sys.argv = [str(scripts / "patch_verl_sgl055_compat.py")]
    try:
        sgl.main()
    finally:
        sys.argv = saved

    # File-patch TaskRunnerV1.run (Ray actor) + driver apply. Driver-only
    # monkeypatch never reached _compute_metrics in prior 3B2 runs.
    metrics_mod = _load("patch_verl_phase3b_metrics", scripts / "patch_verl_phase3b_metrics.py")
    file_status = metrics_mod.file_patch_task_runner()
    apply_status = metrics_mod.apply()
    print(f"[launch] phase3b metrics: file={file_status} apply={apply_status}", flush=True)

    # Mimic `python -m verl.trainer.main_ppo ...`
    sys.argv = ["verl.trainer.main_ppo", *hydra_args]
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")


if __name__ == "__main__":
    main()
