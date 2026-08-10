#!/usr/bin/env python3
"""Shim for already-patched veRL TaskRunner hooks. Prefer patch_verl_grpo_metrics.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_py = Path(__file__).with_name("patch_verl_grpo_metrics.py")
_spec = importlib.util.spec_from_file_location("patch_verl_grpo_metrics", _py)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

file_patch_task_runner = _mod.file_patch_task_runner
apply = _mod.apply
main = _mod.main

if __name__ == "__main__":
    main()
