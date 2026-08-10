#!/usr/bin/env python3
"""Patch veRL so GRPO diagnostics run inside the Ray TaskRunner process.

Idempotent. Safe to run every launch before main_ppo.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

REPO = Path("/workspace/deepresearch")
MAIN_PPO = Path("/workspace/verl/verl/trainer/main_ppo.py")

MARKER_BEGIN = "# === GRPO_METRICS_HOOK_BEGIN ==="
MARKER_END = "# === GRPO_METRICS_HOOK_END ==="
# Keep loading the legacy filename inside already-patched TaskRunner bodies.
LEGACY_MARKER_BEGIN = "# === PHASE3B_METRICS_HOOK_BEGIN ==="

HOOK = f'''        {MARKER_BEGIN}
        try:
            import importlib.util as _ilu
            import sys as _sys
            from pathlib import Path as _Path

            _repo = _Path("{REPO}")
            if str(_repo) not in _sys.path:
                _sys.path.insert(0, str(_repo))
            _patch_py = _repo / "scripts" / "patch_verl_grpo_metrics.py"
            _spec = _ilu.spec_from_file_location("patch_verl_grpo_metrics", _patch_py)
            _mod = _ilu.module_from_spec(_spec)
            assert _spec and _spec.loader
            _spec.loader.exec_module(_mod)
            print(f"[grpo] TaskRunner metrics: {{_mod.apply()}}", flush=True)
        except Exception as _grpo_exc:
            print(
                f"[grpo] TaskRunner metrics hook failed: "
                f"{{type(_grpo_exc).__name__}}: {{_grpo_exc}}",
                flush=True,
            )
        {MARKER_END}
'''

RUN_DEF = '    def run(self, config: DictConfig):\n        """Run the PPO training process."""\n'
OLD_NEXT = RUN_DEF + "        configure_verl_logging()\n"


def file_patch_task_runner(path: Path = MAIN_PPO) -> str:
    """Inject apply() at the start of TaskRunnerV1.run (idempotent)."""
    text = path.read_text(encoding="utf-8")
    if MARKER_BEGIN in text or LEGACY_MARKER_BEGIN in text:
        return "already_file_patched"

    if OLD_NEXT not in text:
        raise RuntimeError(f"Unexpected TaskRunnerV1.run body in {path}; refuse to patch")

    backup = path.with_suffix(path.suffix + ".bak_grpo_metrics")
    if not backup.exists():
        shutil.copy2(path, backup)

    replacement = RUN_DEF + HOOK + "        configure_verl_logging()\n"
    path.write_text(text.replace(OLD_NEXT, replacement, 1), encoding="utf-8")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return "file_patched"


def apply() -> str:
    """Monkeypatch PPOTrainer._compute_metrics in the *current* process."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    from src.rl.grpo_metrics import compute_grpo_batch_metrics as _compute_batch
    from src.rl.grpo_metrics import summarize_console_line

    log_tag = "grpo"

    mod = importlib.import_module("verl.trainer.ppo.v1.trainer_base")
    cls = mod.PPOTrainer
    if getattr(cls._compute_metrics, "_grpo_metrics_patched", False):
        return "already_patched"

    orig = cls._compute_metrics

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps=None, epoch=None):  # noqa: ANN001
        orig(self, batch, metrics, timing_raw, global_steps=global_steps, epoch=epoch)
        try:
            import numpy as np
            import transfer_queue as tq

            non_padding_mask = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields", "uid", "rm_scores", "response_mask"],
            )
            extras = list(data["extra_fields"])
            uids = list(data["uid"])
            rm = data["rm_scores"]
            seq_scores = None
            try:
                seq_scores = rm.to_padded_tensor().sum(-1).tolist()  # type: ignore[attr-defined]
            except Exception:
                try:
                    import torch

                    if torch.is_tensor(rm):
                        seq_scores = rm.sum(-1).tolist()
                    elif hasattr(rm, "sum"):
                        seq_scores = list(rm.sum(-1))
                    else:
                        seq_scores = [float(x) for x in list(rm)]
                except Exception:
                    seq_scores = None

            extras_np = [extras[i] for i in range(len(extras)) if non_padding_mask[i]]
            uids_np = [uids[i] for i in range(len(uids)) if non_padding_mask[i]]
            scores_np = None
            if seq_scores is not None:
                scores_np = [seq_scores[i] for i in range(len(seq_scores)) if non_padding_mask[i]]

            phase_m = _compute_batch(extras_np, uids=uids_np, sequence_scores=scores_np)
            metrics.update(phase_m)
            if phase_m:
                print(
                    f"[{log_tag}] step={global_steps} {summarize_console_line({**metrics, **phase_m})}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[{log_tag}] metrics patch skipped: {type(exc).__name__}: {exc}", flush=True)
        return metrics

    _compute_metrics._grpo_metrics_patched = True  # type: ignore[attr-defined]
    cls._compute_metrics = _compute_metrics
    return f"patched:{log_tag}"


def main() -> None:
    status_file = file_patch_task_runner()
    status_apply = apply()
    print(f"file={status_file} apply={status_apply}")


if __name__ == "__main__":
    main()
