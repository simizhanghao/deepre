#!/usr/bin/env python3
"""Patch veRL V1 trainer to log Phase-3B2 answer/format/search/zero-std metrics.

Idempotent. Safe to run every launch before main_ppo.
"""

from __future__ import annotations

import importlib
import sys


def apply() -> str:
    # Ensure repo is importable when launched inside eca-verl.
    repo = "/workspace/deepresearch"
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from src.rl.phase3b_metrics import compute_phase3b_batch_metrics, summarize_console_line

    mod = importlib.import_module("verl.trainer.ppo.v1.trainer_base")
    cls = mod.PPOTrainer
    if getattr(cls._compute_metrics, "_phase3b_metrics_patched", False):
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
            extras = data["extra_fields"].tolist()
            uids = data["uid"].tolist()
            # Sequence score = sum of token-level rm_scores (nested or padded).
            rm = data["rm_scores"]
            try:
                # Nested tensor path: sum last dim per sample.
                seq_scores = rm.to_padded_tensor().sum(-1).tolist()  # type: ignore[attr-defined]
            except Exception:
                try:
                    import torch

                    t = rm if torch.is_tensor(rm) else torch.as_tensor(rm)
                    seq_scores = t.sum(-1).tolist()
                except Exception:
                    seq_scores = None

            extras_np = [extras[i] for i in range(len(extras)) if non_padding_mask[i]]
            uids_np = [uids[i] for i in range(len(uids)) if non_padding_mask[i]]
            scores_np = None
            if seq_scores is not None:
                scores_np = [seq_scores[i] for i in range(len(seq_scores)) if non_padding_mask[i]]

            phase_m = compute_phase3b_batch_metrics(extras_np, uids=uids_np, sequence_scores=scores_np)
            metrics.update(phase_m)
            # Compact console line so Ray spam does not hide the 5 key signals.
            if phase_m:
                print(
                    f"[phase3b] step={global_steps} {summarize_console_line({**metrics, **phase_m})}",
                    flush=True,
                )
        except Exception as exc:  # never break training for metrics
            print(f"[phase3b] metrics patch skipped: {type(exc).__name__}: {exc}", flush=True)
        return metrics

    _compute_metrics._phase3b_metrics_patched = True  # type: ignore[attr-defined]
    cls._compute_metrics = _compute_metrics
    return "patched"


if __name__ == "__main__":
    print(apply())
