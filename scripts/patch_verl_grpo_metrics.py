#!/usr/bin/env python3
"""Patch veRL so GRPO diagnostics run inside the Ray TaskRunner process.

Idempotent. Safe to run every launch before main_ppo.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(os.environ.get("ECA_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
_main_spec = importlib.util.find_spec("verl.trainer.main_ppo")
MAIN_PPO = Path(_main_spec.origin).resolve() if _main_spec and _main_spec.origin else Path()

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

    capture_dir = os.environ.get("ECA_ATTRIBUTION_CAPTURE_DIR", "").strip()
    forward_only = os.environ.get("ECA_FORWARD_ONLY_CAPTURE", "").strip() == "1"
    if capture_dir and not forward_only:
        raise RuntimeError("ECA_ATTRIBUTION_CAPTURE_DIR requires ECA_FORWARD_ONLY_CAPTURE=1")

    if forward_only and not getattr(cls._update_actor, "_eca_forward_only", False):
        def _forward_only_update_actor(self, batch, metrics):  # noqa: ANN001
            metrics["attribution/actor_update_skipped"] = 1.0
            return batch

        _forward_only_update_actor._eca_forward_only = True  # type: ignore[attr-defined]
        cls._update_actor = _forward_only_update_actor
        print(
            "[attribution] FORWARD_ONLY: actor backward/optimizer/scheduler are disabled",
            flush=True,
        )

    root_pivot = os.environ.get("ECA_ROOT_PIVOT", "0").strip().lower() in {"1", "true", "yes"}
    if root_pivot and not getattr(cls._update_actor, "_eca_root_pivot", False):
        orig_update_actor = cls._update_actor

        def _root_pivot_update_actor(self, batch, metrics):  # noqa: ANN001
            import transfer_queue as tq
            from src.rl.root_pivot import labels_from_boundaries

            extra = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            boundaries = [
                str((row.get("reward_extra_info") or {}).get("boundary", ""))
                for row in list(extra["extra_fields"])
            ]
            labels = labels_from_boundaries(boundaries)
            metrics["root_pivot/need_rows"] = float((labels == 1).sum())
            metrics["root_pivot/no_rows"] = float((labels == -1).sum())
            return orig_update_actor(self, batch, metrics)

        _root_pivot_update_actor._eca_root_pivot = True  # type: ignore[attr-defined]
        cls._update_actor = _root_pivot_update_actor
        print("[root-pivot] trainer boundary-label audit active", flush=True)

    if not getattr(cls._init_dataloader, "_eca_horizon_patched", False):
        orig_init_dataloader = cls._init_dataloader

        def _init_dataloader(self):  # noqa: ANN001
            orig_init_dataloader(self)
            raw_horizon = os.environ.get("ECA_SCHEDULE_HORIZON", "").strip()
            if raw_horizon:
                horizon = int(raw_horizon)
                if horizon < self.total_training_steps:
                    raise ValueError("ECA_SCHEDULE_HORIZON cannot be below segment target")
                updates = horizon * self.parameter_sync_step
                self.config.actor_rollout_ref.actor.optim.total_training_steps = updates
                if getattr(self, "use_critic", False):
                    self.config.critic.optim.total_training_steps = updates
                print(
                    f"[grpo] fixed scheduler horizon={horizon} segment_target={self.total_training_steps}",
                    flush=True,
                )

        _init_dataloader._eca_horizon_patched = True  # type: ignore[attr-defined]
        cls._init_dataloader = _init_dataloader

    if getattr(cls._compute_metrics, "_grpo_metrics_patched", False):
        return "already_patched"

    orig = cls._compute_metrics

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps=None, epoch=None):  # noqa: ANN001
        orig(self, batch, metrics, timing_raw, global_steps=global_steps, epoch=epoch)
        try:
            import numpy as np
            import torch
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

            tensor_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["advantages", "old_log_probs", "rollout_log_probs", "response_mask"],
            ).to_padded_tensor()
            valid_rows = torch.as_tensor(non_padding_mask, dtype=torch.bool)
            response_mask = tensor_data["response_mask"][valid_rows].bool()
            advantages = tensor_data["advantages"][valid_rows]
            old_lp = tensor_data["old_log_probs"][valid_rows]
            rollout_lp = tensor_data["rollout_log_probs"][valid_rows]
            valid_adv = advantages[response_mask].float()
            valid_delta = (old_lp - rollout_lp)[response_mask].float()
            if valid_adv.numel():
                phase_m["grpo/advantage_std"] = float(valid_adv.std(unbiased=False))
                phase_m["grpo/positive_adv_token_rate"] = float((valid_adv > 0).float().mean())
            if valid_delta.numel():
                ratios = torch.exp(valid_delta.clamp(min=-20, max=20))
                for q in (0.01, 0.10, 0.50, 0.90, 0.99):
                    phase_m[f"rollout_corr_diag/importance_ratio_p{int(q * 100):02d}"] = float(
                        torch.quantile(ratios, q)
                    )
                max_abs = float(valid_delta.abs().max())
                # This is a trajectory-level correction diagnostic, not an
                # exact-backend parity test. In async multi-turn training the
                # behavior and recomputed policy logprobs need not be identical.
                phase_m["rollout_corr_diag/max_abs_logprob_delta"] = max_abs
            metrics.update(phase_m)

            cur_capture = os.environ.get("ECA_CUR_CAPTURE_JSONL", "").strip()
            if cur_capture:
                cur_path = Path(cur_capture)
                cur_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cur_path, "a", encoding="utf-8") as handle:
                    for uid, extra in zip(uids_np, extras_np, strict=True):
                        reward = extra.get("reward_extra_info") or {}
                        agent_metrics = extra.get("metrics") or {}
                        row = {
                            "step": int(global_steps),
                            "uid": str(uid),
                            "sample_id": str(extra.get("sample_id", reward.get("sample_id", ""))),
                            "cur_forced_arm": str(extra.get("cur_forced_arm", reward.get("cur_forced_arm", ""))),
                            "canonical_prompt_sha256": str(extra.get("canonical_prompt_sha256", "")),
                            "canonical_prompt_len": int(extra.get("root_prompt_len", 0)),
                            "cur_forced_prefix_ids": [int(x) for x in extra.get("cur_forced_prefix_ids", [])],
                            "cur_forced_action_valid": int(extra.get("cur_forced_action_valid", 0)),
                            "cur_policy_failure": int(extra.get("cur_policy_failure", 0)),
                            "cur_forbidden_search_attempts": int(extra.get("cur_forbidden_search_attempts", 0)),
                            "route_first": str(extra.get("route_first", "none")),
                            "finish": int(extra.get("finish", 0)),
                            "search_count": int(extra.get("search_count", 0)),
                            "duplicate_query_count": int(extra.get("duplicate_query_count", 0)),
                            "observation_tokens": int(extra.get("observation_tokens", 0)),
                            "response_tokens": int(extra.get("response_tokens", 0)),
                            "assistant_tokens": int(agent_metrics.get("assistant_tokens", 0)),
                            "generation_seconds": float(agent_metrics.get("generate_sequences", 0.0)),
                            "answer_f1": float(reward.get("answer_f1", 0.0)),
                            "answer_em": float(reward.get("answer_em", reward.get("em", 0.0))),
                            "format": float(reward.get("format", 0.0)),
                            "pred": str(reward.get("pred", "")),
                            "gold": str(reward.get("gold", "")),
                        }
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"[cur] captured step={global_steps} rows={len(extras_np)} path={cur_path}", flush=True)

            capture_root = os.environ.get("ECA_ATTRIBUTION_CAPTURE_DIR", "").strip()
            if capture_root:
                capture_path = Path(capture_root)
                capture_path.mkdir(parents=True, exist_ok=True)
                numeric = tq.kv_batch_get(
                    keys=batch.keys,
                    partition_id=batch.partition_id,
                    select_fields=[
                        "responses", "response_mask", "rm_scores",
                        "old_log_probs", "rollout_log_probs", "advantages",
                    ],
                ).to_padded_tensor()
                keep = torch.as_tensor(non_padding_mask, dtype=torch.bool)

                def _cpu(name, dtype=None):
                    value = numeric[name][keep].detach().cpu()
                    if dtype is not None:
                        value = value.to(dtype=dtype)
                    return value.numpy()

                reward_keys = (
                    "total_reward", "answer_reward", "evidence_reward",
                    "cost_reward", "format_reward",
                )
                component_rows = []
                identity_rows = []
                prompt_rows = []
                for slot, (uid, extra) in enumerate(zip(uids_np, extras_np, strict=True)):
                    info = extra.get("reward_extra_info") or {}
                    component_rows.append([float(info.get(key, float("nan"))) for key in reward_keys])
                    identity_rows.append({
                        "batch_id": int(global_steps),
                        "step_slot": int(slot),
                        "uid": str(uid),
                        "sample_id": str(extra.get("sample_id", "")),
                        "boundary": str(info.get("boundary", "")),
                        "route_first": str(extra.get("route_first", "none")),
                        "route_token_start": int(extra.get("route_token_start", 0)),
                        "route_token_end": int(extra.get("route_token_end", 0)),
                        "search_count": int(extra.get("search_count", 0)),
                        "response_span_len": int(extra.get("response_tokens", 0)),
                        "prompt_hash": str(extra.get("canonical_prompt_sha256", "")),
                    })
                    prompt_rows.append({
                        "sample_id": str(extra.get("sample_id", "")),
                        "boundary": str(info.get("boundary", "")),
                        "canonical_prompt_sha256": str(extra.get("canonical_prompt_sha256", "")),
                        "canonical_prompt_ids": [int(x) for x in extra.get("canonical_prompt_ids", [])],
                    })

                step = int(global_steps)
                tmp = capture_path / f"step_{step:03d}.npz.tmp"
                final = capture_path / f"step_{step:03d}.npz"
                with open(tmp, "wb") as handle:
                    np.savez_compressed(
                        handle,
                        response_token_ids=_cpu("responses", torch.int32),
                        response_mask=_cpu("response_mask", torch.uint8),
                        token_level_rewards=_cpu("rm_scores", torch.float32),
                        old_log_probs=_cpu("old_log_probs", torch.float32),
                        rollout_log_probs=_cpu("rollout_log_probs", torch.float32),
                        online_advantages=_cpu("advantages", torch.float32),
                        reward_components=np.asarray(component_rows, dtype=np.float32),
                        identity_json=np.asarray(
                            [json.dumps(row, sort_keys=True) for row in identity_rows]
                        ),
                    )
                os.replace(tmp, final)
                manifest = capture_path / "prompt_manifest.jsonl"
                with open(manifest, "a", encoding="utf-8") as handle:
                    for row in prompt_rows:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(
                    f"[attribution] captured step={step} rows={len(identity_rows)} path={final}",
                    flush=True,
                )

            metrics_path = os.environ.get("ECA_TRAIN_METRICS_JSONL", "").strip()
            if metrics_path:
                Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
                row = {"step": int(global_steps), **{k: float(v) for k, v in metrics.items() if np.isscalar(v)}}
                with open(metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
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
    suffix = ":forward_only" if forward_only else ""
    return f"patched:{log_tag}{suffix}"


def main() -> None:
    status_file = file_patch_task_runner()
    status_apply = apply()
    print(f"file={status_file} apply={status_apply}")


if __name__ == "__main__":
    main()
