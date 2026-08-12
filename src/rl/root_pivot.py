"""Root-Pivot v0 loss: isolate root routing credit from trajectory task credit."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributed.tensor import DTensor

SEARCH_TOKEN_ID = 27
INTERNAL_TOKEN_ID = 4159
NEED_SEARCH = 1
NO_SEARCH = -1


def enabled() -> bool:
    return os.environ.get("ECA_ROOT_PIVOT", "0").strip().lower() in {"1", "true", "yes"}


def mode() -> str:
    value = os.environ.get("ECA_ROOT_PIVOT_MODE", "joint").strip().lower()
    if value not in {"task_only", "route_only", "joint"}:
        raise ValueError(f"invalid ECA_ROOT_PIVOT_MODE={value!r}")
    return value


def beta() -> float:
    value = float(os.environ.get("ECA_ROOT_PIVOT_BETA", "1"))
    if not (value > 0 and torch.isfinite(torch.tensor(value))):
        raise ValueError(f"invalid ECA_ROOT_PIVOT_BETA={value!r}")
    return value


def labels_from_boundaries(boundaries: list[str]) -> torch.Tensor:
    mapping = {"NeedSearch": NEED_SEARCH, "NoSearch": NO_SEARCH}
    try:
        return torch.tensor([mapping[x] for x in boundaries], dtype=torch.int64)
    except KeyError as exc:
        raise ValueError(f"Root-Pivot v0 forbids boundary={exc.args[0]!r}") from exc


def labels_from_data(data: TensorDict) -> torch.Tensor:
    """Derive labels from the row-aligned metadata that veRL already dispatches."""
    extras = list(data["extra_fields"])
    boundaries = [
        str((row.get("reward_extra_info") or {}).get("boundary", ""))
        for row in extras
    ]
    return labels_from_boundaries(boundaries)


def _zero_first_jagged(value: torch.Tensor) -> torch.Tensor:
    """Clone a jagged tensor and zero the first response position per row."""
    if not value.is_nested:
        out = value.clone()
        out[:, 0] = 0
        return out
    values = value.values().clone()
    offsets = value.offsets()
    lengths = offsets[1:] - offsets[:-1]
    if bool((lengths <= 0).any()):
        raise ValueError("Root-Pivot received an empty response row")
    values[offsets[:-1]] = 0
    return torch.nested.nested_tensor_from_jagged(values, offsets)


def task_only_data(data: TensorDict) -> TensorDict:
    """Remove trajectory advantage from the root token without changing other tokens."""
    out = data.clone()
    out["advantages"] = _zero_first_jagged(out["advantages"])
    return out


def route_logistic_loss(root_route_logits: torch.Tensor, route_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise logistic loss on the exact frozen-gate margin semantics."""
    if root_route_logits.ndim != 2 or root_route_logits.shape[-1] != 2:
        raise ValueError(f"expected [batch,2] root logits, got {tuple(root_route_logits.shape)}")
    labels = route_labels.to(device=root_route_logits.device, dtype=root_route_logits.dtype)
    if not bool(((labels == NEED_SEARCH) | (labels == NO_SEARCH)).all()):
        raise ValueError("route labels must be +1 NeedSearch or -1 NoSearch")
    margin = root_route_logits[:, 0] - root_route_logits[:, 1]
    return F.softplus(-labels * margin).mean(), margin


def root_pivot_ppo_loss(config, model_output, data: TensorDict, dp_group=None):
    """veRL actor loss adapter; imported in workers through model.external_lib."""
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.losses import ppo_loss as current_ppo_loss

    base = getattr(current_ppo_loss, "_root_pivot_original", current_ppo_loss)
    task_loss, metrics = base(config=config, model_output=model_output, data=task_only_data(data), dp_group=dp_group)
    route_loss, margins = route_logistic_loss(model_output["root_route_logits"], labels_from_data(data))
    rp_mode = mode()
    route_beta = beta()
    if rp_mode == "task_only":
        loss = task_loss
    elif rp_mode == "route_only":
        loss = route_beta * route_loss
    else:
        loss = task_loss + route_beta * route_loss
    metrics.update(
        {
            "root_pivot/task_loss": Metric(AggregationType.MEAN, task_loss.detach()),
            "root_pivot/route_loss": Metric(AggregationType.MEAN, route_loss.detach()),
            "root_pivot/margin_mean": Metric(AggregationType.MEAN, margins.detach().mean()),
            "root_pivot/beta": Metric(AggregationType.MEAN, route_beta),
        }
    )
    return loss, metrics


def install_worker_patch() -> str:
    """Expose exact two-token root logits and replace only the actor PPO loss."""
    if not enabled():
        return "disabled"
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    import verl.workers.engine_workers as engine_workers
    import verl.workers.utils.losses as losses

    cls = FSDPEngineWithLMHead
    if not getattr(cls.prepare_model_outputs, "_root_pivot_patched", False):
        original = cls.prepare_model_outputs

        def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
            result = original(self, output, output_args, micro_batch, logits_processor_func)
            from verl.utils import tensordict_utils as tu

            use_fused = bool(tu.get_non_tensor_data(micro_batch, "use_fused_kernels", default=False))
            if use_fused:
                raise RuntimeError("Root-Pivot requires actor use_fused_kernels=False for exact pair logits")
            raw = output.logits.squeeze(0)
            if isinstance(raw, DTensor):
                raw = raw.full_tensor()
            # In this veRL v1 AgentLoop, loss_mask contains response positions
            # only; it is not aligned to packed prompt+response input_ids.
            # The logit predicting response token 0 is therefore the final
            # prompt position in each packed sequence.
            input_offsets = micro_batch["input_ids"].offsets()
            prompt_lens = micro_batch["prompts"].offsets().diff()
            response_lens = micro_batch["responses"].offsets().diff()
            sequence_lens = input_offsets.diff()
            if not bool((prompt_lens > 0).all() and (response_lens > 0).all()):
                raise RuntimeError("Root-Pivot requires non-empty prompt and response rows")
            if not bool((prompt_lens + response_lens == sequence_lens).all()):
                raise RuntimeError("Root-Pivot prompt/response lengths do not match packed input_ids")
            root_predict_positions = (input_offsets[:-1] + prompt_lens - 1).tolist()
            pair = raw[root_predict_positions][:, [SEARCH_TOKEN_ID, INTERNAL_TOKEN_ID]]
            result["root_route_logits"] = pair
            return result

        prepare_model_outputs._root_pivot_patched = True  # type: ignore[attr-defined]
        cls.prepare_model_outputs = prepare_model_outputs

    if not getattr(losses.ppo_loss, "_root_pivot_loss", False):
        original_loss = losses.ppo_loss
        root_pivot_ppo_loss._root_pivot_original = original_loss  # type: ignore[attr-defined]
        root_pivot_ppo_loss._root_pivot_loss = True  # type: ignore[attr-defined]
        losses.ppo_loss = root_pivot_ppo_loss
        engine_workers.ppo_loss = root_pivot_ppo_loss
    print(
        f"[root-pivot] worker patch active mode={mode()} beta={beta()} "
        f"tokens=({SEARCH_TOKEN_ID},{INTERNAL_TOKEN_ID})",
        flush=True,
    )
    return "patched"


# model.external_lib imports this module inside each actor worker.
if enabled():
    import vexact.integrations.verl.fsdp_enable_invariant  # noqa: F401,E402

    install_worker_patch()
