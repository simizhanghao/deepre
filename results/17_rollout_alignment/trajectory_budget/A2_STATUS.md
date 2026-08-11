# A2 VeXact AgentLoop Status

Date: 2026-08-11

Status: **PASS — `A2_AGENT_LOOP_SMOKE_PASS`**.

Attempt 1 stopped during veRL configuration validation, before model loading:

```text
train_batch_size (2) must be >= actor.ppo_mini_batch_size (4)
```

This is a smoke-launcher sizing error, not a VeXact or AgentLoop result. The
launcher now uses `ppo_mini_batch_size=2`, matching the locked two-question A2
smoke. Gate A1 remains PASS. A2 must be rerun before proceeding to A3.

The retry now reuses the exact two A1 prompts and additionally checks canonical
prompt hashes, Evidence@400 checkpoint identity, and a route-logprob collapse
sentinel before declaring A2 PASS.

Attempt 2 also stopped before model loading: rollout
`log_prob_micro_batch_size_per_gpu` was unset. The launcher now sets it to `1`.
Its new CPU-only path runs the real veRL/Hydra `validate_config`; the complete
A2 configuration passes (`A2_VERL_CONFIG_VALIDATION_PASS`). GPU retry remains
the only pending A2 action.

Attempt 3 reached 4-GPU VeOmni FSDP model construction and loaded Evidence@400,
then failed while constructing the cosine LR scheduler because `lr=0` causes
`min_lr / init_lr` division by zero. This is an A2 launcher-mode error, not a
rollout result. A2 is now a true validation-only job: VeOmni actor
`forward_only=True`, `trainer.val_only=True`, and validation sampling is locked
to `N=2`, `T=0.9`, `top_p=0.95`. No optimizer/scheduler or update is created.

Final retry completed the real veRL `EcaSearchAgentLoop → VeXact` path:

- 4/4 non-empty rollouts from the two exact A1 prompts
- checkpoint hashes and canonical prompt hashes match
- sampled routes: 1 internal, 3 search
- maximum sampled route probability: `0.73096` (no return to `0.997` collapse)

A2 is closed. Proceed to A3/Gate B.
