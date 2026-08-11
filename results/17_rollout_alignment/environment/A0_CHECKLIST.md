# A0 — Environment lock (`eca-verl-vexact`)

## Layout

```text
/data1/hcc/eca-verl-vexact/     # VeXact clone + uv env (NOT eca-verl)
results/17_rollout_alignment/
├── environment/                # this dir: SHA / pin dumps
├── calibration/
├── parity_32x4/
└── trajectory_budget/
```

## Rules

- Freeze docker/container **`eca-verl`** — do not pip upgrade it.
- Install only via VeXact official: `uv sync` with its `pyproject.toml` pins.
- Record VeXact `HEAD` + resolved pins here before any calibration.
- A100: later examples need `INFER_FA_IMPL=triton-invariant` (not FA3/FA4 defaults).
- Launch VeXact commands with `env -u LD_LIBRARY_PATH`; the host CUDA path
  otherwise overrides the locked CUDA 12.9 wheel libraries.

## Current status (2026-08-11)

- [x] Official `uv sync --frozen --extra gpu --extra verl --extra veomni`
- [x] Imports: torch `2.10.0+cu129`, Transformers `5.9.0`, VeXact, veRL, VeOmni
- [x] GPU visibility: one A100-SXM4-80GB, compute capability `8.0`
- [x] Evidence@400 minimal dense VeXact rollout smoke (`outputs/smoke_vexact_dense_20260811`, PASS)
- [x] VeOmni actor-model ↔ VeXact 2-Q exact-contract smoke (full logits + fused LCE, zero difference)

## Day targets

| Day | Done when |
|-----|-----------|
| 1 | clone + `uv sync` OK + minimal dense import/rollout smoke |
| 2 | Evidence@400 exact-20: token_ids + logprobs (Gate A prep) |

Fallback after 2 effective working days → `VEXACT_INTEGRATION_HOLD → HFEXACT_FALLBACK`.
