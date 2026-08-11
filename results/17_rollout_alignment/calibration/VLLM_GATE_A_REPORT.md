# vLLM Route-Token Calibration — Gate A

Date: 2026-08-11

Verdict: **FAIL**

## Purpose

Test whether vLLM can replace the misaligned SGLang rollout path while preserving
the frozen HF behavior policy at the route root.

## Protocol

- Model: `outputs/rl/03_hf_evidence_step400`
- Prompts: the same frozen 20 canonical prompts from Path C
  - 11 `NoSearch`
  - 9 `NeedSearch`
- Raw probe: greedy first token with top-50 model logprobs
- Natural sampling: 16 rollouts per prompt, `temperature=0.9`, `top_p=0.95`
- Runtime: vLLM `0.10.1.dev1`, Transformers `4.54.1`, Torch `2.7.1+cu128`
- Tensor parallelism / dtype: `1` / `bfloat16`
- Route token roots: `<search>` = `27`, `<internal>` = `4159`

## Gate A result

| Metric | Required | Observed | Result |
|---|---:|---:|---|
| Median `|log P_vLLM(search) - log P_HF(search)|` | `<= 0.02` | `0.270759` | FAIL |
| P95 absolute delta | `<= 0.05` | `0.476559` | FAIL |
| `P(internal | NoSearch)` | `> 0.10` | `0.0` | FAIL |
| Mixed-action group rate | `> 0` | `0.0` | FAIL |

All 320 natural samples selected `<search>`: 176/176 for `NoSearch` and
144/144 for `NeedSearch`. Every sample had valid HF and vLLM search-token
logprobs; per-sample absolute deltas ranged from `0.119190` to `0.568814`.

## Interpretation

The failure is not explained by `temperature=0.9`: the raw greedy logprob probe
already shows a large backend mismatch before stochastic sampling. vLLM therefore
does not satisfy the rollout-policy contract for this checkpoint and must not be
used for formal GRPO.

## Decision

Stop vLLM calibration and continue the locked recovery path:

1. Complete the isolated official VeXact environment (`eca-verl-vexact`).
2. Run the same frozen-20 Gate A through minimal VeXact rollout.
3. If VeXact cannot reach minimal rollout within two effective working days,
   switch automatically to HFExact.

Evidence:

- `vllm_gateA/vllm_route_calibration_summary.json`
- `vllm_gateA/vllm_route_calibration.jsonl`
- Reproduction script: `scripts/audit_vllm_route_calibration.py`
