# VeXact Frozen-20 Gate A1

Date: 2026-08-11

## Verdict

`EXACT_ROLLOUT_GATE_A_PASS`

- Evidence@400, frozen 20 questions, BF16, `triton-invariant`
- VeOmni batch-invariant actor forward ↔ VeXact full logits: 20/20,
  maximum absolute difference `0.0`
- VeOmni fused-LCE logprobs ↔ VeXact rollout: 20/20,
  maximum absolute difference `0.0`
- Natural sampling (`T=0.9`, `top_p=0.95`, 16 rollouts/question):
  `P(internal | NoSearch)=0.29545`, mixed-action group rate `1.0`, other count `0`

The old HF comparison remains diagnostic only. Its search-token median/P95
absolute delta from VeXact is `0.000657 / 0.07479`; it does not override the
zero-difference authoritative VeOmni↔VeXact contract.

## Decision

Gate A1 is closed. Proceed to A2 minimal `EcaSearchAgentLoop` integration, then
A3 trajectory-budget repair and Gate B. Formal training remains blocked until
Gate B passes.

Machine-readable result: `vexact_gateA1/vexact_gate_a1_summary.json`.
