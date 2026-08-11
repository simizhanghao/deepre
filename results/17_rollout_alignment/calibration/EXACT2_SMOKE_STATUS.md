# VeXact Exact-Contract 2-Q Smoke

Date: 2026-08-11

## Frozen inputs

- Evidence@400
- one `NoSearch` + one `NeedSearch`
- exact canonical prompt IDs and hashes from `sample_ids_exact2.json`
- BF16, A100, `triton-invariant`, batch-invariant mode

## Phase 1 — Full-logit contract

Status: **PASS**

- VeXact captured complete first-token vocabulary logits.
- VeOmni `build_foundation_model` ran the matching batch-invariant Qwen2 forward.
- Requests matched: `2/2`
- Tokens matched: `2/2`
- Maximum absolute logit difference: `0.0`
- Maximum relative logit difference: `0.0`
- Standalone generated-token logprob difference: `0.0`

Because the complete vocabulary vector matches, both route roots (`search=27`,
`internal=4159`) match exactly on these prompts.

## Phase 2 — Trainer logprob path

Status: **PASS**

Verify the same captured VeXact output against VeOmni fused-LCE logprob
recomputation:

- Requests matched: `2/2`
- Tokens matched: `2/2`
- Maximum absolute logprob difference: `0.0`
- Maximum relative logprob difference: `0.0`

## Verdict

`EXACT_2Q_SMOKE_PASS`. Expansion to frozen-20 Gate A1 is authorized; AgentLoop,
trajectory-budget changes and training remain forbidden until Gate A1 passes.
