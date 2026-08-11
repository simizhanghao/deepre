# Exact-Rollout ECA Closure Report

Date: 2026-08-11

Model: `outputs/rl/03_hf_evidence_step400`

Rollout backend: VeXact registered in veRL; VeOmni/FSDP actor reference

Status: **A1, A2, Gate B and A4 PASS; Boundary@50 is next**

## Why this line was necessary

The earlier SGLang and vLLM audits both changed the first route-token
distribution relative to the authoritative training-side model. The observed
near-deterministic search routing therefore could not be attributed to reward,
temperature or the learned Evidence@400 checkpoint. VeXact was introduced to
make rollout inference use the same model implementation and numerical contract
as the VeOmni/FSDP actor.

## A1 — exact backend calibration

The frozen-20 calibration established backend identity before AgentLoop work:

- full logits: 20/20 requests and tokens matched, maximum absolute delta `0`
- fused-LCE logprobs: 20/20 matched, maximum absolute delta `0`
- natural sampling: `P(internal | NoSearch)=0.29545`
- mixed-action group rate: `1.0`

Verdict: `EXACT_ROLLOUT_GATE_A_PASS`. This closed the ordinary-backend diagnosis
and made VeXact the rollout backend for the remaining ECA recovery line.

## A2 — real AgentLoop integration

A2 moved from a standalone inferencer test into the registered veRL path. Three
launcher-only failures were resolved before the successful run: mini-batch
sizing, rollout logprob micro-batch configuration, and a zero-LR scheduler
division. The final smoke was converted to true forward-only validation, so it
created no optimizer and made no model update.

Final A2 result:

- 2 exact frozen prompts × 2 rollouts = 4/4 non-empty generations
- canonical prompt and Evidence@400 checkpoint sentinels matched
- routes: `1 internal / 3 search`
- maximum sampled route probability: `0.73096`, far from the historical
  `p(search)≈0.997` collapse

Verdict: `A2_AGENT_LOOP_SMOKE_PASS`.

## A3 / Gate B — multi-turn trajectory contract

The production AgentLoop was repaired around an explicit token budget:

- total response trajectory: 2048 tokens
- each assistant turn: at most 256 tokens
- each observation: at most 384 tokens
- final-answer reserve: at least 256 tokens
- no shared `stop_token_ids=[29]`; complete `</search>`, `</internal>` and
  `</answer>` closures are recognized

Attempt 1 completed the runtime path but only 13/16 trajectories finished. The
three failures were long answer turns reaching the 256-token cap. It also showed
that Qwen can tokenize an identical closing tag differently depending on its
left context, so matching only the standalone tag tokenization undercounted
valid closures.

The repair retained every budget. Closing tags are checked on decoded bounded
prefixes as well as exact token sequences, and a capped unclosed fragment gets
one further bounded continuation opportunity. The evaluator distinguishes such
continued fragments from unresolved unclosed turns.

Gate B retry result (8 frozen questions × 2 rollouts):

| Metric | Result | Gate |
|---|---:|---:|
| finish rate | `1.0000` | `≥0.95` |
| trajectory clip ratio | `0` | `<0.05` |
| final-answer missing rate | `0` | `0` |
| final-answer reserve violations | `0` | `0` |
| maximum assistant turn | `256` | `≤256` |
| maximum observation turn | `384` | `≤384` |
| continued capped turns | `2` | diagnostic |
| unresolved unclosed turns | `0` | `0` |
| `P(internal | NoSearch)` | `0.1667` | support present |
| mixed-action group rate | `0.125` | diagnostic |

The first retry summary initially printed FAIL because the evaluator treated the
two explicitly continued fragments as unresolved. Re-evaluating the saved
artifacts with the corrected contract produced `A3_GATE_B_PASS`; no GPU rerun
was needed.

## A4 — frozen exact AgentLoop parity, 32×4

A4 froze Evidence@400, the historical 32 questions, Boundary labels and
sampling (`N=4`, `T=0.9`, `top_p=0.95`). It ran 128 real multi-turn VeXact
AgentLoop trajectories without training.

| Metric | Result | Gate |
|---|---:|---:|
| finish rate | `1.0000` | `≥0.95` |
| trajectory clip ratio | `0` | `<0.05` |
| final-answer missing rate | `0` | `0` |
| final-answer reserve violations | `0` | `0` |
| maximum assistant turn | `256` | `≤256` |
| maximum observation turn | `384` | `≤384` |
| continued capped turns | `20` | diagnostic |
| unresolved unclosed turns | `0` | `0` |
| routes | `29 internal / 99 search` | diagnostic |
| `P(internal | NoSearch)` | `0.31818` | `>0.10` |
| mixed-action group rate | `0.59375` | `>0` |

Verdict: `A4_EXACT_PARITY_PASS`.

## Conclusions and limits

The exact backend now passes three distinct claims: numerical equality to the
actor (A1), correct real AgentLoop integration (A2), and stable multi-turn
budget/route support at 8×2 and 32×4 scales (Gate B/A4). The old apparent search
collapse was therefore primarily a rollout-backend mismatch, not proof that the
Evidence@400 policy lacked an internal branch.

A4 does **not** claim that boundary routing is already optimal or that GRPO will
improve it. It establishes the prerequisite that both root actions remain
sampleable under the actual training rollout path. Reward/table retuning and
alternative optimizers remain frozen until the causal Boundary@50 run.

## Next gate

Run Boundary@50 from Evidence@400 with VeXact as the only causal backend change.
At steps 0/10/25/50 retain the frozen 2-question VeOmni↔VeXact alignment
sentinel. The registered evaluation gate remains:

- NeedSearch search rate `≥0.85`
- NoSearch search rate `≤0.70` (prefer `≤0.50`)
- boundary separation `Δ_boundary ≥0.20` (prefer `≥0.30`)
- Answer/Evidence degradation within the documented tolerance
