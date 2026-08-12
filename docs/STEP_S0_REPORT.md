# Phase 25 S0 Contract and Replay Report

Date: 2026-08-12  
Decision: **`STEP_S0_REPLAY_PASS`**  
Data boundary: fixed CUR-1 Train only; Val3 and original Test were not read.

## What was implemented

- New `eca_step_adaptive_agent` registry; the historical agent loop is
  unchanged.
- Strict `<think>...Search query: ...</think>` parser and bounded
  `CONTINUE | SEARCH | ANSWER` state machine.
- Four-checkpoint / three-search system limits, 2048 response budget, 384
  observation cap, 256 answer reserve, and 128-token step cap.
- Separate controlled reasoning and candidate-query proposal spans. Raw model
  tokens are retained for audit; normalized controller tokens and structural
  tags are masked, so S0/S1 do not silently train the LM policy.

## Train32 contract gate

The first complete Train32 execution returned `rc=0` but correctly failed the
scientific gate on 2/32 rows. Both contained a valid query followed by the
legacy suffix `</search>`. The normalizer handled `<search>` but not its closing
form. That failed artifact is preserved as `train32_attempt1/summary.json`.

After adding every legacy opening and closing tag as a structural boundary,
the identical fixed Train32 rerun produced:

| Metric | Result |
|---|---:|
| rows | 32 |
| checkpoints | 112 |
| parser / close / query / finish rate | 1.0 / 1.0 / 1.0 / 1.0 |
| CONTINUE reaches next checkpoint | 1.0 |
| searches | 48 |
| response clipping | 0 |
| duplicate search execution | 0 |
| tool / answer-reserve violations | 0 / 0 |

Decision: **`STEP_S0_CONTRACT_PASS`**.

## Deterministic Train8 replay

The fixed first eight Train questions used greedy decoding and the frozen
`CONTINUE -> ANSWER` intervention. Replay A and B each passed 8 rows and 16
checkpoints with all contract rates 1.0 and zero violations.

Comparison was keyed by `sample_id` and included canonical/step prompt hashes,
complete response token IDs and masks, checkpoint and raw proposal token IDs,
actions, candidate queries, answers, finish state, and search count.

| Metric | Result |
|---|---:|
| rows A / B | 8 / 8 |
| sample-ID sets equal | true |
| exact complete-trajectory matches | 8 / 8 |
| exact trajectory match rate | 1.0 |

Decision: **`STEP_S0_REPLAY_PASS`**.

## Interpretation and next gate

S0 establishes that the step state, external intervention, completion, and
replay substrate are executable and deterministic. It does not establish that
candidate queries are semantically useful or that adaptive retrieval improves
quality-cost performance. Those claims remain locked behind S1 Train640
matched `SEARCH_NOW`/`CONTINUE_NOW` counterfactual acquisition, its query
availability gate, router fitting, and fresh Val3 evaluation.
