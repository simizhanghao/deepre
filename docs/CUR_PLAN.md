# Conditional Utility Router (CUR) Plan

Status: **CUR-0 complete; Layer-27 small MLP unlocked.**

## Decision

Root-token RL is closed after RP-0: class-conditional signs were locally
correct, but the shared LM update moved both boundary classes in the same
direction.  The active route is a separate router trained from causal utility,
while Evidence@400 and its AgentLoop remain frozen.

## CUR-0 causal contract

- Draw 128 fresh random HotpotQA train-pool questions, excluding the historical
  train-128 and frozen val-200. Boundary-v1 is historical only.
- For every question, reuse one byte-identical canonical prompt and intervene
  only after the Assistant boundary: `do(A0=search)` or `do(A0=internal)`.
- Search forces only the root `<search>` opening. The policy generates the
  query and subsequently follows the natural AgentLoop. Internal forces the
  root `<internal>` opening and disables tools for the complete episode.
- Use N=4 per arm initially. Retain policy failures; retry/exclude only genuine
  infrastructure failures.
- Persist answer F1, EM, number of searches, observation/response tokens,
  duplicate queries, finish/format and intervention-validity separately.
  Evidence F1, Format and lambda never enter the causal route label.

## Gates

1. **Forced-arm smoke:** 2 questions × 2 arms × 2 rollouts. Require exact
   canonical-prompt hashes across arms, correct intervention accounting and no
   internal-arm tool execution.
2. **CUR-0 capture:** 128 questions × 2 arms × 4 rollouts in detached tmux.
3. **Gate 0A:** estimate paired `delta_F1`, `delta_EM`, and expected search cost;
   verify a genuinely bidirectional utility distribution.
4. **Gate 0B:** compare frozen Evidence@400 root margin with `delta_F1` using
   Spearman and direction AUROC.
5. **Gate 0C:** pre-action last-prompt-token hidden states at fixed layers
   18/27/36; question-level 5-fold linear probes, reporting Spearman, high-
   confidence direction AUROC, MAE and RMSE. Layer 27 is primary.
6. Use a small MLP only if the preregistered linear probe is insufficient.
   Add uncertainty modeling only if margin and linear probes are both weak.

CUR-0 final: root margin Spearman `0.177`; primary Layer-27 linear Spearman
`0.318` with only `2.55%` RMSE gain over the constant mean after N=8 label
refinement. This is insufficient for freezing CUR-v0, so the small-MLP branch
is unlocked. Fixed sensitivity layers must not replace Layer 27 post hoc.

Deployment applies cost after prediction:

```text
search iff predicted_delta_F1 - lambda * predicted_search_cost > 0
```

Report separate F1-cost and EM-cost frontiers. CUR-1 expands to 1024 paired
questions with 256/512/1024 learning curves; 2048 is unlocked only by a
data-limited learning curve.

## Artifacts

```text
data/cur/cur0_fresh128/
results/22_cur/cur0_capture/{smoke,full}/
scripts/build_cur_forced_dataset.py
scripts/run_cur0_forced_capture.sh
scripts/audit_cur0_capture.py
src/rl/rewards_cur.py
```
