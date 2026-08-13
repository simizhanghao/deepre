# Phase 25 Step Preference Gate — Fresh Val3 Final Report

Date: 2026-08-13

Decision: `STEP_VAL3_FAIL`

Original Test: sealed and unread

## What was tested

S1 had established real local causal headroom: at byte-identical reasoning
checkpoints, delaying selected searches could reduce calls without reducing
answer quality.  Phase 25 therefore trained exactly one conservative
Search-Safety Gate and gave it one fresh integrated on-policy decision on
Val3.  No model/loss/threshold sweep was performed after observing Val3.

The Gate was trained on 1,022 S1 states from 640 open-Train questions:

- labels: 74 SEARCH and 948 CONTINUE;
- five question-grouped outer folds;
- outer-train-only L27 PCA64 and scalar standardization;
- strict OOF root B3 feature during OOF fitting;
- regret-weighted BCE with SEARCH class reweighting;
- one 64→64→32→1 GELU MLP, three seeds;
- threshold selected to capture at least 95% of positive Search regret.

The selected OOF threshold was `0.4683305`.  It captured `95.323%` of positive
regret, preserved OOF F1 (`+0.00059` versus all Search), but its paired call
ratio was already `0.94954`.  This was a conservative quality-first operating
point, not a promise of the required 25% deployment saving.

The deployment bundle froze Evidence@400, tokenizer/config, query-final-token
L27 PCA, scaler, three Gate models, threshold, and inherited root-B3 artifacts.
The inherited B3 scalar was precomputed for Val3 in fixed ordered batch32 before
any outcomes, avoiding BF16 batch-shape drift in online inference.

Fresh Val3 contains 128 previously unused HotpotQA distractor questions,
selected with seed `2026081203` after excluding 3,047 historical IDs.  Each arm
used one greedy rollout (`temperature=0`, `top_p=1`):

1. No Search: old Evidence agent forced to internal.
2. Old AlwaysSearch: old Evidence agent forced to root Search.
3. Step-AllSearch: bounded Step agent searches at every eligible checkpoint.
4. Frozen Step Gate: the identical Step agent uses the frozen safety Gate.

## Final results

| Arm | F1 | EM | Search calls | Token cost | Generation seconds | Finish |
|---|---:|---:|---:|---:|---:|---:|
| No Search | 0.17428 | 0.12500 | 0.0000 | 53.34 | 15.77 | 1.000 |
| Old AlwaysSearch | **0.56427** | **0.46875** | 1.0000 | 636.62 | 30.55 | 1.000 |
| Step-AllSearch | 0.36496 | 0.27344 | 2.0625 | 1200.25 | 64.34 | 1.000 |
| Frozen Step Gate | 0.36496 | 0.27344 | 2.0000 | 1182.77 | 63.52 | 1.000 |

All 512 scientific-arm trajectories completed; there were no missing final
answers in the reported outcomes.

### Scientific comparison: Gate versus Step-AllSearch

- F1 delta: `0.00000`, paired-bootstrap 95% CI `[0.0, 0.0]`.
- Calls delta: `-0.0625`, 95% CI `[-0.109375, -0.0234375]`.
- Call ratio: `0.969697`, only a `3.03%` reduction.
- Required call ratio: `<=0.75` with F1 delta `>=-0.02`.
- Decision: **FAIL** on retrieval reduction; quality constraint passes.

The Gate preserved answer quality exactly but did not save enough searches.

### Project comparison: Gate versus old AlwaysSearch

- F1 delta: `-0.19931`, 95% CI `[-0.28658, -0.11162]`.
- Token-cost delta: `+546.16`, 95% CI `[+502.48, +589.73]`.
- Required: strictly lower token cost with F1 delta `>=-0.02`.
- Decision: **FAIL** on both quality and cost.

The new Step scaffold is not a deployable replacement for the old Evidence
agent in its current form.

## What the Gate actually changed

Across Val3, the learned Gate received 267 eligible online decisions.  Only 11
probabilities fell below the frozen CONTINUE threshold.  The complete state
machine recorded 256 searches and 128 continues, but most continues were
structural/ineligible cases rather than learned savings.  Downstream search
compensation converted the 11 learned skips into only 8 net saved calls:

- Step-AllSearch total calls: 264;
- Step Gate total calls: 256;
- seven questions changed call count (six by −1, one by −2);
- all seven retained identical F1;
- 127/128 final answer strings were identical between the Step arms.

The online probability distribution was strongly conservative: median
`P(SearchRequired)=0.6637`, with only 11/267 eligible states below `0.4683`.

## Interpretation

This is not an execution or parity failure.  Every arm exited successfully,
finish rate was 1.0, the online Gate used the frozen artifacts, and the original
Test remained sealed.  The result separates two limitations.

First, the safety frontier did not provide enough deployable coverage.  OOF
already predicted a call ratio of 0.9495 at 95% regret capture; Fresh Val3
realized 0.9697.  The small degradation may reflect ordinary on-policy shift,
but the dominant shortfall was visible before Val3: the conservative threshold
could not simultaneously protect 95% of regret and approach 25% savings.

Second, the Step scaffold itself is substantially worse than the old agent.
Step-AllSearch loses 0.1993 F1 and uses about 1.89× the token cost of old
AlwaysSearch.  Adaptive routing cannot make the project pass merely by shaving
several searches from a scaffold whose quality/cost base point is already
inferior.

The scientifically valid conclusion is therefore narrow:

> Early-search causal headroom exists locally, and the frozen Gate can identify
> a small set of safe skips, but this representation/calibration and bounded
> Step scaffold do not convert that headroom into the preregistered on-policy
> efficiency gain or a better deployable system.

No Val3 threshold retuning is allowed.  Under the frozen plan, Phase 25 closes
and the original Test remains locked.

## Reproducibility pointers

- Gate training summary: `results/25_step_adaptive/step_gate/models/summary.json`
- Deployment freeze: `results/25_step_adaptive/step_gate/models/deployment_freeze.json`
- Val3 split freeze: `data/cur/step_val3_fresh128/manifest.json`
- Per-question outcomes: `results/25_step_adaptive/val3/analysis/per_question.jsonl`
- Final machine-readable decision: `results/25_step_adaptive/val3/analysis/summary.json`
