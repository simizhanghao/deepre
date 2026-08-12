# CUR-0 Causal Diagnostic Report

Status: **CUR-0 complete. N=8 refinement confirms modest linear signal; small MLP unlocked.**

## Capture integrity

The frozen Evidence@400 policy was evaluated under exact paired root-action
interventions on 128 fresh questions with four rollouts per arm.

| Check | Result |
|---|---:|
| Rows | 1024/1024 |
| Canonical prompt identity | PASS |
| Forced-action validity | 100% |
| Internal-arm tool violations | 0 |
| Finish rate | 99.71% |
| Actor update | disabled |

The final weakref/DataLoader warning occurred during interpreter teardown after
16/16 steps, capture, and audit. The main process returned zero.

## Answer-extraction correction

The initial CUR reward inherited an unsafe non-overlapping regex: an unmatched
literal `<answer>` in the search-arm user suffix could consume the model's real
closing tag. The immutable raw capture remains preserved. The true final answer
opening was still present inside `pred`, so 509 rows were deterministically
re-extracted; F1 changed in 422 rows. `rewards_cur.py` now selects the final
opening tag, preventing recurrence. All causal results below use the repaired
artifact and record its raw source.

## Gate 0A — causal utility

| Metric | do(search) | do(internal) | Delta |
|---|---:|---:|---:|
| F1 | 0.5957 | 0.2577 | +0.3380 |
| EM | 0.5020 | 0.1895 | +0.3125 |
| Finish | 0.9941 | 1.0000 | -0.0059 |
| Searches | 1.0 | 0.0 | +1.0 |

Direction counts: 77 search-positive, 20 internal-negative, 31 exact-zero.
This passes the bidirectional-utility gate. Bootstrap CIs yield 49 confident
search, 3 confident internal, and 76 borderline questions. The 76 borderline
questions are preregistered for four additional rollouts per arm (total N=8).

## Gate 0B — frozen root margin

Margin is `logP(<search>) - logP(<internal>)` at the exact canonical root.

| Metric | Result |
|---|---:|
| Spearman(margin, Delta F1) | 0.1773 |
| Direction AUROC, nonzero | 0.5643 |
| Direction AUROC, high confidence | 0.8605 |
| Mean margin | +1.4004 |
| Positive-margin rate | 128/128 |

The high-confidence AUROC is based on only three confident internal examples
and is not by itself decisive. Overall ranking is weak and every margin favors
search, reproducing the global-intercept failure. Frozen root margin is not a
sufficient router.

## Gate 0C — pre-action hidden-state linear probes

Question-level shuffled 5-fold out-of-fold Ridge, with scaling and alpha
selection restricted to each training fold. Layer 27 is primary; 18/36 are
sensitivity only.

| Layer | Spearman | HC AUROC | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 18 | 0.3118 | 0.8231 | 0.3488 | 0.4201 |
| **27** | **0.3126** | **0.7211** | **0.3504** | **0.4231** |
| 36 | 0.2809 | 0.8163 | 0.3580 | 0.4302 |

The constant-mean RMSE is 0.4350, so primary-layer RMSE improves by only 2.74%.
All folds select the strongest candidate regularization (alpha=1000). This is
real but modest representation signal, not yet a strong linear-router result.
Search-count cost is constant at one under this intervention and needs no model.

## Current decision

The registered N=8 supplement completed successfully:

| Check | Result |
|---|---:|
| Supplemental questions | 76 |
| Supplemental rows | 608/608 |
| Supplemental action validity | 100% |
| Supplemental internal tool violations | 0 |
| Supplemental finish rate | 99.67% |
| Merged rows | 1632 |
| Questions at N=4 / N=8 per arm | 52 / 76 |

### Final Gate 0A after N=8

| Metric | do(search) | do(internal) | Delta |
|---|---:|---:|---:|
| F1 | 0.5916 | 0.2499 | +0.3418 |
| EM | 0.5010 | 0.1807 | +0.3203 |
| Finish | 0.9951 | 1.0000 | -0.0049 |

Direction counts stabilize at 80 search-positive, 21 internal-negative and 27
exact-zero. Bootstrap CIs identify 67 confident search, 6 confident internal
and 55 still-borderline questions. The remaining borderline cases are mostly
small/zero effects; CUR-0 does not recursively allocate more samples to them.

### Final Gate 0B after N=8

| Metric | Result |
|---|---:|
| Spearman(margin, Delta F1) | 0.1766 |
| Direction AUROC, nonzero | 0.5378 |
| Direction AUROC, high confidence | 0.6779 |
| Positive-margin rate | 128/128 |

N=8 removes the misleading high-confidence margin AUROC seen with only three
internal examples. The frozen LM root margin is effectively a global search
intercept and is rejected as the router.

### Final Gate 0C after N=8

| Layer | Spearman | HC AUROC | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 18 (sensitivity) | 0.3390 | 0.8085 | 0.3442 | 0.4218 |
| **27 (primary)** | **0.3177** | **0.7139** | **0.3568** | **0.4310** |
| 36 (sensitivity) | 0.2870 | 0.6915 | 0.3681 | 0.4414 |

The constant-mean RMSE is 0.4423, so the preregistered Layer-27 probe improves
RMSE by only 2.55%. All five folds again select the maximum Ridge alpha=1000.
Label refinement therefore did not turn the primary linear probe into a strong
utility model. Layer 18 cannot be selected post hoc despite its better
sensitivity metrics.

**Final CUR-0 decision:** reject root-margin-only routing; do not freeze the
linear probe as CUR-v0; do not collect more CUR-0 rollouts. The preregistered
small-MLP branch is now unlocked, using Layer 27 only and the same question-level
cross-validation splits. Uncertainty/self-knowledge features remain locked
until the small MLP is evaluated.
