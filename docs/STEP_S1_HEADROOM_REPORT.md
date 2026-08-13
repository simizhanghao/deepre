# Phase 25 S1 Step Counterfactual Headroom Report

Date: 2026-08-13  
Decision: **`STEP_ADAPTIVE_HEADROOM_PASS`**  
Boundary: CUR-1 Train640 only; Val3 and original Test were not read.

## Acquisition contract

The deterministic base pass completed all 640 questions and 2560 checkpoints.
Parser, closure, finish, and CONTINUE-to-next-checkpoint rates were 1.0; there
were no clipping, tool, reserve, or duplicate-execution violations.

Eligible states require a valid non-`NONE`, nonduplicate candidate query and
legal SEARCH/CONTINUE actions. A singleton is selected once; otherwise the
earliest and latest eligible checkpoints are selected, capped at two per
question. This froze 1022 states: 382 questions contributed two and 258
contributed one. Every state was replayed as `SEARCH_NOW` and `CONTINUE_NOW`,
then both arms returned to the same deterministic completion policy with
future Search still available.

The paired acquisition completed 2044 scientific rows plus four batch-padding
rows. Prefix/checkpoint/action parity was exact for all 1022 scientific pairs.

## Primary causal results

| Metric | Result |
|---|---:|
| selected states | 1022 |
| prefix + intended action exact rate | 1.0000 |
| Search helpful (`F1_S > F1_C`) | 54 / 1022 = 5.28% |
| Continue better (`F1_C > F1_S`) | 63 / 1022 = 6.16% |
| exact F1 ties | 905 / 1022 = 88.55% |
| Continue safe (`F1_C >= F1_S`) | 94.72% |
| cost-saving Continue within 0.02 F1 | **61.94%** |
| mean F1, SearchNow / ContinueNow | 0.40907 / 0.40953 |
| mean calls, SearchNow / ContinueNow | 2.3464 / 1.7006 |
| mean `F1_S - F1_C` | -0.00046 |
| mean `Calls_S - Calls_C` | +0.6458 |
| mean token-proxy delta, S - C | +185.20 |

The distribution is highly sparse but not one-sided: Search has a strict
quality advantage on 54 states, while Continue has a strict advantage on 63.
The remaining 905 are quality ties. Therefore the useful problem is primarily
safe cost avoidance, with a smaller set of consequential Search states that a
gate must protect.

## Local Oracle frontier

Starting from SEARCH_NOW at every selected state, the local Oracle substitutes
the cheapest-quality-loss CONTINUE decisions until calls fall by at least 25%:

| Metric | Fixed Search | Oracle at 25% reduction |
|---|---:|---:|
| aggregate calls | 2398 | 1798 |
| retrieval reduction | — | **25.02%** |
| mean F1 | 0.40907 | **0.43799** |
| frozen quality floor | — | 0.38907 |

The Oracle clears the quality floor by 0.0489 and improves over fixed Search by
0.0289 F1 while removing 600 calls. Together with a 61.94% cost-saving
Continue rate, this passes both preregistered headroom conditions.

## Query evidence

- Overall base checkpoint query-field validity was 99.65%.
- The stricter valid, nonduplicate eligibility rate across all base
  checkpoints was 63.75%; repetition, not missing syntax, explains the gap.
- SearchNow retrieved at least one gold supporting title on 82.97% of selected
  states; mean supporting-title Recall@5 was 58.66%.

Candidate retrieval is therefore sufficiently grounded to proceed. Some
surface queries remain poor (including numeric fragments), but the aggregate
retrieval evidence does not trigger the query-proposal-SFT branch.

## Decision and next lock

S1 supplies the missing scientific evidence: the marginal value of searching
now is heterogeneous, the local Oracle has more than enough quality-cost
headroom, and query retrieval is usable. The only allowed next method is the
frozen Counterfactual Step Preference Gate:

1. quality difference beyond 0.02 selects the higher-F1 arm;
2. within 0.02, select the lower-call arm;
3. fit one weighted-BCE gate using the frozen step representation/features;
4. no model/loss/feature sweep;
5. only a frozen gate bundle may unlock fresh integrated Val3.

Val3 and the original Test remain sealed.
