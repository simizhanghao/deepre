# Root-Pivot Decision-Aligned Credit Plan

Status: **RP-0 complete — `RP0_FAIL`; formal @10 locked**.

Execution status (2026-08-12): implementation/unit gates and balanced-data gate
PASS; all nine one-step branches completed. The mechanism gate failed because
balanced route-only moved both frozen classes toward internal. Cross-job token
trajectories were also non-identical, so the gradient cosine is intentionally
null. Formal @10 remains locked. See [ROOT_PIVOT_RP0_REPORT.md](ROOT_PIVOT_RP0_REPORT.md).

## Causal claim

Exact rollout experiments closed the trajectory-level optimizer sweep:
standard GRPO moved both classes toward search; RF++ baseline and GRPO-no-std
moved both toward internal. GRPO-no-std retained a frozen class gap of `.520`
versus Evidence@400 `.608`, suggesting conditional information remains but is
overwhelmed by a global trajectory-induced intercept shift.

Root-Pivot tests one intervention only: isolate task-quality credit from the
pivotal root routing decision.

## Frozen v0 objective

`L = L_task(non-root policy tokens) + beta * L_route(root only)`.

- `R_task = R_answer + .5 R_evidence + .1 R_format` for both classes.
- Search cost is absent from trajectory task reward.
- The first response/root token receives zero task advantage.
- `L_route = softplus(-y * (logit_search - logit_internal))`.
- `y=+1` for NeedSearch and `-1` for NoSearch.
- Undetermined is forbidden in v0.
- No classifier head; token IDs and score semantics are identical to the
  frozen route-margin evaluator (`search=27`, `internal=4159`).
- Actor fused-logprob kernels are disabled because exact unchosen-action logits
  must be materialized; rollout remains VeXact exact.

The dataset is a deterministic, interleaved 8 NeedSearch + 8 NoSearch batch,
with four rollouts per question. Optimizer, Evidence@400 initialization,
sampling and trajectory budgets otherwise remain frozen.

## RP-0 mechanism smoke

From the same initialization and balanced batch, run non-resumable one-step
counterfactuals:

1. task-only;
2. route-only with beta=1;
3. joint with beta fixed once as `||g_task|| / (||g_route|| + eps)`.

The gradient cosine is recovered from the three pre-clip norms:

`cos = (||g_joint||^2 - ||g_task||^2 - beta^2||g_route||^2) /
       (2 beta ||g_task|| ||g_route||)`.

Repeat attribution for all, NeedSearch and NoSearch subsets. These runs retain
only metrics and route-margin summaries; temporary optimizer/model states are
deleted after evaluation.

RP-0 hard gate:

- route-only: `delta M_NS < 0` and `delta M_Need > 0`;
- joint: `delta M_NS < 0` and `delta M_Need >= 0`;
- finite positive gradient norms and cosine in `[-1,1]` within tolerance;
- finish `>=.95`, response clip `<.05`, and no label/mask/score assertion.

Failure stops formal training. Route-only failure means implementation,
labels, or scoring must be fixed. Route-only PASS plus joint failure identifies
shared-parameter gradient conflict and activates a root-only Route Adapter.

## Formal gates after RP-0

Only RP-0 PASS unlocks Root-Pivot @10. Save steps 5 and 10 and probe frozen
margins at 1/3/5/7/10. Step10 requires:

- `M_NS < .864`;
- `M_Need >= 1.272`;
- `D=M_Need-M_NS >= .608`;
- Exact PASS, finish `>=.95`, clip `<.05`, mixed support `>0`.

Strong PASS is `M_NS<0` with `M_Need>1.272`.

Step10 PASS continues unchanged to step25, where frozen 32x4 requires Need SR
`>=.85`, NoSearch SR `<=.70`, delta `>=.20`; strong targets are `.85/.50/.35`.
Answer and Evidence may not degrade by more than about two absolute points from
Evidence@400. Step25 PASS continues to step50 and stops. A successful candidate
then triggers enabled/disabled Boundary-v2 refresh; never continue directly to
step100.
