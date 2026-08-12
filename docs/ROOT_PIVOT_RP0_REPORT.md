# Root-Pivot RP-0 — Mechanism Gate Report

Date: 2026-08-12
Verdict: **`RP0_FAIL`; formal Root-Pivot @10 remains locked.**

## Scope and protocol

RP-0 tested whether root-routing credit can be isolated from trajectory task
credit at the frozen Evidence@400 initialization. The frozen objective was:

`L = L_task(non-root response tokens) + beta * L_route(root token only)`.

- `L_task` used answer + `.5 × evidence + .1 × format`; search cost was zero.
- The first response token received zero task advantage.
- `L_route = softplus(-y × (logit_search-logit_internal))`, with NeedSearch
  `y=+1` and NoSearch `y=-1`.
- The batch was deterministic and balanced: 8 NeedSearch + 8 NoSearch,
  four VeXact rollouts per prompt (64 trajectories).
- Three non-resumable one-step branches were run from Evidence@400:
  task-only, route-only and joint. The same matrix was repeated on Need-only
  and No-only subsets, for nine completed branches in total.
- Every temporary checkpoint was evaluated with the frozen 20-question route
  probe and then deleted.

The final implementation smoke exited `0`. The full matrix exited `1` only
because the pre-registered scientific gate failed, not because a branch
crashed.

## Main balanced-batch result

Route margin is `log P(<search>)-log P(<internal>)`.

| Branch | grad norm | Need margin | Δ Need | NoSearch margin | Δ NoSearch |
|---|---:|---:|---:|---:|---:|
| Evidence@400 | — | `1.4722` | — | `.8636` | — |
| task-only | `.11643` | `1.4028` | `-.0694` | `.7614` | `-.1023` |
| route-only | `1907.955` | `1.1944` | `-.2778` | `.5909` | `-.2727` |
| joint | `.15917` | `1.2917` | `-.1806` | `.6932` | `-.1705` |

The fixed scale computed once was
`beta=||g_task||/||g_route||=6.1023363e-5`. The joint branch passed basic
trajectory health (`finish=.984375`, response clip `.015625`) but failed the
direction gate: NoSearch moved down as intended, while NeedSearch also moved
down. Route-only itself already failed the required `Need↑ / NoSearch↓` gate.

## Label and score-direction audit

The class-only branches show that the token IDs, labels and loss sign are
implemented correctly:

| Route-only training subset | grad norm | Δ Need | Δ NoSearch | Effect |
|---|---:|---:|---:|---|
| NeedSearch only | `873.195` | `+.2222` | `+.2273` | both margins rise |
| NoSearch only | `2763.451` | `-.2500` | `-.2841` | both margins fall |

Thus the failure is not a swapped label or reversed margin. Each class mainly
produces a global route-intercept update through the shared Transformer. The
NoSearch route loss was `1.1273` versus NeedSearch `.2254`, and its gradient
norm was about `3.16×` larger despite equal class counts. In the balanced
route-only branch, that harder NoSearch correction dominated and pulled both
classes toward internal.

This is stronger evidence than the earlier optimizer sweep: even direct
root-token supervision does not produce a class-conditional one-step update in
the shared parameter space under this independent logistic objective.

## Attribution limitation: trajectories were not identical

RP-0 recorded a SHA-256 over every complete response-token and response-mask
sequence. Separate VeXact jobs with the same global seed did not reproduce the
same per-request continuations:

| Subset | exact trajectories common to task/route/joint | total |
|---|---:|---:|
| all | `9` | `64` |
| NeedSearch | `6` | `32` |
| NoSearch | `6` | `32` |

Pairwise exact overlap on the full batch was only `17–18/64`. Therefore the
planned task-vs-route gradient cosine is deliberately reported as `null`; a
norm identity across different sampled trajectories would not be a valid
causal attribution. The beta estimate is useful as an observed scale ratio,
but it is not a strict same-trajectory calibration.

The route-only directional result remains interpretable because its loss uses
only the frozen root prompt, Boundary label and the two root logits; sampled
continuation tokens do not enter that loss. Task-only, joint and cosine
comparisons carry the trajectory limitation above.

## System health

- All nine branches completed one optimizer step and checkpoint evaluation.
- Full balanced branches had finish rates `.984375–1.0` and response clip
  `.015625`.
- Format validity was `1.0`, final-answer reserve violations were zero, and
  mixed route support remained present.
- Exact root logits were extracted at the final prompt position that predicts
  response token zero; labels were derived in actor row order from the already
  dispatched `extra_fields.reward_extra_info.boundary`.
- Four unit tests cover route-loss sign, root task masking, Undetermined
  rejection and actor metadata label order.

## Decision

Do **not** start Root-Pivot @10, do not tune beta, and do not reopen the
optimizer or reward sweep. The registered gate required route-only
`NoSearch↓/Need↑`; observed route-only was `NoSearch↓/Need↓`.

A follow-up method is not auto-authorized by this run. Before selecting one,
the next plan must explicitly address both measured failures:

1. class-conditional separation rather than another global route-intercept
   update; and
2. true fixed-trajectory/direct-gradient capture if task-route cosine is still
   needed.

The earlier Route-Adapter fallback was registered for route-only PASS plus
joint FAIL. That condition was not met, so it must not be silently treated as
already unlocked.
