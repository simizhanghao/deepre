# Fixed-Policy Attribution Capture Report

Date: 2026-08-12
Status: `CAPTURE_PASS`; optimizer launch intentionally stopped.

## Question and scope

This experiment does not replay the changing-policy historical 10-step run.
It measures, at the frozen Evidence@400 initialization, which root-routing
signal the official veRL estimators produce under the matched rollout contract:
VeXact, train-smoke-128 order, seed 42, `N=4`, `T=.9`, `top_p=.95`, Boundary-v1
reward and the registered trajectory budget.

The execution path was rollout → reward → actor old-logprob → tensor dump.
The trainer's actor-update method was replaced by an identity hook. Both Ray
processes printed the forward-only sentinel; every step reported
`attribution/actor_update_skipped=1`; `save_freq=-1`; no checkpoint file exists.
There was no backward, optimizer step or scheduler step.

## Capture gates

The 2-batch smoke passed before the full run was unlocked: 128 trajectories,
32 groups of exactly four, complete finite estimator tensors and root logits,
`P(internal|NoSearch)=.25`, and three mixed NoSearch groups.

The formal capture also passed:

- 640 trajectories, 160/160 uid groups of exactly four
- 490 search and 150 internal routes
- Boundary rows: 348 NeedSearch, 124 NoSearch, 168 Undetermined
- `P(internal|NoSearch)=.37097`; 25 mixed NoSearch groups (gate ≥15)
- exact masked actor old-logprobs are finite
- final dense tensors have shape `640×946`; `response_mask` retains the exact
  policy-token positions while right padding is zero
- mean root `p(search)`: NoSearch `.7002`, NeedSearch `.8247`, Undetermined
  `.7888`

Raw tensors are in
`results/19_optimizer_attribution/full/raw/attribution_capture.npz`; the hash,
fixed protocol and negative update/checkpoint assertions are recorded in
`capture_manifest.json`.

## Reward and length mechanism

| Boundary × route | n | mean reward | mean policy tokens |
|---|---:|---:|---:|
| NeedSearch × search | 284 | .3344 | 265.59 |
| NeedSearch × internal | 64 | .1000 | 20.47 |
| NoSearch × search | 78 | -.2000 | 231.49 |
| NoSearch × internal | 46 | .9478 | 19.46 |
| Undetermined × search | 128 | .3057 | 235.66 |
| Undetermined × internal | 40 | .2250 | 19.73 |

The local reward directions are correct: `internal-search=+1.1478` on NoSearch
and `search-internal=+.2344` on NeedSearch. The exact policy-token length ratio
`search/internal=12.6424×`; unlike the earlier span proxy, this excludes masked
tool observations. Search trajectories therefore broadcast a learning signal
over roughly twelve times as many policy tokens.

## Official veRL estimator attribution

`g=A_root*(1[action=search]-p_search)`; positive mass pushes toward search.
`C=|G_Need+G_U|/(|G_NS|+eps)`.

| estimator | G_NS | G_Need | G_U | net G | C | offline gate |
|---|---:|---:|---:|---:|---:|---|
| GRPO | -39.244 | +56.930 | +17.414 | +35.101 | 1.894 | PASS |
| GRPO-no-std | -25.500 | +9.524 | +1.607 | -14.369 | .437 | PASS |
| REINFORCE++ | +6.427 | +2.189 | +3.293 | +11.909 | .853 | **FAIL** |
| REINFORCE++ baseline | -131.166 | +48.296 | +7.704 | -75.166 | .427 | PASS |

GRPO is not locally blind: it gives NoSearch the correct sign. It fails at the
batch level because NeedSearch plus Undetermined search-positive mass is 1.89×
the opposing NoSearch mass, yielding a net search push. This matches the actual
step-10 frozen root-margin movement and explains why correct local rewards did
not prevent global search drift.

Removing per-prompt std scaling nearly eliminates that competition imbalance:
GRPO-no-std and RF++ baseline have almost identical competition ratios (`.437`
vs `.427`). The primary mechanistic suspect is therefore GRPO's prompt-local
standard-deviation normalization, not the whole group-relative idea. RF++
baseline gives the same conditional signs with the strongest NoSearch
counterforce and 25 mixed NoSearch groups, so it is the best-supported next
optimizer-only validation. Absolute gradient magnitudes are not compared as a
quality score across differently normalized estimators; the registered sign,
relative-mass and support gates drive the decision.

The independent second risk is the `12.642×` exact policy-token length gap.
With the frozen `token-mean` loss, RF++ baseline still broadcasts a trajectory
advantage over every valid policy token. The first validation must not change
loss aggregation because doing so together with the estimator would destroy
causal identification. If RF++ baseline retains global search drift online,
length-neutral `seq-mean-token-mean` is the pre-registered next single variable.

Plain REINFORCE++ is rejected. In this multi-turn batch, its official discounted
return implementation resets across zero-masked observation positions; all
root advantages became the same globally whitened value, and NoSearch received
the wrong positive-search sign. It must not be run as the next optimizer.

## Decision

Select one matched **REINFORCE++ baseline @10** optimizer-only validation as the
next candidate, with GRPO-no-std retained as a documented ablation rather than
a parallel run. Do not change reward, data, trajectory budget, sampling,
backend or `loss_agg_mode=token-mean`. Before launch, require an official-function
parity check on saved Phase 19 tensors. GRPO@25, plain RF++, SAPO, BPO and reward
retuning remain locked.
