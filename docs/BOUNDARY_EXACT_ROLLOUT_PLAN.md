# Exact-Rollout Boundary Learning

Status: Step 10 `REVIEW_STEP10_DIRECTION`; GRPO@25 is locked. Phase 19
Fixed-Policy Attribution capture is `CAPTURE_PASS`; RF++ baseline@10 is the
selected candidate pending report review.

## Causal contract

The experiment changes only the rollout backend relative to the historical
Boundary Stage-II run. The following are frozen and hash-checked before every
segment:

- initialization: Evidence@400
- train/validation data: original smoke-128 pool
- Boundary-v1 table
- `src/rl/rewards_boundary.py`
- GRPO, `N=4`, temperature `0.9`, top-p `0.95`
- evidence weight `0.5`, NoSearch cost `0.30`, actor LR `1e-6`

No boundary refresh, reward retuning, REINFORCE, SAPO or branching is allowed in
this run.

## Execution nodes

Training is split into targets 10, 25 and 50 so each research gate is inspected
before spending the next block. The optimizer scheduler horizon is always fixed
at 50, including the 10- and 25-step processes; resuming therefore preserves the
same learning-rate schedule as one uninterrupted 50-step run.

Each node saves one resumable checkpoint containing model, optimizer and RNG
state plus a lightweight HF model. After the next node and its evaluation pass,
the superseded full checkpoint is removed; lightweight step-10/25 artifacts and
the full step-50 checkpoint remain.

## Telemetry

Every training step records:

- `sr_need`, `sr_no`, `delta_boundary`, OSR and USR
- overall, NoSearch and NeedSearch mixed-group rates
- `delta_R_NS = E[R_internal] - E[R_search]` in mixed NoSearch groups
- `delta_R_Need = E[R_search] - E[R_internal]` in mixed NeedSearch groups
- zero-std group rate, reward/advantage std and positive-advantage token rate
- gradient norm, PPO clipping, KL
- actor/rollout importance-ratio p01/p10/p50/p90/p99
- Gate-B finish, response clipping, answer reserve and per-turn budgets

At steps 10/25/50, a frozen-20 VeXact root-logit probe records
`M_route=logP(search)-logP(internal)`. A frozen two-prompt sentinel compares the
checkpoint's VeOmni/FSDP implementation with VeXact using full logits and
fused-LCE logprobs. That checkpoint sentinel is the exact-alignment hard gate.
Whole-trajectory `old_log_probs` versus `rollout_log_probs` is retained only as
an importance-ratio diagnostic: async multi-turn behavior/recomputed policy
logprobs are not an exact backend-parity comparison and therefore do not stop
training.

## Registered decisions

- Step 10: continue only if NoSearch search rate or NoSearch route margin moves
  toward internal without a clear NeedSearch collapse. Exact-alignment failure
  is a hard stop; loss of internal/mixed support is policy collapse, not TIM.
- Step 25: target `delta_boundary >= 0.10`, mixed groups `>0.15`, and no clear
  NeedSearch collapse. Reward gaps determine whether a failure points to reward
  or optimization.
- Step 50: NeedSearch search rate `>=0.85`, NoSearch search rate `<=0.70`, and
  `delta_boundary >=0.20`; final val-200 Answer/Evidence regression must remain
  within the previously registered tolerance.

Only the observed failure signature can unlock REINFORCE, SAPO, Root-BPO or an
on-policy Boundary refresh.

## Phase 19 — Optimizer Attribution

Step 10 closed the systems question but failed the registered direction gate:
the frozen-20 route margin moved from `0.864 -> 1.943` on NoSearch and
`1.472 -> 2.750` on NeedSearch. Full logits and fused-LCE remained exact
(`max |delta|=0`), mixed groups remained available, reward gaps were usually in
the intended direction, and clipping/importance-ratio/KL telemetry was healthy.
This is a global-search-gradient failure, not TIM, exploration loss or SAPO's
importance-drift signature.

The historical 640-row JSONL is summary-only and cannot support exact offline
attribution. Phase 19 therefore performs a **Fixed-Policy Attribution Capture**:
Evidence@400 is frozen and the matched `N=4`, `T=.9`, `top_p=.95` VeXact
rollout/reward/actor-logprob path is run with `_update_actor` replaced by an
identity hook. There is no backward, optimizer step, scheduler step or
checkpoint. This estimates the learning signal at the training start; it is not
presented as a replay of the changing-policy historical 10-step run.

First run 2 batches / 128 trajectories. The smoke must prove exact reward/mask/
old-logprob tensors, group size four, valid canonical prompt hashes and exact
root probabilities. Only then run 10 batches / 640 trajectories. Directly call
the official veRL definitions for `grpo`, GRPO without per-group std scaling,
`reinforce_plus_plus`, and `reinforce_plus_plus_baseline`. Report Boundary/route
counts, exact policy-token lengths, reward decomposition, advantage mass,
root-route gradient proxy `g=A*(1[action=search]-p_search)`, Gradient
Competition Ratio and Length Gradient Ratio.

The offline data-completeness gate is hard: per trajectory it requires total
and component rewards, prompt group, exact policy-token mask/count, chosen root
action, and root `p_search`. Missing fields must produce
`OPTIMIZER_OFFLINE_DATA_INCOMPLETE`; aggregate metrics or response-span length
must not be presented as exact attribution.

`OPTIMIZER_OFFLINE_PASS` for an estimator requires `G_NoSearch < 0`,
`G_NeedSearch > 0`, `|G_NoSearch| >= 0.25*|G_NeedSearch|`, and at least 15
mixed NoSearch prompt groups in the full capture.

- If RF++ baseline passes, run a strictly matched Evidence@400 -> step-10
  `reinforce_plus_plus_baseline` experiment. The only causal change is the
  advantage estimator and its required internal options.
- Its primary gate is frozen-20: NoSearch margin `<0.864` and NeedSearch margin
  `>=1.272`; train-batch rates are diagnostic only.
- If RF++ baseline fails offline, do not spend a GPU run. Move to Root-Pivot /
  Boundary-localized credit assignment.
- Do not run GRPO@25, naked RF++, SAPO, BPO, Boundary refresh or reward retuning
  while Phase 19 is unresolved.
- Stop after the four-estimator report. Do not automatically launch the winning
  optimizer; the next GPU training run is selected only after report review.

Formal result: 640 trajectories / 160 groups passed all tensor-integrity gates;
`P(internal|NoSearch)=.371`, mixed NoSearch groups `25`, exact policy-token
length ratio `12.642×`. GRPO produced `G_NS=-39.24`, `G_Need=+56.93`,
`G_U=+17.41`, `C=1.894` and thus a net search push. RF++ baseline produced
`G_NS=-131.17`, `G_Need=+48.30`, `G_U=+7.70`, `C=.427`. Plain RF++ failed the
NoSearch sign; GRPO-no-std passed as a secondary ablation. See
`docs/FIXED_POLICY_ATTRIBUTION_REPORT.md`.

## Phase 20 — RF++ baseline staged validation

Phase 19 identifies prompt-local std normalization as the primary suspect:
GRPO-no-std and RF++ baseline reduce Gradient Competition from `1.894` to
`.437/.427`. The exact `12.642×` search/internal policy-token gap is the
pre-registered second risk, but Phase 20 keeps `loss_agg_mode=token-mean` so the
only causal change from the failed Exact-VeXact GRPO line is
`algorithm.adv_estimator=reinforce_plus_plus_baseline`.

Freeze Evidence@400 initialization, data/order/seed, VeXact AgentLoop, Boundary
table/reward, `N=4`, `T=.9`, `top_p=.95`, LR, batch size and trajectory budget.
Before GPU optimization, saved Phase 19 tensors must match a direct call to the
official veRL RF++-baseline estimator and the resolved Hydra config must name
that exact estimator.

At step 10, the frozen-20 hard gate is two-dimensional relative to Evidence@400
(`M_NS=.864`, `M_Need=1.472`):

- direction: `M_NS < .864` (prefer change `<= -.20`)
- preservation: `M_Need >= 1.272` (prefer `>=1.472`)
- VeOmni↔VeXact exact sentinel PASS, mixed support `>0`, and no KL/clip/
  importance-ratio failure signature

`RFPP_BASELINE_DIRECTION_PASS` continues from its own step-10 optimizer state to
step 25. Step 25 requires NeedSearch SR `>=.85`, NoSearch SR `<=.70`, boundary
delta `>=.20`, mixed support and exact/trajectory gates. PASS continues unchanged
to step 50 and the Candidate ECA-v1 quality/cost/system evaluation.

Failure branches are locked by signature:

- step-10 still globally search → RF++ baseline plus only
  `seq-mean-token-mean` at @10
- step-10 globally internal → GRPO-no-std @10
- step-10 PASS but step-25 degrades → dual search-disabled/search-enabled probe,
  then on-policy Boundary refresh if labels changed
- only after estimator, length and Boundary freshness are cleared may Root-Pivot
  / step-level credit assignment be considered

SAPO and branching remain excluded because importance drift and counterfactual
support scarcity are absent.

### Phase 20 step-10 result

RF++ baseline completed ten optimizer steps and passed exact alignment,
trajectory, mixed-support and optimizer-health checks, but failed the frozen
direction gate. Frozen-20 margins were `M_NS=-1.591` and `M_Need=-.972`:
the NoSearch direction improved, while NeedSearch crossed into a strong global
internal shift instead of staying `>=1.272`. Therefore its optimizer state is
not continued to step 25. Per the locked failure tree, the next and only active
branch is GRPO with `norm_adv_by_std_in_grpo=false`, still using
`loss_agg_mode=token-mean`, from Evidence@400 to step 10.

### GRPO-no-std step-10 result

The clean four-GPU run completed and formally returned
`GRPO_NO_STD_DIRECTION_FAIL`. NoSearch margin improved from `.864` to `-1.409`,
but NeedSearch collapsed from `1.472` to `-.889`; this is global internal bias,
not conditional routing. Exact full-logit and fused-LCE deltas were zero, the
trajectory gate passed, mixed support was `.625`, clip fraction was zero and
importance-ratio P99 was `1.079`. Therefore the optimizer/normalization sweep is
closed. The next method is Root-Pivot decision-aligned credit, as detailed in
`docs/GRPO_NO_STD_STEP10_REPORT.md`.
