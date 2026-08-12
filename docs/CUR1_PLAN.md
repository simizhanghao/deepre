# CUR-1 — Counterfactual Outcome Router

Status: **train/validation capture PASS; offline B0–B6 selection next. Test sealed.**

## Research-mode transition

CUR-0 closed causal diagnosis. The active question is no longer whether search
utility exists or whether the exact rollout system works; it is whether a
frozen pre-action representation can predict model-specific search uplift.

From CUR-1 onward:

```text
one expensive counterfactual acquisition
→ all cheap offline baselines/ablations together
→ one untouched held-out decision
```

An expensive experiment is allowed only for a new question distribution, new
policy, or new environment. Layer, hidden size, loss, probe, and regularization
comparisons must reuse existing offline artifacts.

## Frozen policy and intervention contract

- Policy: Evidence@400, frozen; forward-only VeXact exact rollout.
- Environment: Candidate-BM25 and the current AgentLoop contract.
- Arms: same canonical prompt followed by `do(search)` or `do(internal)`.
- Search forces only the root search opening; the model writes the query and
  continues naturally. Internal disables tools for the complete episode.
- Outcomes: F1 primary, EM secondary. Policy failures remain outcomes;
  infrastructure failures are retried/excluded with explicit accounting.
- CUR-0 and all historical train/evaluation IDs are excluded from fresh splits.

## One-shot data design

CUR-0 becomes pilot/development data and is never final test data.

| Split | Questions | Rollouts per arm | Trajectories | Role |
|---|---:|---:|---:|---|
| Existing CUR-0 pilot | 128 | existing N=4/8 | 1632 | development/train only |
| Fresh train | 640 | N=1 | 1280 | outcome-model fitting |
| Fresh validation | 128 | N=4 | 1024 | architecture/calibration selection |
| Fresh test | 128 | N=8 | 2048 | one final untouched decision |

New acquisition totals 896 questions and 4352 trajectories. Pilot plus fresh
train gives 768 training questions. Before any rollout, write immutable split
IDs, prompt hashes, exclusion hashes, model/tokenizer hashes, and seed.

The first train/validation capture used batch 16 (80/16 steps). The locked
next-run throughput profile preserves the same trajectory counts and sampling
contract while changing only scheduling: train batch 64 / 20 steps,
validation batch 32 / 8 steps, `max_num_batched_tokens=16384`, VeXact
`max_cache_blocks=1024`, and actor/ref/rollout log-prob micro-batch 4. Smoke
remains at its conservative batch/memory settings.

Breadth is concentrated in train; replication is concentrated in validation
and test. No additional sampling is triggered by individual model results.

## CUR-Uplift v1 representation

All features are extracted at the final canonical-prompt token before action.

### Semantic view

- Layer-18 hidden state.
- PCA to 64 dimensions.
- Every PCA/scaling transform is fitted on training questions only inside each
  fold, then applied to validation/test. No global preprocessing is allowed.

### Representation-dynamics view

- `||h27-h18||`, `||h36-h27||`;
- cosine similarities `(18,27)`, `(27,36)`, `(18,36)`;
- hidden norms and layer-update ratios;
- route margins at layers 18/27/36, defined consistently as the search versus
  internal logit difference after the frozen final norm and LM head.

Do not concatenate three full 2048-dimensional states.

## Potential-outcome model

The primary model predicts two potential outcomes rather than a binary route:

```text
mu_internal(q) = sigmoid(b(q) - tau(q)/2)
mu_search(q)   = sigmoid(b(q) + tau(q)/2)
delta(q)       = mu_search(q) - mu_internal(q)
```

`b(q)` captures question difficulty; `tau(q)` captures search uplift. Training
uses every rollout-level `Y(q,A)` directly.

Loss is question-balanced and arm-balanced:

```text
L = mean_q [ 0.5 * mean_rollouts L(search,q)
           + 0.5 * mean_rollouts L(internal,q) ]
```

Thus CUR-0 N=8 questions do not receive twice the weight of N=4 questions.
The current search cost is exactly one call under `do(search)`, so no cost head
is fitted. Deployment is `search iff predicted_delta_F1 > lambda`.

## One offline comparison matrix

All candidates use identical splits and report out-of-fold/development results
together; none unlocks new trajectory collection:

| ID | Candidate |
|---|---|
| B0 | constant / always-internal / always-search / random budget |
| B1 | frozen root margin |
| B2 | Layer-27 linear |
| B3 | Layer-27 small MLP |
| B4 | Layer-18 linear |
| B5 | semantic-only CUR outcome model |
| B6 | primary two-view CUR-Uplift v1 |

Architecture and calibration are selected using pilot/train plus fresh
validation only. Fresh test is evaluated once after selection. AUROC is a
secondary diagnostic, not the selection target.

## Held-out metrics and gate hierarchy

Primary metrics on fresh test:

- effect ranking: Spearman(`predicted_delta`, `delta_F1`);
- effect error: RMSE and paired bootstrap difference versus B2;
- decision regret against the paired potential-outcome oracle;
- Oracle Utility Recovery at search budgets 25%, 50%, and 75%;
- F1/search-rate and EM/search-rate frontiers.

Proposed gates from the design memo:

All A/B/C results are emitted by one sealed-test evaluation invocation after
model/PCA/checkpoint/hyperparameter hashes are locked. Priority is `C > B > A`.

### Gate A — effect estimation evidence

- report Spearman, RMSE and MAE;
- compare CUR and B2 with per-question paired squared-error differences
  `d_q = error_CUR(q)^2 - error_B2(q)^2`;
- A-Strong iff the upper 95% question-level paired-bootstrap CI of mean `d_q`
  is below zero;
- A-Weak iff point-estimate error improves but that CI crosses zero;
- hierarchical bootstrap (questions, then rollouts within question) is a
  sensitivity analysis.

Gate A controls only whether estimator-superiority can be claimed. It cannot
veto a router that passes Gate C.

### Gate B — conditional ranking

- test Spearman must exceed 0.40;
- ranking must improve over frozen root margin and B2 Layer-27 linear;
- AUROC remains secondary.

### Gate C — decision value (highest hard gate)

- matched search budgets are fixed at 25%, 50%, and 75%;
- report answer F1, regret, and Oracle Utility Recovery;
- Recovery@50 >= 0.75 and mean Recovery@25/50/75 >= 0.70;
- CUR must exceed random, frozen root-margin, and B2 Layer-27 linear at matched
  budgets; compare policy value against the best baseline with question-level
  paired bootstrap;
- Gate C FAIL closes static CUR even if Gate A looks strong;
- Gate C PASS can produce a Candidate even if only A-Weak holds, but the paper
  may claim decision-useful ranking rather than estimator superiority.

Fixed search budgets are primary. A lambda sweep is secondary and produces a
quality/search-rate frontier without selecting a privileged lambda.

### Gate D — deployment parity

Run only after Gate C PASS, on a fixed 32–64-question validation subset at N=1.
It is an engineering contract check, not a second performance estimate:

- router decision equals AgentLoop first action;
- internal tool violations equal zero;
- routed search is executed;
- finish and exact sentinels pass.

Fresh test is not reused by Gate D.

## Terminal branches

- Gate C PASS followed by Gate D PASS: freeze CUR Candidate, then perform one
  on-policy forced-arm refresh for CUR-v2 after the policy/environment changes.
- FAIL: close static pre-action routing. The only remaining upgrade is a short-
  internal/self-knowledge router using answer confidence, hidden trajectory,
  and entropy. No further static layer/MLP/probe sweep is allowed.
- Uncertainty features are part of the terminal post-internal branch, not an
  automatic next experiment after any individual baseline.

## Test seal

Fresh test is opened once by `evaluate_cur1_final.py`, which verifies locked
model, PCA, split, budget-grid and outcome hashes, then emits Gates A/B/C and
the final scientific verdict together. No model or calibration changes are
allowed afterward. Gate D uses validation only.
