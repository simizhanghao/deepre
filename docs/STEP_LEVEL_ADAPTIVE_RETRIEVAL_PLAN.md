# Phase 25 — Step-Level Adaptive Retrieval Preregistration

Status: **S0 PASS; S1 causal headroom PASS; the single Step Preference Gate is next.**

S0 result: fixed Train32 passed all hard checks after one preserved failed
attempt exposed an unhandled legacy `</search>` suffix. The fixed Train8 was
then replayed independently twice with exact trajectory match `8/8`. See
[STEP_S0_REPORT.md](STEP_S0_REPORT.md).

S1 result: 1022 exact-prefix Train states produced deterministic SEARCH_NOW /
CONTINUE_NOW pairs. Cost-saving Continue is 61.94%; the local Oracle removes
25.02% of calls while improving F1 from 0.4091 to 0.4380. Decision:
`STEP_ADAPTIVE_HEADROOM_PASS`. See
[STEP_S1_HEADROOM_REPORT.md](STEP_S1_HEADROOM_REPORT.md).

## S0 code-audit finding

The current `EcaSearchAgentLoop` is not a step-level loop:

- `_STOP_STRINGS` contains only `</search>`, `</answer>`, and `</internal>`;
- `</think>` is never a generation boundary;
- a closed `<search>` is executed immediately;
- a closed `<internal>` receives a fixed final-answer nudge;
- the existing continuation path is only truncation recovery, not a reasoning
  checkpoint;
- at most two searches are already enforced, but there is no matched
  continue-vs-retrieve branch at an intermediate prefix.

Therefore existing trajectories cannot be relabelled as step-level decisions.
S0 requires a new registered loop name and must leave
`EcaSearchAgentLoop` unchanged for historical reproducibility.

## Frozen S0 interface (before implementation)

- New registry name: `eca_step_adaptive_agent`.
- It uses a separate frozen step-system prompt. Reusing the historical system
  prompt is forbidden because that prompt explicitly demands root
  Search/Internal choice and wins over the new checkpoint instruction.
- Every checkpoint is generated as two controlled raw proposal spans
  (reasoning, then candidate query). The controller normalizes only text before
  any legacy structural tag, records both raw token streams, and loss-masks the
  normalized checkpoint plus its `<think>`, `Search query:`, and `</think>`
  structure. This keeps Phase25 a separate-router experiment rather than an LM
  policy update. ANSWER similarly forces and masks `<answer>` only after the
  external ANSWER action.
- Checkpoint close sequence: decoded complete `</think>`; use the existing
  decoded-prefix fallback to handle BPE boundary merges.
- System maximum reasoning checkpoints: `4`; maximum executed searches: `3`.
  S1 counterfactual acquisition samples at most 2 checkpoints per question.
- Both counterfactual arms share byte-identical canonical prompt and assistant
  checkpoint token IDs.
- The checkpoint must end with one parseable proposed query field. Exact
  sentinel `Search query: NONE` is permitted and recorded separately. The
  model has no tool access until the branch decision is externally applied.
- Retrieve arm appends the normal masked `<observation>` turn. Continue arm
  appends a frozen masked `retrieval skipped; continue reasoning` control turn.
- Prefix/KV rule for S0: logical prefix reuse inside one AgentLoop request;
  token-cost accounting also reports a conservative replay proxy so future
  backend KV optimization cannot improve the scientific Gate silently.
- Smoke source: fixed 32 CUR-1 Train IDs. Its first 8 form the deterministic
  replay subset; greedy decoding; two independent runs must match checkpoint
  and response token IDs exactly.
- Val3 provisional freeze seed: `2026081203`. Final numerical S2 gates must be
  committed before Val3 construction, not after S0 observations.

Implementation order is fixed: parser/unit tests → two-question no-tool smoke
→ Train32 contract → 8-question deterministic replay → S1 counterfactual
acquisition. The first four stages are now complete.

## Frozen action and trajectory budgets

```text
actions                  = CONTINUE | SEARCH | ANSWER
total response budget    = 2048 tokens
observation cap          = 384 tokens
final-answer reserve     = 256 tokens
reasoning-step cap       = 128 tokens
max checkpoints          = 4
max executed searches    = 3
S1 sampled checkpoints   <= 2/question
```

`Search query:` is a proposal only. It never executes a tool. The external
state machine applies exactly one action:

- CONTINUE: append a frozen masked control turn and request another checkpoint;
- SEARCH: execute the proposal, append the normal masked observation, request
  another checkpoint;
- ANSWER: append one frozen masked final-answer nudge and generate a closed
  `<answer>...</answer>`.

`Search query: NONE` is not synonymous with ANSWER. SEARCH is invalid when the
query is NONE; CONTINUE and ANSWER remain available.

## Phase 25A system smoke

One fixed Train32 only; no scientific model selection. Hard checks:

- parser valid and `<think>` closure `1.0`;
- candidate-query field parse `>= .98`;
- finish `>= .98`;
- tool violations, duplicate-query loops and reserve violations `0`;
- response clipping `< .05`;
- an externally forced CONTINUE must reach another checkpoint where SEARCH is
  still available.

## Step counterfactual contract

At one byte-identical checkpoint prefix and proposed query, branch:

```text
do(SEARCH_NOW)   vs   do(CONTINUE_NOW)
```

After this one intervention, both arms use the same frozen completion policy
and retain future Search access. Phase25B uses greedy completion so N1 is a
deterministic potential outcome rather than a stochastic single-draw label.
The target is continuous bounded Step SkipRegret:

```text
max(0, F1_SearchNow - F1_ContinueNow)
```

Question-balanced and checkpoint-balanced weights prevent questions with more
states from dominating training.

## Query Availability — corrected observable gate

The proposed conditional `P(query != NONE | R_skip > 0)` is not identifiable
when query is NONE: SearchNow cannot be executed, so `R_skip` is missing. We
therefore report two preregistered quantities without pretending otherwise:

1. overall valid-query rate across eligible checkpoints (`>= .90` hard gate);
2. positive-regret coverage among states where a valid query makes the paired
   counterfactual observable (`>= .90`, explicitly a conditional lower-bound
   diagnostic, not the unobservable population probability).

If either fails, Step-Risk fitting is locked and the only permitted repair is
a small query-proposal SFT followed by a fresh S0/S1 capture.

## Step-Risk features and Val3 gate

Frozen first model: step mean/P10 logP and entropy; Train-only L27 PCA64; step
index, response/query lengths and query-is-NONE; previous-search count, tokens
since search and duplicate-query similarity; frozen root B3 scalar. Fit one
linear uncertainty baseline and one small MLP only.

Fresh Step-Val3: 128 questions, seed `2026081203`, excluding all historical
IDs including DSSR Val2 and original sealed Test. Primary PASS:

```text
average SearchCalls <= 0.75
F1_adaptive >= F1_AlwaysSearch - 0.02
```

Strong PASS additionally reaches AlwaysSearch F1 at `<=0.75` calls. Report a
second `<=1.0` calls operating point for Search reallocation, plus full F1 vs
calls and F1 vs TokenCost Pareto curves, response/observation tokens and
wall-clock. Only Val3 PASS may open Test once. Failure ends adaptive-routing
R&D and retains the frozen Evidence Agent.

## Why this is the sole remaining route

Static CUR and DSSR both failed their fresh validation gates. Their common
constraint is a single question-level decision made before the information
deficit has been localized. Phase 25 moves only the decision time:

```text
canonical question
→ bounded internal reasoning step
→ step uncertainty / knowledge-gap decision
→ continue internally OR retrieve for that step
→ answer; permit another decision only at the next frozen checkpoint
```

Evidence@400, VeXact exact rollout, Candidate-BM25, answer/evidence reward and
the canonical prompt remain frozen. This is not another root Router, optimizer
sweep, or reward revision.

## Frozen causal hypothesis

For multi-hop QA, question-level observables cannot reliably reveal which
intermediate fact will be missing. A reasoning checkpoint containing the
current subclaim/query exposes that deficit. Retrieval triggered at that
checkpoint should preserve Always-Search quality while reducing calls/tokens.

## Stages and locks

### S0 — execution contract smoke

- 8 open Train questions only; original Test and DSSR Val2 are forbidden.
- Greedy Evidence@400 reasoning with explicit, parseable step checkpoints.
- At each checkpoint save token IDs, chosen-token log-probs, entropy/margin,
  L27 state, proposed search query, retrieval decision and full token costs.
- Re-run the same 8 questions twice; response/checkpoint tokens and extracted
  features must be bit-exact.
- Retrieval must restart/continue under one explicitly recorded context rule;
  no hidden leakage from gold/supporting facts.

### S1 — open-Train counterfactual capture

- Use CUR-1 Train640 only.
- An eligible checkpoint has a valid non-`NONE`, nonduplicate candidate query
  and admits both SEARCH and CONTINUE under the frozen state budget. Select a
  singleton once; otherwise select the earliest and latest eligible checkpoint,
  never more than two per question.
- Pair `SEARCH_NOW` with `CONTINUE_NOW` under matched prefix token IDs. Replay
  identical prior actions to the target; after the single intervention both
  arms return to the same deterministic frozen completion policy and retain
  future Search access.
- Preserve quality and costs separately:

  ```text
  delta F1, delta EM, delta actual SearchCalls,
  response/observation/raw-generation tokens and total token proxy
  ```

- Save candidate query, retrieved document IDs/titles, supporting-title hits,
  Recall@5, and duplicate status in the same capture.
- Before fitting any Router, compute SearchHelpful, ContinueSafe,
  CostSavingContinue and the Local Oracle quality-cost frontier. Hard headroom
  requires CostSavingContinue `>= .25` and Oracle quality at `>=25%` retrieval
  reduction no worse than fixed Search by `.02` F1. Failure stops adaptive
  routing.
- Only after headroom PASS fit one weighted-BCE lexicographic Step Preference
  Gate: quality differences beyond `.02` choose the better arm; within `.02`,
  choose the cheaper arm. No B0-B6 or loss sweep.

### S2 — fresh Val3 freeze and decision

- Draw a new 128-question split from the same source pool with a new seed,
  excluding every historical ID including DSSR Val2 and original sealed Test.
- Freeze IDs, prompts, contexts, checkpoint policy, model and feature hashes
  before generation.
- Compare Always Search, Never Search, fixed-interval retrieval, uncertainty
  baseline and the one step router.
- Report answer F1, retrieval calls, TokenCost, latency, finish/format, and
  risk–coverage.

### S3 — original Test

Original Test remains sealed. It may open once only after S2 thresholds and
the final checkpoint/router bundle are frozen.

## S0 implementation decisions that must be resolved from code evidence

Before acquisition, inspect the current AgentLoop parser and stop/continue
contract and freeze:

1. the exact token sequence defining a reasoning checkpoint;
2. whether continuation reuses KV/prefix or performs a canonical replay;
3. maximum checkpoints/search calls per question;
4. how a generated query is represented without granting the model tool
   access on the continue-without-search arm;
5. the Val3 seed and numerical quality/cost gates.

No GPU acquisition starts until these five values appear in the frozen S0
manifest and an independent contract audit passes.

## Final Step Preference Gate freeze (2026-08-13)

S1 passed the causal headroom gate.  The final router is therefore a
conservative safe-skip classifier: its default action is SEARCH, and it emits
CONTINUE only when the current search can be skipped with sufficiently low
quality risk.  This supersedes the earlier linear/MLP method matrix; there is
one small MLP and no further loss, architecture, feature, or threshold sweep.

Training labels are frozen to the lexicographic S1 preference outcome:
`SEARCH=74`, `CONTINUE=948`.  The sample weight is
`max(abs(delta_F1), 0.02)` and weighted BCE additionally uses the Train-fold
class ratio `N_continue/N_search` for the SEARCH positive class.

Leakage control is strict:

- five outer folds grouped by question; all checkpoints from one question
  stay in one fold;
- L27 PCA64 and scalar standardization are fitted on outer-train only;
- the root/static scalar uses the existing strict OOF K0/B3 prediction during
  Gate OOF, never the full-Train B3 prediction;
- inner early stopping also splits by question;
- three fixed seeds are averaged; the final three models are refitted on all
  open Train states using the median selected epoch per seed.

The frozen input is the candidate-query final token's native-HF L27 state,
plus step mean/P10 chosen-token logP, mean checkpoint predictive entropy, step
index, previous-search count, tokenized query length, maximum lexical
similarity to a previous query, and the frozen root B3 scalar.  Exact replay is
reconstructed from the frozen step prompt and response prefix, and every
prompt/state SHA256 must match before a feature is accepted.

The deployment threshold is selected once from OOF predictions.  SEARCH is
the safe default.  Among thresholds capturing at least 95% of total positive
Search regret, choose the one with minimum paired counterfactual search-call
cost (then token cost).  After full-Train refit, freeze model, PCA, scaler,
threshold, tokenizer/config and feature-schema hashes.

Only then construct fresh Val3 and run the integrated on-policy comparison:
No Search, old Evidence@400 Always Search, Step-AllSearch, and Frozen Step
Gate (AllContinue is diagnostic only).  The scientific PASS is at most 0.75x
the Step-AllSearch calls with F1 no worse than 0.02; the project PASS is lower
TokenCost than old AlwaysSearch with F1 no worse than 0.02.  Use paired
question bootstrap.  Val3 and the original Test remain sealed during fitting.

## S2 final status (2026-08-13)

The one-shot Fresh Val3 evaluation is complete on 128 previously unused
questions (seed `2026081203`).  All four arms used one greedy trajectory per
question; all finished with valid answers.  The frozen Step Gate preserved the
Step-AllSearch answer exactly at aggregate level, but reduced retrieval calls
from `2.0625` to only `2.0000` per question (`0.9697x`, a `3.03%` reduction).
This misses the preregistered `<=0.75x` scientific cost gate, so Scientific
PASS is false.

The old Evidence@400 root-Search baseline remained substantially stronger and
cheaper: F1 `0.56427` at `1.0` call and `636.62` response-token cost, versus
Step Gate F1 `0.36496` at `2.0` calls and `1182.77` tokens.  Project PASS is
therefore also false.  The paired Step-Gate minus Step-AllSearch F1 difference
is exactly `0.0`; calls differ by `-0.0625` per question (95% paired-bootstrap
CI `[-0.109375, -0.0234375]`).

The failure is diagnostic rather than infrastructural.  The full online Gate
made 267 learned eligible decisions and emitted only 11 learned CONTINUE
actions below threshold.  Later search compensation left a net saving of only
8 calls over 128 questions.  This is consistent with Train OOF, where the
95%-regret-capture threshold already predicted only a `5.05%` call reduction;
Fresh Val3 realized `3.03%`.  Thus the primary limitation is the frozen
safety-calibration/coverage frontier, not a rollout crash or a large surprise
distribution shift.  Separately, the bounded Step scaffold itself underperforms
the old root-Search agent by `0.19931` F1 while costing `546.16` more tokens,
which closes this version as a deployable replacement.

Per the frozen decision rule, S3 remains locked: the original Test was not
opened.  Phase 25 closes as `STEP_VAL3_FAIL`; no post-hoc threshold retuning on
Val3 is permitted.
