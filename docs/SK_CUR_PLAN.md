# DSSR — Dual-Signal Safe-Skip Router

Status: **closed — `SELF_KNOWLEDGE_ROUTER_FAIL`; Test sealed.**

## Execution ledger

- `DSSR_VAL2_FREEZE_AUDIT_PASS`: seed `2026081202`, 128 questions,
  historical exclusion count `3046`, eligible pool `6329`, original Test
  sealed. Same-path deterministic rebuild reproduced an identical manifest.
- `DSSR_SK0_REPLAY_PASS`: two independent 8-question model loads produced
  bit-exact response tokens and bit-exact L18/L27/L36 endpoint states
  (`max_abs_delta=0`). Probe validity and answer closure were both `1.0`.
- Tokenizer audit: generated `<answer>\n` may merge `>\n` into one token.
  Answer boundaries are therefore frozen by decoded character spans mapped
  back to whole content tokens, rather than isolated tag-token IDs.
- Train640 Probe audit: `DSSR_TRAIN_PROBE_AUDIT_PASS`; Probe F1 `0.2893`,
  Search N1 F1 `0.5495`, mean SkipRegret `0.3188`, with 249 Search-positive,
  336 ties, and 55 Probe-positive questions. The one historical Search policy
  failure remains retained as specified.
- Model freeze: `DSSR_K0_K3_TRAIN_FREEZE_PASS`. K3 uses strict out-of-fold B3
  scores while fitting on Train, preventing stacked-feature target leakage.
  K0-K3 OOF metrics are diagnostics only and do not select a model.
- Val2 Probe1 is complete: 128/128 valid and closed, mean F1 `0.3249`.
  Search4 independently passed with F1 `0.6116`.
- Final K3@50: F1 `0.5294`, Recovery `0.3240`, TokenCost ratio `0.6228`.
  Only the cost gate passed; K3 beat all baselines only at the 25% budget.
  Decision: `SELF_KNOWLEDGE_ROUTER_FAIL`. See
  `docs/DSSR_VAL2_REPORT.md`. Question-level routing is closed and original
  Test remains sealed.

## Stage transition

Static pre-action CUR is closed after `VALIDATION_UNLOCK_FAIL`. Its best
operating-point model, B3 (L27 pre-action PCA64 dual-head MLP), reached
Recovery@50 `0.3801` and F1@50 `0.4226`, below the frozen unlock thresholds
`0.65` and `AlwaysSearch-0.02=0.4659`. Fresh Test remains sealed.

The failure is not treated as a request for a larger static router. DSSR
changes the observable information: one deterministic, tool-free short
internal attempt exposes self-knowledge before the search decision.

```text
canonical question
→ deterministic short internal probe
→ provisional answer + intrinsic confidence + post-answer states
→ estimate skip regret
→ return probe answer OR discard probe context and launch root Search Agent
```

If search is selected, the Evidence Agent restarts from the byte-identical
canonical prompt. Probe content never enters its context, preserving the
existing `do(search)` policy and outcomes.

## Frozen scope

- Policy: Evidence@400.
- Probe: force `<internal>`, then one greedy (`temperature=0`) tool-disabled
  generation with a total 96-token response cap, requiring a closed
  `<answer>...</answer>`.
- An incomplete/invalid probe deterministically routes to search.
- No verbalized confidence, self-consistency, probe sampling, extra layers,
  attention features, or new static-router sweep.
- Existing CUR-1 Train640 root-search N=1 outcomes are reused; add one probe
  per question.
- Existing Validation128 retires permanently to development history.
- Fresh Val2 contains 128 previously unused questions, each with Probe N=1 and
  root Search N=4 (640 trajectories).
- Existing Test128 remains sealed. It is opened only after all Val2 gates pass,
  with Probe N=1 and root Search N=8 (1152 trajectories).

## Frozen features

At the provisional-answer endpoint:

1. Prefix-20 statistics over the first 20 answer-content tokens: mean entropy,
   mean top1-top2 logit margin, and mean chosen-token log-prob;
2. full-answer confidence: mean answer-token log-prob, P10 log-prob, minimum
   log-prob, answer length, and closure success;
3. post-answer Layer-27 state at the last non-empty answer-content token,
   Train-only PCA64;
4. post-answer dynamics: cosine `(18,27)`, `(27,36)`, `(18,36)` and relative
   updates `18→27`, `27→36`.

The forced `<internal>` prefix, tags, and whitespace are excluded from
confidence statistics. The Probe is generated once; Prefix-20 and full-answer
features are two views of the same trajectory.

No intermediate-layer logit lens is used.

## Models

| ID | Definition |
|---|---|
| K0 | frozen static B3 from CUR-1 (searchability prior) |
| K1 | TARG-style Prefix-20 uncertainty gate |
| K2 | full Probe confidence-only Safe-Skip router |
| K3 | K0 prior + Probe confidence + post-L27 PCA64 + post-answer dynamics |

K2/K3 directly predict the bounded decision target

```text
SkipRegret(q) = max(0, F1_search(q) - F1_probe(q))
```

and rank unsafe-to-skip questions highest for Search. K3 preserves K0 as the
static searchability signal while the Probe contributes self-knowledge. No
model family is added after seeing Val2.

## Val2 decision gates

At matched 25%, 50%, and 75% search budgets, report F1, regret, Recovery,
response tokens, observation tokens, the frozen total-token proxy, measured
latency, and a Safe-Stop risk–coverage curve.

All four gates must pass:

1. Recovery@50 >= 0.65;
2. F1@50 >= AlwaysSearch F1 - 0.02;
3. TokenCost@50 <= 0.65 × AlwaysSearch TokenCost;
4. K3 exceeds `max(K0,K1,K2)` at at least two of three budgets. If a simpler
   K1/K2 already passes all safety/quality/cost gates, freeze the simpler
   passing candidate rather than privileging K3.

Failure emits `SELF_KNOWLEDGE_ROUTER_FAIL` and leaves Test sealed. No further
question-level router is allowed; the sole remaining research escalation is a
separately preregistered reasoning-step-level adaptive retrieval phase.
Passing freezes model/PCA/schema/config/split hashes before the single Test
acquisition.

## Final Test gates

- Recovery@50 >= 0.75;
- mean Recovery@25/50/75 >= 0.70;
- F1@50 >= AlwaysSearch F1 - 0.02;
- TokenCost@50 <= 0.65 × AlwaysSearch TokenCost.

Only a full Test PASS unlocks AgentLoop integration parity.

## Frozen implementation contracts

1. **Probe:** canonical prompt → forced `<internal>` → greedy continuation;
   total response cap 96. Confidence uses answer-content tokens only; post
   hidden states use the last non-empty answer-content token. Save Prefix-20.
2. **Primary TokenCost proxy:** with canonical prompt tokens `P`, Probe
   response `Rp`, Search response `Rs`, and Search observation `Os`:

   ```text
   Stop         = P + Rp
   Search       = 2P + Rp + Rs + Os
   AlwaysSearch = P + Rs + Os
   ```

   Report wall-clock latency separately; TokenCost is not called FLOPs.
3. **Val2:** HotpotQA distractor train pool, 128 questions, seed
   `2026081202`; exclude every historical train/eval/RL/CUR-0 ID and all
   CUR-1 Train/Validation/original-Test IDs. Freeze IDs, parquet, contexts,
   prompt manifest, exclusion hash, and split hash before generation.

## Terminal project cascade

```text
Level 0: Prefix-20 cheap uncertainty
Level 1: Dual-signal Safe-Skip decision
Level 2: Evidence Agent search/reasoning
optional final research phase after DSSR failure: step-level adaptive retrieval
```
