# Next Steps — Text ECA (v2)

> [ROADMAP.md](ROADMAP.md) · [RESULTS_BOARD.md](RESULTS_BOARD.md)

## Locked root cause (do not re-open)

Under `results/16_audit_routing_exploration/worker_mismatch/` (**CLOSED**):

| Step | Result |
|------|--------|
| Path C / B-current | SGLang **80/80 search**; first-gen length OK; stop OK |
| HF Root Score | `p̃_internal≈0.65` → not π≈0 |
| sampler-align | **top_p falsified**; HF@.95 NoSearch internal≈**28%** |
| greedy-tim | HF greedy = search; tok0 agree PathB **100%** |
| Path B forensic | params OK (`T=0.9,top_p=0.95,top_k=-1`); **SGLang logp(tok0)≈−0.003 (p≈0.997)** vs HF ≈−0.39 (p≈0.68) |

**Hard verdict:** `SGLANG_ROUTE_TOKEN_LOGIT_TIM`  
Trainer/HF vs SGLang rollout are **not the same behavior policy** on the route token.  
Further SGLang kernel forensics / Mixed-action / Branching / REINFORCE / top_p=1-as-fix: **STOP**.

### vLLM replacement check (2026-08-11)

The same frozen-20 route-root calibration also **failed on vLLM**:

- median / P95 HF-vLLM search-token logprob delta: `0.270759 / 0.476559`
  (Gate A limits: `0.02 / 0.05`)
- 320/320 natural samples selected `<search>`
- `P(internal | NoSearch)=0`; mixed-action group rate `0`

Verdict: `VLLM_ROUTE_TOKEN_GATE_A_FAIL`. Temperature is not causal because the
greedy raw-logprob probe already disagrees. Do not use vLLM for formal GRPO;
continue to VeXact, then HFExact on the locked two-working-day fallback.

**Upgraded scope:** for Evidence@400 + Qwen2.5-3B + BF16, both tested
high-throughput rollout paths remove meaningful `<internal>` support at the route
root. Treat this as a cross-backend trainer/rollout implementation mismatch, not
an SGLang-only bug. Do not test a third ordinary inference backend.

---

## NOW — Rollout Alignment Recovery (`results/17_rollout_alignment/`)

**Current gate (2026-08-11): `EXACT_ROLLOUT_GATE_A_PASS`.** Frozen-20
VeOmni↔VeXact full logits and fused-LCE logprobs are both exact (`max |δ|=0`);
natural sampling has `P(internal | NoSearch)=0.29545` and mixed-group rate
`1.0`. A1 is closed. **A2 is also PASS**: the real AgentLoop→VeXact path
returned 4/4 rollouts with exact prompt/checkpoint sentinels and route support.
**A3/Gate B and A4 exact 32×4: PASS. The trajectory-level optimizer sweep is
now closed after three exact-stack step-10 tests. NEXT: Root-Pivot v0.** See
`results/17_rollout_alignment/calibration/VEXACT_GATE_A1_REPORT.md`.

Boundary@50 is implemented as the staged 10/25/50 Exact-VeXact run documented
in `docs/BOUNDARY_EXACT_ROLLOUT_PLAN.md`. GRPO step 10 increased frozen-20 route
margins for both classes (`NoSearch 0.864->1.943`, `NeedSearch 1.472->2.750`),
so GRPO@25 is locked. The historical JSONL lacks exact tensors, so Phase 19
first performs a matched Evidence@400 Fixed-Policy Attribution Capture
(2-batch smoke, then gated 10-batch/640 full capture). It is forward-only:
rollout → reward → actor logprob → dump, with no backward/optimizer/scheduler/
checkpoint. The resulting exact tensors feed official veRL GRPO, GRPO-no-std,
RF++ and RF++-baseline estimators. Phase 19 then performs an optimizer-attribution
audit; RF++ baseline@10 is allowed only if its offline conditional-gradient
gate passes.

Phase 19 capture is now `CAPTURE_PASS` (640/640; 160 groups; 25 mixed
NoSearch groups). GRPO's conditional signs are correct but its competition
ratio is `1.894`, explaining the observed global search drift. RF++ baseline
passes with `G_NS=-131.17`, `G_Need=+48.30`, and `C=.427`; plain RF++ fails.
The near-equality of GRPO-no-std and RF++-baseline competition ratios identifies
prompt-local std normalization as the primary suspect. The `12.642×` exact
policy-token length gap is the registered second risk. The next candidate is
one matched RF++-baseline@10 run with `token-mean` unchanged. That online run
passed all system gates but produced global internal bias (`M_NS=-1.591`,
`M_Need=-.972`). The registered GRPO-no-std fallback was then run cleanly to
step 10 on four GPUs. It also passed exactness/trajectory/optimizer gates but
produced global internal bias (`M_NS=-1.409`, `M_Need=-.889`). Removing local
std normalization changed the early dynamics, not the inability to learn
opposite root decisions. Do not continue either branch to step25 and do not
open another optimizer sweep. Implement Root-Pivot v0: preserve task credit,
mask Undetermined, and directly supervise the two root route logits with an
initial gradient-scale-calibrated fixed coefficient. See
`docs/FIXED_POLICY_ATTRIBUTION_REPORT.md` and
`docs/GRPO_NO_STD_STEP10_REPORT.md`.

The backend-diagnosis line is now **closed**. The active program is
**Exact-Rollout ECA Closure**: advance gate-by-gate through
`A2 → A3/Gate B → A4 32×4 → Boundary@50`; do not reopen HF/SGLang/vLLM,
temperature or reward diagnosis unless a registered gate explicitly fails.

```text
results/17_rollout_alignment/
├── environment/       # SHA / pins lock
├── calibration/       # Gate A (20-Q route root)
├── parity_32x4/       # after Gate A+B
└── trajectory_budget/ # Gate B artifacts (after Gate A1 PASS only)
```

### Locked decisions (2026-08-11)

| Item | Decision |
|------|----------|
| Artifacts | **`17_rollout_alignment/`** (not under `16`) |
| Env | Freeze **`eca-verl`**; new **`eca-verl-vexact`** from VeXact official pinned stack (**do not** clone/upgrade `eca-verl`) |
| VeXact pin | Clone `verl-project/vexact` → record `HEAD` SHA under `environment/`; follow its `pyproject.toml` for veRL/VeOmni/torch/transformers |
| VeXact fail → HFExact | **2 effective working days** (not calendar); auto gate → `VEXACT_INTEGRATION_HOLD → HFEXACT_FALLBACK` |
| Exact reference | **VeOmni/FSDP actor forward**, not vanilla HF |
| Historical reference | Existing HF scores remain a continuity diagnostic; they are not the authoritative VeXact contract |
| Trajectory Budget | **Only after Gate A1 PASS**; Gate B before any formal GRPO |
| Architecture | **Minimal VeXact hook first**; abstract `RolloutBackend` only after Gate A1 PASS |
| Precision | Keep official BF16 exact stack; FP16 is a later ablation only if VeXact exactness fails |

### Sequence (A0–A4)

```text
Env A0 — Environment
  fresh eca-verl-vexact · official pins · record all SHA/version
        ↓
A1-S — 2-Q Exact Contract Smoke
  1 NoSearch + 1 NeedSearch · same checkpoint/prompt_ids/dtype/model definition
  VeOmni/FSDP actor forward ↔ VeXact rollout route-token logprobs
        │ exact smoke PASS
        ↓
A1-20 — Frozen-20 Exact Calibration
  Gate A0-HF: old HF continuity diagnostic (non-authoritative)
  Gate A1-Exact: VeOmni/FSDP actor ↔ VeXact logprobs + VeXact natural sampling
        │ PASS (2026-08-11; exact max |δ|=0, support present)
        ▼
A2 — EcaSearchAgentLoop minimal integration
        ↓
A3 — Trajectory Budget repair
        ↓
Gate B — length / clip / finish
        ↓
A4 — 32×4 Exact parity
        ↓
Boundary-aware Stage-II @50 (same reward/table; backend only)
```

### VeXact → HFExact auto-fallback (no verbal re-approval)

Effective working day = real debug time (exclude model download / image pull / queue / network).

| Day | Target |
|-----|--------|
| Day 1 | Official VeXact example installs + minimal dense rollout OK |
| Day 2 | Evidence@400 2-Q smoke, then exact-20 only after smoke PASS |

Trigger **`VEXACT_INTEGRATION_HOLD → HFEXACT_FALLBACK`** if any of:

- **A.** After 2 effective working days, Evidence@400 still cannot do minimal rollout-only  
- **B.** Official example works, but Qwen2.5-3B cannot minimal `generate`  
- **C.** Would require large VeXact/veRL **core** forks  

Then run the same exact-contract gate on HFExact. Do **not** return to SGLang/vLLM
forensics and do not try TensorRT-LLM or another ordinary high-throughput backend.

### Gate A0-HF — Historical continuity (diagnostic)

- Preserve old HF route-root scores and natural-sampling support as the historical policy reference.
- Record HF ↔ VeOmni and HF ↔ VeXact deltas, but do **not** require bitwise or `0.02/0.05` agreement.
- If VeOmni and VeXact agree with each other but both lose the HF `<internal>` mass, trigger Model-Implementation Parity Audit; do not train.

### Gate A1-Exact — Authoritative rollout contract

- reference: VeOmni/FSDP actor forward using the exact VeXact-compatible model definition
- candidate: VeXact rollout, same checkpoint, exact prompt IDs, dtype and route-token IDs
- 2-Q smoke first: one `NoSearch`, one `NeedSearch`; any large actor↔rollout divergence stops expansion to 20
- frozen-20 route tok0: median `|δ_exact| = |log P_VeXact − log P_VeOmni| ≤ 0.02` nat; P95 `≤ 0.05`
- NoSearch: `P(internal) > 0.10` and `mixed_action_group_rate > 0`
- If logprob aligned but stochastic internal still 0 → debug **sampler**, not reward

Gate outcomes:

- **Exact aligned + support present** → `EXACT_ROLLOUT_GATE_A_PASS`; proceed to A2/A3.
- **Exact aligned + both all-search while old HF has internal mass** → Model-Implementation Parity Audit (`HF ↔ VeOmni`); training forbidden.
- **Actor ↔ VeXact not aligned** → `VEXACT_GATE_A_FAIL`; debug only within the two-working-day budget, then HFExact fallback.

Gate A1 **before** any AgentLoop, Trajectory Budget or reward change.

### A2 — Minimal AgentLoop integration smoke

Status: **PASS** (`1 internal / 3 search`, max sampled route probability
`0.73096`, all identity sentinels PASS).

- 1 NoSearch + 1 NeedSearch, `N=2`, `lr=0`, response cap 128
- real veRL `EcaSearchAgentLoop` dispatches generation through the registered
  VeXact async server and returns non-empty token/logprob outputs
- first-generate-only: this proves interface/registration/weight-path wiring;
  it intentionally makes **no** multi-turn stop, tool, finish or budget claim
- sentinels: exact A1 canonical-prompt hashes; Evidence@400 config/tokenizer/index
  hashes; sampled route-token probability must not return to the historical
  `p(search)≈0.997` collapse (`max sampled route logp ≤ -0.05`)
- artifact: `results/17_rollout_alignment/trajectory_budget/a2_agent_loop_smoke/`

A2 PASS unlocks A3, where VeXact per-turn stopping and the registered trajectory
budgets are repaired before Gate B. It does not unlock training by itself.

### A3 — Multi-turn trajectory repair

Implement the budget in tokens at every turn:

- total trajectory `2048`
- each assistant turn `≤256`
- each observation `≤384`
- preserve `≥256` tokens for the final answer

Remove production reliance on the shared last-token sentinel `stop_token_ids=[29]`.
Termination must recognize the complete `</search>`, `</internal>` and
`</answer>` token sequences; if VeXact cannot stop on a sequence, truncate at
the complete sequence in AgentLoop before appending the next observation.

### Gate B — Trajectory Contract (after Gate A; before GRPO)

- 8–16 frozen questions × 2 rollouts, `lr=0`, real multi-turn AgentLoop
- `finish_rate ≥ 0.95`
- `clip_ratio < 0.05` (prefer `< 0.01`)
- budgets: total 2048 · max assistant turn 256 · max obs 384 · reserve answer 256
- final-answer missing rate approximately 0; observations never consume the reserve
- inspect turn generation lengths and confirm route support remains present

**Gate B FAIL → forbid Boundary@50.**

Attempt 1 completed all 16 real trajectories without runtime failure and passed
the route-support and hard-budget checks, but finished only 13/16. The three
failures were long second-turn answers capped at 256 tokens before closure.
Retry keeps the 256-token cap and total budget unchanged: capped, unclosed turns
receive one further bounded continuation opportunity. Closing-tag detection also
checks decoded token prefixes because Qwen can tokenize the same XML suffix
differently depending on its left context.

Retry 1 passed Gate B from the saved rollout artifacts: finish `16/16`, clip
`0`, missing answer `0`, reserve violations `0`, unresolved unclosed turns `0`,
`P(internal|NoSearch)=0.1667`, mixed-action group rate `0.125`. Two 256-token
fragments used the bounded continuation path and subsequently closed normally.

### A4 — Exact AgentLoop parity (`32×4`, no training)

Freeze Evidence@400, the historical 32 questions and Boundary labels,
`N=4`, `T=0.9`, `top_p=0.95`; change only SGLang → VeXact. PASS requires:

- `P(internal | NoSearch) > 0.10`
- mixed-action group rate `>0`

This gate closes the counterfactual-support question. It does not require
already-perfect boundary routing.

Status: **PASS** on 128 real AgentLoop trajectories. Finish `1.0`, clip `0`,
missing answer `0`, unresolved closures `0`, `P(internal|NoSearch)=0.31818`,
mixed-action group rate `0.59375`. See
`docs/EXACT_ROLLOUT_CLOSURE_REPORT.md`.

### Boundary @50 PASS (after Exact Rollout)

- NeedSearch `search_rate ≥ 0.85`
- NoSearch `search_rate ≤ 0.70` (prefer ≤0.50)
- `Δ_boundary ≥ 0.20` (prefer ≥0.30)
- Answer/Evidence loss vs Evidence@400 within ~2–3pp / ~3–5pp

Run the same Evidence@400 initialization, data, Boundary table, reward and GRPO
settings; the rollout backend is the causal change. At steps `0/10/25/50`, run
the frozen 2-Q VeOmni↔VeXact alignment sentinel and require near-zero delta.
Monitor Answer, Evidence F1, both conditional search rates, `Δ_boundary`,
NoSearch internal/mixed support, zero-std group rate, route entropy/logprobs,
clip/importance ratios and search count. Do not relax the registered gate.

### After Boundary@50

- PASS → freeze `Candidate ECA-v1`; add OSR/USR to evaluation; refresh the
  Boundary table once from checkpoint@50 (dual search-disabled/search-enabled
  probe), then one short @50 run before fresh held-out and Full-Corpus Wikipedia.
- FAIL with mixed support but no learning → clean GRPO↔REINFORCE optimizer-only comparison.
- FAIL with importance-ratio drift, clip spike and entropy/reward collapse → SAPO.
- FAIL because root-action support disappears → Root-action Branching/BPO.

These methods are failure-signature branches, not parallel experiments and not
pre-authorized before Boundary@50 supplies the corresponding evidence.

### Forbidden until Gate A (+ B for train)

- Mixed-action GRPO · Root Branching / BPO · REINFORCE · α/reward retune  
- More SGLang kernel digging · Importance Sampling / OAPL as primary fix  
- Blind Evidence@400 retrain · CIPO/CIGPO · premature `RolloutBackend` refactor
- TensorRT-LLM / third ordinary backend · temperature escalation · FP16 before BF16 exact-stack diagnosis

### If Gate A1 PASS then Boundary@50 still FAIL

Only then: **REINFORCE** (optimizer-only) **or** Root-action Branching — same Exact backend.

---

## Later

Candidate ECA freeze → Full-Corpus Wikipedia → CIPO if evidence-use bottleneck → Phase4 → Open-Web / multimodal

## Naming

`hotpotqa_200` = **dev-200**. Train: `data/rl/train_smoke_128`.  
Frozen sample IDs: reuse from `results/16_.../worker_mismatch/sample_ids.json` as **inputs**; all new outputs under `results/17_rollout_alignment/`.
