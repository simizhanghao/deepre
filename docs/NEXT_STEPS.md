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

---

## NOW — Rollout Alignment Recovery (`results/17_rollout_alignment/`)

```text
results/17_rollout_alignment/
├── environment/       # SHA / pins lock
├── calibration/       # Gate A (20-Q route root)
├── parity_32x4/       # after Gate A+B
└── trajectory_budget/ # Gate B artifacts (after Gate A PASS only)
```

### Locked decisions (2026-08-11)

| Item | Decision |
|------|----------|
| Artifacts | **`17_rollout_alignment/`** (not under `16`) |
| Env | Freeze **`eca-verl`**; new **`eca-verl-vexact`** from VeXact official pinned stack (**do not** clone/upgrade `eca-verl`) |
| VeXact pin | Clone `verl-project/vexact` → record `HEAD` SHA under `environment/`; follow its `pyproject.toml` for veRL/VeOmni/torch/transformers |
| VeXact fail → HFExact | **2 effective working days** (not calendar); auto gate → `VEXACT_INTEGRATION_HOLD → HFEXACT_FALLBACK` |
| Trajectory Budget | **Only after Gate A PASS**; Gate B before any formal GRPO |
| Architecture | **Minimal VeXact hook first**; abstract `RolloutBackend` only after Gate A PASS |

### Sequence (A0–A4)

```text
A0 — Environment
  fresh eca-verl-vexact · official pins · record all SHA/version
        ↓
A1 — Minimal VeXact Calibration
  Evidence@400 · exact 20 samples · route root only
  (prompt_ids → token_ids + logprobs; no AgentLoop / no budget edit)
        ↓
Gate A — HF vs VeXact route logprob + natural sampling
        │ PASS
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
| Day 2 | Evidence@400 exact-20: `token_ids` + `logprobs` + sampling (AgentLoop not required) |

Trigger **`VEXACT_INTEGRATION_HOLD → HFEXACT_FALLBACK`** if any of:

- **A.** After 2 effective working days, Evidence@400 still cannot do minimal rollout-only  
- **B.** Official example works, but Qwen2.5-3B cannot minimal `generate`  
- **C.** Would require large VeXact/veRL **core** forks  

Then run the **same Gate A** on HFExact. Do **not** return to SGLang forensics.

### Gate A — Rollout Alignment

- route tok0: median `|δ| = |log P_rollout − log P_HF| ≤ 0.02` nat; P95 `≤ 0.05`
- NoSearch: `P(internal) > 0.10` and `mixed_action_group_rate > 0`
- If logprob aligned but stochastic internal still 0 → debug **sampler**, not reward

Gate A **before** any Trajectory Budget change (isolate causal variable).

### Gate B — Trajectory Contract (after Gate A; before GRPO)

- `finish_rate ≥ 0.95`
- `clip_ratio < 0.05` (prefer `< 0.01`)
- budgets: total 2048 · max assistant turn 256 · max obs 384 · reserve answer 256
- no 100% trajectories at max length; obs must not eat final-answer reserve

**Gate B FAIL → forbid Boundary@50.**

### Boundary @50 PASS (after Exact Rollout)

- NeedSearch `search_rate ≥ 0.85`
- NoSearch `search_rate ≤ 0.70` (prefer ≤0.50)
- `Δ_boundary ≥ 0.20` (prefer ≥0.30)
- Answer/Evidence loss vs Evidence@400 within ~2–3pp / ~3–5pp

### Forbidden until Gate A (+ B for train)

- Mixed-action GRPO · Root Branching / BPO · REINFORCE · α/reward retune  
- More SGLang kernel digging · Importance Sampling / OAPL as primary fix  
- Blind Evidence@400 retrain · CIPO/CIGPO · premature `RolloutBackend` refactor

### If Gate A PASS then Boundary@50 still FAIL

Only then: **REINFORCE** (optimizer-only) **or** Root-action Branching — same Exact backend.

---

## Later

Candidate ECA freeze → Full-Corpus Wikipedia → CIPO if evidence-use bottleneck → Phase4 → Open-Web / multimodal

## Naming

`hotpotqa_200` = **dev-200**. Train: `data/rl/train_smoke_128`.  
Frozen sample IDs: reuse from `results/16_.../worker_mismatch/sample_ids.json` as **inputs**; all new outputs under `results/17_rollout_alignment/`.
