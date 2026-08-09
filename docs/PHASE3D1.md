# Phase 3D1 — Uniform Cost GRPO (λ_s=0.40)

> Status: **FAIL / STOPPED @~250** (2026-08-09). Do **not** continue to 400.  
> Audit: [results/phase3d1_uniform_cost_lambda040_stopped/](../results/phase3d1_uniform_cost_lambda040_stopped/)  
> Ckpt kept: `outputs/rl/grpo_sftv1_cost_3d1/global_step_250` (also 50/100/150/200)

## Setup

```text
Init     = SFT-v1 (fresh, resume=disable)
Reward   = EM + 0.5 EvidF1 + 0.1 Format − 0.40 × N_search
Train    = smoke128, n=4, batch=8, lr=1e-6, max_search=2
Target   = 400 steps (matched 3C budget) → stopped early
```

λ_s=0.40 came from 3D0 offline operational pick ([PHASE3D0.md](PHASE3D0.md)).

## Train-window metrics (`[phase3c]` lines)

| Window | answer | evidence | search_rate | KL | zero_std | finish |
|--------|-------:|---------:|------------:|---:|---------:|-------:|
| 1–20 | 0.138 | 0.174 | **0.080** | 0.004 | 0.244 | 0.944 |
| 21–50 | 0.166 | 0.219 | **0.000** | 0.016 | 0.300 | 0.964 |
| 51–100 | 0.182 | 0.245 | **0.000** | 0.050 | 0.423 | 0.983 |
| 101–150 | 0.206 | 0.294 | **0.000** | **0.227** | 0.485 | 0.994 |
| 151–200 | 0.233 | 0.308 | **0.000** | **0.484** | 0.500 | 0.998 |
| **201–250** | **0.257** | **0.329** | **0.000** | **0.575** | **0.645** | 0.999 |

Only **5 / 251** logged steps had `search_rate > 0`; last nonzero was **step 5**. After that, search is **identically zero**.

## vs 3B / 3C (reference)

| | Answer (dev-200 GEN) | Search | KL (train late) |
|--|---------------------:|-------:|----------------:|
| 3B@100 | 0.19 | 0.09 | ~0.08 (61–100) |
| 3C@400 | **0.54** | **1.00** | ~0.015 |
| **3D1@250 train** | ~0.26 (train window) | **0.00** | **~0.58** |

3D1 did **not** produce a quality–cost Pareto point. It reproduced the **no-search shortcut** with worse KL than 3B.

## Why TB looked “unconverged”

Not “needs more steps”:

1. **Policy target flipped to never-search** → no stable “always-search” attractor like 3C.  
2. **KL rising 0.01→0.58** → drifting from SFT, not locking a healthy mode.  
3. **High entropy / grad spikes** → unstable updates under dead search + sparse answer signal.  
4. `actor/loss≈0` is misleading; watch `search_rate` + `kl_loss`.

## Root cause

```text
R = … − 0.40 × N_search

Online GRPO:
  one search ≈ −0.40 total
  Evidence gain only if search happens and SF matches
  → early batches learn: never call search
  → search_rate → 0 (after step 5)
  → Evidence channel useless for retrieval
  → Answer only weak internal EM (~0.25)
  → KL explodes while zero_std climbs
```

3D0 offline assumed paired **correct** search trajectories still available. Online, the policy **stops generating search**, so the offline ranking optimum never appears as a living tradeoff.

Also: λ=0.40 was calibrated to stop *unnecessary* search when both answers are correct; under stochastic GRPO it became a **global search bias kill-switch**.

## Problems (explicit)

1. **Uniform Cost @0.40 fails the 3D1 PASS gate** (search not ≤0.8 with quality held — search went to 0 and quality far below 3C).  
2. **Offline λ ≠ online behavior** — need online smoke λ ladder or capability-aware cost.  
3. **Matched-400 budget aborted** — continuing would only deepen no-search + KL.  
4. **dev-200 Agent GEN for 3D1 not run** (not worth until a non-collapsed ckpt exists).  
5. Intermediate ckpts 50–200 are diagnostic only; **do not** treat as deployable Cost agent.

## Verdict

> **FAIL.** Uniform Cost with λ_s=0.40 is not a valid ECA Cost solution under this stack. It collapses to 3B-like no-search with KL blowup.

## Next (ROADMAP gate → triggered)

```text
Option A (NOW): 3D1b online λ∈{0.05,0.10,0.15,0.20} × 40–60 step — see PHASE3D1B.md
                 pick first λ with search_rate ∈ [0.4, 0.9] and KL stable
Option B (planned): 3D2 Capability-Aware Cost
                 (uniform cost only shifts global bias — gate condition met)
```

Recommended path: **short lower-λ probe (A) then 3D2 if still no routing structure**.
