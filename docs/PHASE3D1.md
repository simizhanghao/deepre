# Phase 3D1 — Uniform Cost GRPO

> Status: **READY TO LAUNCH** (after 3D0). Fresh from SFT-v1.  
> Reward: \(R=R_A+0.5R_E+0.1R_F-0.40\,N_{search}\)  
> Matched budget: **400 steps** (same as 3C@400).

## Frozen knobs (= 3C except cost + OUT_DIR)

```text
Init           = SFT-v1 merged (NOT 3C@400)
λ_e            = 0.5
λ_s            = 0.40   # 3D0 operational
duplicate      = 0
n / batch / lr = 4 / 8 / 1e-6
max_search     = 2
train          = smoke128
SAVE_FREQ      = 50
resume_mode    = disable
OUT_DIR        = outputs/rl/grpo_sftv1_cost_3d1
TB             = :6008  experiment grpo_sftv1_cost_3d1
```

## Launch

```bash
tmux kill-session -t eca-grpo-3d1 2>/dev/null || true
STEPS=400 ECA_SEARCH_COST_WEIGHT=0.40 bash scripts/tmux_grpo_cost.sh
tmux attach -t eca-grpo-3d1
```

## R&D PASS gate (→ skip or trigger 3D2)

Vs 3C-GEN dev-200 (after merge+Agent eval of 3D1@400):

| Axis | Target |
|------|--------|
| Answer EM | drop ≲ 2–3pp vs 3C 0.540 |
| Evid F1 | drop ≲ 3–5pp vs 0.667 |
| search_rate | ≤ 0.80 (≥20% relative ↓ from 1.00) |
| Routing | I/Direct✓ search↓ ; S/search-req search stays high |
| Stability | finish≳0.95, no NaN, KL sane |

If only global bias moves → **trigger 3D2** capability-aware cost.
