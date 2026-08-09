# Phase 3D0 — Uniform Cost offline calibration

> Status: **DONE** (2026-08-09). No training.  
> Artifacts: `data/rl/phase3d0_calib_512/` · `results/phase3d0_offline_lambda_sweep/`  
> Chosen for 3D1: **λ_s = 0.40** (operational). Strict offline flip-all: **0.50**.

## Goal

Answer: how large must search cost be to prefer **internal on capability-I** without killing **search on search-required-S**?

Formula under test:

\[
R = R_A + 0.5 R_E + 0.1 R_F - \lambda_s N_{search}
\]

## Protocol

| Item | Choice |
|------|--------|
| Calib | 256 I (Direct✓) + 256 S (Direct✗∧Oracle✓) = **512** |
| Disjoint | exclude smoke128 train/val + hotpotqa_200 |
| Trajectories | synthetic paired (CPU): internal vs search+typical evidence |
| Typical Evid | ~2/3 gold SF → EvidF1≈0.67 (matches 3C-GEN) |
| λ grid | 0, 0.05, 0.10, 0.20, 0.30, 0.40, **0.50** |
| Not used for λ | frozen **dev-200** (formerly val-200) — selection on calib only |

## Results (prefer rates)

| λ_s | prefer-internal on I | prefer-search on S | mean Δ(search−internal) on I |
|----:|---------------------:|-------------------:|-----------------------------:|
| 0.00 | 0.00 | 1.00 | +0.35 |
| 0.05 | 0.00 | 1.00 | +0.30 |
| 0.10 | 0.00 | 1.00 | +0.25 |
| 0.20 | 0.00 | 1.00 | +0.15 |
| 0.30 | 0.00 | 1.00 | +0.05 |
| **0.40** | **0.76** | **1.00** | **−0.05** |
| **0.50** | **1.00** | **1.00** | **−0.15** |

### Takeaways

1. **λ∈{0.05…0.30} cannot stop evidence-farming** on I when both answers are correct (Evidence +0.5·F1 still outweighs cost).  
2. Break-even ≈ `0.5 × EvidF1` ≈ 0.33 for typical F1; **0.40** is first operational flip.  
3. Search-required (S) stays prefer-search even at **0.50** (margin ≫ 0).  
4. Therefore 3D1 starts at **λ_s=0.40**, not 0.20.

## Eval naming (locked)

```text
hotpotqa_200  →  development benchmark (dev-200)
                 used for SFT/3C selection — NOT final untouched test
Phase 4       →  freeze a disjoint 500–1000 final held-out
```

## Next

```bash
# 3D1 fresh GRPO @400 (do not resume 3C)
STEPS=400 ECA_SEARCH_COST_WEIGHT=0.40 bash scripts/tmux_grpo_cost.sh
```

See [PHASE3D1.md](PHASE3D1.md).
