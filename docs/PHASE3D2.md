# Phase 3D2 — Capability-Aware Cost

## Status

| Item | Status |
|------|--------|
| Tooling | READY |
| Bootstrap \(p_{int}\) (SFT-v1, agent_notool) | DONE · coverage 100% · mean≈0.197 |
| **Window-1 GRPO @50** | **DONE** · ckpt `outputs/rl/grpo_sftv1_cap_3d2/global_step_50` |
| Segmented →400 | **HOLD** — wait routing diagnosis |

Log: `logs/grpo_grpo_sftv1_cap_3d2_to50_20260809_194901.log`  
TB run: `grpo_sftv1_cap_3d2`

## Locked knobs

| Item | Decision |
|------|----------|
| \(\lambda_s\) | **0.30** |
| \(\lambda_e\) / \(\lambda_f\) | 0.5 / 0.1 |
| Cost | \(\mathbf{1}[N_s>0]\) |
| Prompt | agent identity + tools DISABLED (do not advertise `<search>`) |
| Sampling | T=0.9, n=4, fixed seeds |
| Refresh | 50; window-1 = single 50 |
| Missing sid | `ECA_PINT_STRICT=1` |
| Segmented | same OUT_DIR + resume (after PASS) |

\[
R = R_A + 0.5(1-p_{int})R_E + 0.1 R_F - 0.30\, p_{int}\,\mathbf{1}[N_s>0]
\]

## Window-1 results (@50)

### Train-window means

| steps | search | answer | evidence | finish | KL | \(\Delta_{route}\) |
|------:|-------:|-------:|---------:|-------:|---:|-------------------:|
| 1–10 | 0.41 | 0.09 | 0.22 | 0.93 | 0.001 | −0.17 |
| 11–20 | 0.54 | 0.06 | 0.27 | 0.96 | 0.001 | −0.05 |
| 21–30 | 0.73 | 0.04 | 0.34 | 0.98 | 0.003 | +0.01 |
| 31–40 | 0.91 | 0.04 | 0.43 | 0.98 | 0.007 | −0.06 |
| **41–50** | **0.98** | **0.24** | **0.49** | **0.99** | **0.015** | **~0** |

- Search extinction streak: **0** (unlike Uniform λ≥0.10)  
- Late search → **always-search** (~0.98)  
- Mean \(\Delta_{route}\) over run ≈ **−0.05** (not >0)  
- Answer mid-collapse then recovers late (step50 ans≈0.42 on that batch)  
- KL healthy (≪ 3D1’s 0.5+)

### vs 3C early (same budget)

| | search 41–50 | answer 41–50 |
|--|-------------:|-------------:|
| 3C | 0.68 | 0.07 |
| **3D2** | **0.98** | **0.24** |

3D2 late searches *more* than 3C early (batch \(p_{int}\) mostly low → cost often near 0).

### vs Uniform Cost (failure ref)

| | behavior by ≤50 |
|--|--|
| λ=0.10 / 0.40 | search → 0 |
| **3D2** | search never collapses |

### Bootstrap \(p_{int}\) table

hist: 0×92 · 0.25×8 · 0.50×5 · 0.75×9 · 1.0×14 · mean 0.197  
Phase2 Direct sanity: high-\(p_{int}\) Direct✓≈0.74 vs low≈0.05.

## Verdict (window-1)

**SOFT_PASS on stability / FAIL on routing.**

| Gate | Result |
|------|--------|
| Not extinct like Uniform | **PASS** |
| Not KL explode | **PASS** |
| \(\Delta_{route}>0\) / conditional routing | **FAIL** (≈0 or negative) |
| Avoid always-search lock-in | **FAIL** late (~0.98) |

Interpretation: gating prevented global kill-switch, but frozen SFT-v1 \(p_{int}\) (mostly 0) makes most questions “search-free cost”, so policy drifts to Evidence farming / always-search. **Do not auto-continue to 400** until refresh@50 + routing diagnosis.

## Next (recommended)

1. Export HF @50 → refresh \(p_{int}\) v1 → measure \(\|p_{int}^{v1}-p_{int}^{v0}\|\)  
2. Stratified audit: search rate on \(p_{int}\in\{0.75,1.0\}\) trajectories  
3. Only if \(\Delta_{route}\) improves → segmented resume; else adjust gate / λ_s on high \(p_{int}\)

## Artifacts (code)

- `src/rl/rewards_3d2.py`
- `scripts/build_capability_pint_table.py` / `audit_pint_table.py`
- `scripts/run_grpo_capability.sh` / `tmux_grpo_capability.sh`
- `scripts/run_phase3d2_segmented.sh`
- `scripts/selftest_rewards_3d2.py`
