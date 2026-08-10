# Phase 3D2b — On-policy Search-Boundary-Aware ECA

## Why (from 3D2-v0@50)

3D2-v0 = capability-only (\(p_{int}\)) gated cost:

- ✅ blocked Uniform search-extinction  
- ❌ late search≈0.98, \(\Delta_{route}\approx0\)  
- Root cause: ~72% \(p_{int}=0\) → reward degenerates to 3C; \(p_{int}=0\) ≠ search-utility  

Close **3D2-v0**. Upgrade to **boundary** via search-disabled **vs** search-enabled (SAAS-style).

## SAAS formal boundary (δ=2, n=4)

\[
n_d=\#\{\text{correct in search-disabled}\},\quad
n_e=\#\{\text{correct in search-enabled}\}
\]

\[
S(q)=
\begin{cases}
\texttt{NoSearch} & n_d \ge \delta \\
\texttt{NeedSearch} & n_d=0,\ n_e>0 \\
\texttt{Undetermined} & \text{otherwise}
\end{cases}
\]

Default \(\delta=2\) (SAAS).

## ECA reward (our adaptation — keeps Evidence)

| Boundary | Reward |
|----------|--------|
| **NoSearch** | \(R=R_A+0.1R_F-\alpha\,\mathbf1[N_s>0]\) · **Evidence OFF** |
| **NeedSearch** | \(R=R_A+0.5R_E+0.1R_F\) · **no search cost** (v1; Nmin later) |
| **Undetermined** | \(R=R_A+0.5R_E+0.1R_F\) · no search cost |

Stage-II init: **3C@400 HF** (not fresh SFT). Smoke **50 steps**, then decide.

Core metric:

\[
\Delta_{\mathrm{boundary}}=P(\mathrm{search}\mid\mathrm{NeedSearch})-P(\mathrm{search}\mid\mathrm{NoSearch})
\]

## Open questions (implementation defaults in brackets)

1. **α for NoSearch?** [0.30] align with 3D2-v0 λ_s  
2. **NeedSearch redundant cost?** SAAS uses \(-\alpha\max(0,N_s-N_{\min})\). Candidate `max_search=2` → v1 **no cost**; add Nmin in v1.1?  
3. **Correctness for boundary labels:** SAAS train uses F1; we use **EM** for \(n_d/n_e\)? [EM]  
4. **Search-enabled probe:** full multi-turn Candidate-BM25 agent vs single forced retrieval? [full agent, max_search=2, T=0.9]  
5. **Stage-II init:** 3C HF merge as `MODEL_PATH` (new OUT_DIR) vs FSDP resume from `global_step_400`? [HF merge + new dir]  
6. **Matched control:** 3C@400 continue Evidence-only 50 vs Boundary 50 — run in parallel or after smoke? [after 3D2b smoke PASS]  
7. **Refresh cost:** 128×(4 disabled+4 enabled) ≈1024 agent rollouts / refresh — host GPU vs steal train GPUs? [host GPU 4 for label; train 4–7]  
8. **STRICT missing boundary?** same as 3D2 — [FAIL if coverage&lt;1]  
9. **SAAS group-wise adv for disabled/enabled in same GRPO batch** — we only freeze labels offline; OK for smoke? [yes for v1]  
10. **When to trigger REINFORCE?** only if boundary labels OK but mixed-action groups still rare  

## Artifacts

- `scripts/build_search_boundary_table.py`  
- `src/rl/rewards_3d2b.py`  
- `scripts/run_grpo_boundary.sh` / `tmux_grpo_boundary.sh`  
- `scripts/audit_boundary_table.py`  

## Bootstrap audit @3C@400 (DONE 2026-08-09)

| Boundary | n | frac | mean \(n_d\) | mean \(n_e\) |
|----------|---|------|-------------:|-------------:|
| NeedSearch | 67 | 52.3% | 0.00 | 3.28 |
| NoSearch | 28 | 21.9% | 3.39 | 3.64 |
| Undetermined | 33 | 25.8% | 0.21 | 0.61 |

- coverage 128/128 · δ=2 · n=4 · T=0.9 · elapsed≈1.73h  
- Undetermined: 26 both-fail (`n_d=n_e=0`) + 7 weak-disabled (`n_d=1`)  
- vs 3D2-v0: ~72% `p_int=0` mislabeled as search-required → now NeedSearch only 52%  
- Artifact: `outputs/rl/boundary/boundary_latest.json` (gitignored)

## Status

TOOLING ✅ · bootstrap ✅ · Stage-II GRPO@50 **RUNNING** (manual relaunch after chain `pandas` miss)  
- train: tmux `eca-grpo-3d2b` · `logs/grpo_3d2b_latest.log` · TB `:6010`

Defaults locked for v1: α=0.30 · EM labels · full agent enabled probe · HF 3C@400 init · STRICT coverage=1
