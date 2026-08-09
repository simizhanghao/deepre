# Results Board — Evidence-Cost-Aware Deep Research Agent

> Updated **2026-08-09**. Smoke / val-200 / GRPO smoke128 unless noted.  
> Model family: **Qwen2.5-3B-Instruct** → SFT-v1 → GRPO (veRL + SGLang, 4×A100).  
> Plan freeze: [ROADMAP.md](ROADMAP.md) · [NEXT_STEPS.md](NEXT_STEPS.md)

## Executive summary

| Phase | Status | One-line outcome |
|-------|--------|------------------|
| 0–1 | done | HotpotQA contracts + Candidate-BM25 baselines |
| 2 (SFT) | **CLOSED** | Freeze **SFT-v1** as RL init |
| 3A | **CLOSED** | Search-agent rollout smoke OK |
| 3B | **CLOSED @100** | Pipeline OK; **no-search shortcut** |
| **3C** | **CLOSED @400** | Evidence restores search; answer+evid ↑; search→1 |
| **3C-GEN** | **PASS** | **dev-200** Agent EM 3C **0.54** > SFT 0.475 > 3B 0.19; search 1.0 vs 0.09 ([PHASE3C_GEN](PHASE3C_GEN.md)) |
| **3D0** | **DONE** | calib-512 offline → **λ_s=0.40** (0.20/0.30 cannot stop Evid farming) ([PHASE3D0](PHASE3D0.md)) |
| **3D1** | **FAIL @250** | λ=0.40 → search=0 after step5, KL~0.58; not a tradeoff ([PHASE3D1](PHASE3D1.md)) |
| 3D1b / 3D2 | **NEXT** | Lower λ smoke and/or capability-aware cost |
| 3E / CIPO | later | Full-Corpus; CIPO if evidence-use/gold audit fails |
| P4 | later | matched-step + GRPO vs REINFORCE |
| 5M multimodal | **deferred** | After text ECA |

**Final ckpts (local, not in git):**

```text
outputs/sft_qwen25_3b_coldstart_v1_merged
outputs/rl/grpo_sftv1_smoke/global_step_100          # 3B
outputs/rl/grpo_sftv1_evidence_3c/global_step_400    # 3C
```

---

## Phase 2 — SFT freeze (val-200)

Details: [PHASE2_CLOSED.md](PHASE2_CLOSED.md) · [PHASE2E4](PHASE2E4_SFTV1_BASELINES.md)

| Setting | Base | SFT-v0 | **SFT-v1** |
|---------|-----:|-------:|----------:|
| Direct EM | 0.180 | 0.170 | **0.175** |
| Candidate EM | 0.435 | 0.470 | **0.485** |
| Oracle EM | 0.595 | 0.650 | **0.660** |
| Evid F1 Oracle | — | 0.818 | **0.835** |
| Evid F1 Candidate | — | 0.665 | **0.725** |
| route internal/search | — | 29%/71% | **12%/88%** |

---

## Phase 3A — Rollout smoke

| Run | finish | search_count | notes |
|-----|-------:|-------------:|-------|
| n=8 | 1.0 | 1.0 | obs mask OK |
| n=32 | 0.969 | — | closed → 3B |

---

## Phase 3B — Answer-only GRPO @100

Details: [PHASE3B2.md](PHASE3B2.md) · [audit](../results/phase3b2_grpo_sftv1_baseline_step100_20260809/)  
Reward: \(R=EM+0.1\times\mathrm{format}\)

| Window | score (approx) | answer | search | zero_std | notes |
|--------|---------------:|-------:|-------:|---------:|-------|
| early | ~0.23 | — | — | — | pipeline up |
| 61–100 | ~0.29 | ~0.205 | **0** | **0.77** | **pathology** |

**Close reason:** format/finish≈1 but policy learns **never search** + high group zero-std.

---

## Phase 3C — Evidence GRPO @400

Details: [PHASE3C.md](PHASE3C.md) · [audit](../results/phase3c_grpo_sftv1_evidence_step400_20260809/)  
Reward: \(R=EM+0.5\times\mathrm{EvidF1}+0.1\times\mathrm{format}\)  
Init: **SFT-v1** (not 3B ckpt). Stopped at 400 by request after disk-full resume from 300.

| Window | answer | evidence | search_rate | zero_std | finish | kl |
|--------|-------:|---------:|------------:|---------:|-------:|---:|
| 1–50 | 0.098 | 0.272 | 0.408 | 0.193 | 0.970 | 0.003 |
| 51–100 | 0.203 | 0.498 | 0.950 | 0.220 | 0.971 | 0.016 |
| 101–200 | 0.482 | 0.558 | 1.000 | 0.351 | 0.998 | 0.013 |
| 201–300 | 0.572 | 0.590 | 1.000 | 0.479 | 0.999 | 0.016 |
| 301–350 | 0.602 | 0.595 | 0.999 | 0.430 | 0.999 | 0.020 |
| **351–399** | **0.614** | **0.617** | **0.999** | **0.582** | **0.999** | **0.015** |

### Head-to-head (same infra)

| Metric | 3B@61–100 | 3C@61–100 | 3C@351–399 |
|--------|----------:|----------:|-----------:|
| answer | 0.205 | ~0.22 | **0.614** |
| evidence | ~0 | **~0.51** | **0.617** |
| search | 0 | **~0.98** | **~1.0** |
| zero_std | 0.77 | **~0.24** | 0.58 |

**Close reason:** Evidence objective succeeds; late **search≡1** + rising zero_std → **3D Cost**, not longer 3C.

### Ops notes

- Disk full @~324; resume from `global_step_300`; `SAVE_FREQ=50`.  
- TB: `outputs/rl/tensorboard/grpo_sftv1_evidence_3c` (port 6007).  
- `grad_norm` spikes = GRPO noise (OK); entropy≠loss; `agent/*` = behavior monitors.

---

## Ablation tree (locked)

```text
SFT-v1
 ├── 3B Answer+Format           CLOSED @100
 ├── 3C Answer+Evidence+Format  CLOSED @400
 └── 3D Answer+Evidence−Cost(+Dup)  NEXT (fresh from SFT-v1)

Multimodal (Phase 5M): separate branch after text mainline — see ROADMAP.md
```

## Immediate next

1. **3D1b** lower λ smoke (`0.10–0.20`) and/or **3D2** capability-aware cost  
2. Do not GEN-eval collapsed λ=0.40 as success  

## Causal chain (current)

```text
3B  Answer-only → search=.09  Answer=.19   (no-search)
3C  +Evidence   → search=1.00 Answer=.54   (always-search, GEN PASS)
3D0 offline     → λ_s=0.40 flips I ranking on synthetic pairs
3D1 Uniform@0.40→ search=0 after step5, KL↑  (FAIL — global kill-switch)
```

## What is not claimed

- smoke128 train-window ≠ leaderboard; **dev-200 is a development set** (not final untouched test).  
- Not production cost-optimal yet (3D1 not run).  
- Not open-web / multimodal (L3 / 5M deferred).
