# Results Board — Evidence-Cost-Aware Deep Research Agent

> Updated **2026-08-11**. Smoke / val-200 / GRPO `train_smoke_128` unless noted.  
> Model family: **Qwen2.5-3B-Instruct** → SFT-v1 → GRPO (veRL; Exact Rollout recovery NOW).  
> Plan freeze: [ROADMAP.md](ROADMAP.md) · [NEXT_STEPS.md](NEXT_STEPS.md)

## Executive summary

| Stage | Status | One-line outcome |
|-------|--------|------------------|
| 0–1 | done | HotpotQA contracts + Candidate-BM25 baselines |
| SFT-v1 | **CLOSED** | Freeze as RL init |
| Rollout smoke | **CLOSED** | Search-agent loop OK |
| Answer-only GRPO | **CLOSED @100** | Pipeline OK; **no-search shortcut** |
| Evidence GRPO | **CLOSED @400** | Evidence restores search; answer+evid ↑; search→1 |
| Evidence GEN | **PASS** | val-200 Agent EM Evidence **0.54** > SFT 0.475 > Answer-only 0.19 |
| Offline cost λ | **DONE** | calib-512 → **λ_s=0.40** |
| Uniform cost | **FAIL @250** | λ=0.40 → search=0; not a tradeoff |
| Uniform λ diagram | **CLOSED** | No stable Pareto → trigger Boundary |
| Capability cost @50 | **CLOSED** | SOFT_PASS stability / FAIL routing |
| Boundary Stage-II | routing **FAIL** | search≡1 · Δ_boundary≈0 @~42 |
| Routing / TIM audit | **CLOSED** | `SGLANG_ROUTE_TOKEN_LOGIT_TIM` (HF≈0.68/0.32 vs SGLang≈0.997/0) |
| **Rollout Alignment Recovery** | **NOW** | VeXact preferred · HFExact fallback · then Boundary@50 |

**Final ckpts (local, not in git):**

```text
outputs/00_sft_v1_merged
outputs/rl/02_hf_answer_only_step100
outputs/rl/03_hf_evidence_step400
outputs/rl/04_table_search_boundary/
outputs/rl/06_ckpt_grpo_boundary/
```

Audits under `results/01_*` … `results/16_*`.

---

## SFT-v1 freeze (val-200)

| Setting | Base | SFT-v0 | **SFT-v1** |
|---------|-----:|-------:|----------:|
| Direct EM | 0.180 | 0.170 | **0.175** |
| Candidate EM | 0.435 | 0.470 | **0.485** |
| Oracle EM | 0.595 | 0.650 | **0.660** |
| Evid F1 Oracle | — | 0.818 | **0.835** |
| Evid F1 Candidate | — | 0.665 | **0.725** |
| route internal/search | — | 29%/71% | **12%/88%** |

---

## Answer-only GRPO @100

Audit: [../results/08_audit_grpo_answer_only_step100/](../results/08_audit_grpo_answer_only_step100/)  
Reward: \(R=EM+0.1\times\mathrm{format}\)

| Window | score (approx) | answer | search | zero_std |
|--------|---------------:|-------:|-------:|---------:|
| 61–100 | ~0.29 | ~0.205 | **0** | **0.77** |

**Close reason:** learns **never search** + high group zero-std.

---

## Evidence GRPO @400

Audit: [../results/09_audit_grpo_evidence_step400/](../results/09_audit_grpo_evidence_step400/)  
Reward: \(R=EM+0.5\times\mathrm{EvidF1}+0.1\times\mathrm{format}\)  
Init: **SFT-v1**. Stopped at 400.

| Window | answer | evidence | search_rate | zero_std |
|--------|-------:|---------:|------------:|---------:|
| 1–50 | 0.098 | 0.272 | 0.408 | 0.193 |
| 351–399 | **0.614** | **0.617** | **0.999** | **0.582** |

GEN: [../results/10_eval_grpo_evidence_val200/](../results/10_eval_grpo_evidence_val200/) (`gen_sft` / `gen_3b` / `gen_3c`).

**Close reason:** late **search≡1** → Boundary / Cost, not longer Evidence train.

---

## Cost / Boundary

- Offline λ: `results/11_sweep_offline_cost_lambda` · data `data/rl/calib_cost_lambda_512`
- Uniform FAIL: `results/12_audit_uniform_cost_fail`
- λ diagram: `results/13_audit_uniform_cost_lambda_diagram`
- Capability @50: `results/14_audit_capability_cost_step50`
- Boundary stop summary: `results/15_summary_boundary_grpo_stopped.md`
- Routing Exploration HF smoke: `results/16_audit_routing_exploration/` (debug n=8×2)
- Training-parity SGLang: `results/16_audit_routing_exploration/parity_sglang_32x4/` (32×4, gitignored dumps)

### Routing Exploration Audit (HF smoke 2026-08-10)

Protocol: Evidence@400 · tools-enabled first-action · T∈{0.9,1.1,1.3} · n_rollouts=2 · max_samples=8 · seed=42 · **HF `react_loop`**.

| T | P(internal\|NoSearch) | mixed_action_group_rate | note |
|--:|----------------------:|------------------------:|------|
| 0.9 | **0.50** | 0.125 | NoSearch still mixed |
| 1.1 | 0.00 | 0.125 | noisy / small-n dip |
| 1.3 | **0.50** | **0.375** | gate temperature |

**Gate:** `NATURAL_EXPLORATION_OK` (HF only).

### Training-parity Routing Exploration (SGLang 32×4, 2026-08-10)

Protocol: Evidence@400 · `EcaSearchAgentLoop` · veRL+SGLang · 8×A100 · STEPS=1 · lr=0 · n=4 · 32 Q stratified · T∈{0.9,1.3} · `GPU_MEM_UTIL=0.55`.

| T | n dump | P(internal\|NoSearch) | P(search\|*) | mixed_action_group_rate |
|--:|-------:|----------------------:|-------------:|------------------------:|
| 0.9 | 128 | **0.0** | **1.0** | **0.0** |
| 1.3 | 128 | **0.0** | **1.0** | **0.0** |

Train metrics agree: `agent/search_rate=1`, `agent/internal_rate=0`, `boundary/delta_boundary=0`.

**Gate:** `TRAINING_PARITY_EXPLORATION_FAIL`  
**Read:** real worker path has **no** spontaneous `internal` / mixed groups — HF smoke overstated exploration.  
**Superseded by:** Path C/B + sampler-align + Path B forensic → `SGLANG_ROUTE_TOKEN_LOGIT_TIM` → **Rollout Alignment Recovery** (not Mixed-action / Dual-arm).

### Routing Worker Mismatch — Path C interim (2026-08-10)

Protocol: Evidence@400 · full `EcaSearchAgentLoop` · 8×A100 · STEPS=1 · lr=0 · **20 Q × 4** (11 NoSearch + 9 NeedSearch, padded for GPU divisor) · T=0.9 · top_p=0.95 · dump `worker_mismatch/dump_pathC.jsonl`.

| metric | value |
|--------|------:|
| dump lines | **80** |
| `route_first=search` | **80/80** |
| `route_first=internal` | **0** |
| train `search_rate` / `internal_rate` | **1.0 / 0.0** |
| `sr_NoSearch` / `sr_NeedSearch` | **1.0 / 1.0** (n=44 / 36) |
| `delta_boundary` | **0.0** |
| `search_count` | **1.0** (all traj) |
| `finish_rate` / format | **1.0 / 1.0** |
| **`response_length` / `clip_ratio`** | **2048 / 1.0** ← LENGTH_CONTRACT FAIL |
| step wall | ~88.7 s |

Forensic: `stop_token_ids=[29]` only (`>`), **last-token collision** → `STOP_HANDLING_RISK` (not causal for root choice). Canonical prompt ends at `<|im_start|>assistant\n`.

**Status:** stacked issues — root all-search **and** multi-turn length pathology.  
**HF Root Score:** `BACKEND_MISMATCH_LIKELY` — NoSearch median `p̃_internal≈0.65`.  
**Path B-current (2026-08-10):** first-gen-only · `stop=[29]` · `max_new=128` → **80/80 search**;  
`response_length≈26.4` / `clip≈0.0125` / `stop_reason=stop` → **first-gen length OK**.  

**Sampler-align (2026-08-10):** `top_p` **falsified**. HF@.95 NoSearch `p_internal≈0.284`.  

**Greedy-TIM + Path B forensic (2026-08-10):**  
- HF greedy 20/20 search; Path B tok0 agree 100%  
- Forensic `sampling_params`: `T=0.9, top_p=0.95, top_k=-1, logprobs=true`（配置正确）  
- **SGLang `logp(tok0)≈−0.003` (p≈0.997)** vs HF ≈−0.39 (p≈0.68) → **route-token TIM**  
Verdict: `SGLANG_ROUTE_TOKEN_LOGIT_TIM`.  
SGLang is **not** an acceptable RL rollout contract for ECA until replaced/calibrated.  
Do **not** dig SGLang kernels further; do **not** Mixed-action / Branching / REINFORCE yet.

## Immediate next

Artifacts: `results/17_rollout_alignment/{environment,calibration,parity_32x4,trajectory_budget}/`  
(Details locked in [NEXT_STEPS.md](NEXT_STEPS.md).)

1. **A0** — fresh `eca-verl-vexact` from VeXact official pinned stack (do not clone `eca-verl`)  
2. **A1 / Gate A** — Evidence@400 exact-20 route calibration (no budget edit yet)  
3. On Gate A PASS: **A2–A3 / Gate B** → **A4** 32×4 → Boundary@50 (reward/table unchanged)  
4. VeXact blocked after **2 effective working days** → auto `HFEXACT_FALLBACK` (same Gate A)

## What is not claimed

- `train_smoke_128` train-window ≠ leaderboard; **dev-200 is a development set**.  
- Not production cost-optimal yet.  
- Not open-web / multimodal.
