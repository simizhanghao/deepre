# Results Board — Evidence-Cost-Aware Deep Research Agent

> Updated **2026-08-12**. Smoke / val-200 / GRPO `train_smoke_128` unless noted.
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
| **Rollout Alignment Recovery** | **CLOSED/PASS** | VeXact exact contract established |
| Root-Pivot RP-0 | **FAIL** | direct route loss still moved both classes internal; @10 locked |
| CUR-0 | **CLOSED** | 1632 rows; bidirectional utility; margin rejected; Layer-27 linear modest; small MLP unlocked |
| CUR-1 static | **CLOSED/FAIL** | B3 Recovery@50=.380, F1@50=.423; Validation Unlock FAIL; test sealed |
| DSSR Safe-Skip | **FAIL** | Val2 K3@50 F1/Recovery/TokenCost=.5294/.3240/.6228; Test sealed |
| Step-Level Adaptive Retrieval S0 | **PASS** | Train32 contract PASS; deterministic Train8 replay exact 8/8; S1 next |
| Step-Level Adaptive Retrieval S1 | **PASS** | 1022 exact causal pairs; safe-Continue=.6194; Oracle -25.02% calls and +.0289 F1 |

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

**vLLM Gate A (2026-08-11): FAIL.** Frozen Evidence@400 exact-20 calibration:
median / P95 HF-vLLM route-logprob delta `0.270759 / 0.476559`; all `320/320`
natural samples chose `<search>`; `P(internal | NoSearch)=0`. This mismatch is
already present in the greedy raw-logprob probe, so temperature is not causal.
Verdict: `VLLM_ROUTE_TOKEN_GATE_A_FAIL` → continue VeXact, then HFExact fallback.
See `results/17_rollout_alignment/calibration/VLLM_GATE_A_REPORT.md`.

**Exact-contract correction (2026-08-11):** old HF remains a historical
continuity reference, but the authoritative Gate A1 comparison is
**VeOmni/FSDP actor forward ↔ VeXact rollout** under the same model definition,
checkpoint, prompt IDs and BF16 stack. Run one NoSearch + one NeedSearch smoke
before frozen-20. Exact agreement with all-search triggers HF↔VeOmni
Model-Implementation Parity Audit; actor↔VeXact disagreement triggers
`VEXACT_GATE_A_FAIL` and the two-working-day HFExact fallback.

**VeXact Gate A1 (2026-08-11): PASS.** Frozen-20 full-vocabulary logits and
fused-LCE logprobs match VeOmni exactly (`20/20`, maximum absolute difference
`0.0`). VeXact natural sampling preserves route support:
`P(internal | NoSearch)=0.29545`, mixed-action group rate `1.0`, other count `0`.
Verdict: `EXACT_ROLLOUT_GATE_A_PASS`. See
`results/17_rollout_alignment/calibration/VEXACT_GATE_A1_REPORT.md`.

Backend diagnosis is closed. Active line: **Exact-Rollout ECA Closure** —
A2 AgentLoop wiring, A3/Gate B trajectory contract, A4 exact 32×4 support,
then the causal Boundary@50 rerun. Boundary@50 carries a frozen 2-Q alignment
sentinel at steps 0/10/25/50.

## Immediate next

Artifacts: `results/17_rollout_alignment/{environment,calibration,parity_32x4,trajectory_budget}/`  
(Details locked in [NEXT_STEPS.md](NEXT_STEPS.md).)

1. ✅ **Env A0** — official-pinned `eca-verl-vexact`; original `eca-verl` untouched
2. ✅ **A1-S** — 2-Q exact-contract smoke (`max |δ|=0`)
3. ✅ **A1-20 / Gate A1-Exact** — exact contract + stochastic support PASS
4. ✅ **A2 AgentLoop→VeXact** — 4/4 rollouts; identity sentinels PASS;
   routes `1 internal / 3 search`; max sampled route probability `0.73096`
5. ✅ **A3 / Gate B** — finish `16/16`, clip/missing/reserve violations `0`
6. ✅ **A4 no-train 32×4** — finish `128/128`,
   `P(internal|NoSearch)=0.31818`, mixed-group rate `0.59375`
7. ❌ Exact Boundary GRPO/RF++/GRPO-no-std optimizer line closed: global route bias
8. ❌ Root-Pivot RP-0: 9/9 branches completed; route-only
   `ΔNoSearch=-.273`, `ΔNeed=-.278`; formal @10 locked. See
   [ROOT_PIVOT_RP0_REPORT.md](ROOT_PIVOT_RP0_REPORT.md).

## What is not claimed

- `train_smoke_128` train-window ≠ leaderboard; **dev-200 is a development set**.  
- Not production cost-optimal yet.  
- Not open-web / multimodal.
# Phase 24 DSSR

- Val2 freeze: `DSSR_VAL2_FREEZE_AUDIT_PASS` (128 fresh questions; seed
  2026081202; 3046 historical IDs excluded; Test sealed).
- SK-0: `DSSR_SK0_REPLAY_PASS` on 8 questions; response tokens and endpoint
  hidden states were bit-exact across independent runs; valid/closed rate 1.0.
- Train Probe audit: PASS; Probe/Search mean F1 `0.2893/0.5495`, mean
  SkipRegret `0.3188`, positive-regret rate `0.3891`.
- K0-K3 Train freeze: PASS. Strict OOF B3 priors were used for K3 stacking.
- Val2 Search4: PASS, 512 rows, F1 `0.6116`. Final DSSR decision:
  `SELF_KNOWLEDGE_ROUTER_FAIL`. K3@50 F1/Recovery/TokenCost-ratio are
  `.5294/.3240/.6228`; gates are fail/fail/pass/fail. Test remains sealed.
- Next and only allowed routing phase: preregistered reasoning-step-level
  adaptive retrieval. See `DSSR_VAL2_REPORT.md`.

# Phase 25 Step-Level Adaptive Retrieval

- Independent `eca_step_adaptive_agent`; historical AgentLoop unchanged.
- Fixed Train32 final audit: 32 rows, 112 checkpoints; parser, close, query,
  finish, and CONTINUE-to-next rates all `1.0`; 48 searches; zero clipping,
  duplicate search execution, tool violation, or answer-reserve violation.
- A preserved first attempt failed on two legacy `</search>` suffixes. The
  structural normalizer was completed and the exact same Train32 rerun passed.
- Fixed Train8 replay: both A and B independently passed 8 rows / 16
  checkpoints; complete trajectory exact-match rate `1.0` (8/8), including
  raw and normalized token IDs, masks, actions, queries, and answers.
- Gate: `STEP_S0_REPLAY_PASS`. Val3 and original Test were not read. S1
  Train640 matched counterfactual acquisition is next.

## S1 causal headroom

- Train640 base and 2044 scientific branch rows completed; exact prefix/action
  parity `1.0` across 1022 SEARCH_NOW/CONTINUE_NOW pairs.
- Search-helpful / Continue-safe / cost-saving-Continue rates:
  `.0528/.9472/.6194`; strict F1 ties are `.8855`.
- SearchNow versus ContinueNow mean F1: `.40907/.40953`; mean calls:
  `2.3464/1.7006`; mean token-proxy delta `+185.20` for SearchNow.
- Query evidence: any supporting-title hit `.8297`, supporting-title Recall@5
  `.5866`.
- Local Oracle at `.2502` retrieval reduction improves F1 from `.4091` to
  `.4380` (floor `.3891`). Gate: `STEP_ADAPTIVE_HEADROOM_PASS`.
- Next: one frozen lexicographic weighted-BCE Step Preference Gate, then fresh
  integrated Val3 only. Original Test remains sealed.
