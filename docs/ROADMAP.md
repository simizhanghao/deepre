# Roadmap — Evidence-Cost-Aware Deep Research Agent (v2)

> Updated **2026-08-12 (v2.3)**. Problem-driven tech only.

> Board: [RESULTS_BOARD.md](RESULTS_BOARD.md) · Actions: [NEXT_STEPS.md](NEXT_STEPS.md)

## Principle

```text
Do NOT insert multimodal into Phase 3D.
Do NOT add AutoSearch / CIPO / REINFORCE / Branching “because papers are new”.
Tech enters only via pre-registered failure gates.
SGLang and vLLM are historical mismatch baselines —
mainline RL requires trainer-aligned Exact Rollout (VeXact / HFExact).
```

Causal story:

```text
3B  sparse answer     → no-search
3C  + evidence        → always-search
3D1 uniform cost      → quality–cost baseline FAIL routing
3D2 capability / Boundary → Stage-II routing FAIL (search≡1)
Routing audit         → SGLANG_ROUTE_TOKEN_LOGIT_TIM
                      → trainer π ≠ SGLang μ on route token
vLLM calibration      → VLLM_ROUTE_TOKEN_GATE_A_FAIL
                      → 320/320 search; high-throughput mismatch scope widened
Exact Rollout         → VeXact contract PASS
optimizer/root sweep → GRPO/RF++/no-std/RP-0 all fail conditional separation
NOW                   → Conditional Utility Router on frozen Evidence@400
```

---

## A. Text ECA mainline

```text
NOW → Conditional Utility Router (CUR)
│
├── ✅ Answer-only CLOSED @100
├── ✅ Evidence CLOSED @400 + GEN PASS
├── ✅ Offline λ → λ_s=0.40
├── ❌ Uniform Cost FAIL @~250
├── ✅ Capability-only Cost @50 CLOSED (routing FAIL)
├── ❌ Boundary Stage-II on SGLang routing FAIL (Δ≈0; search≡1)
├── ✅ Routing Sampler Alignment Audit CLOSED
│     gate = SGLANG_ROUTE_TOKEN_LOGIT_TIM
│     HF@.95 NoSearch internal≈0.28; SGLang tok0 p(search)≈0.997
│
├── ✅ Exact-Rollout ECA Closure (A1+A2+Gate B+A4 PASS) → `results/17_rollout_alignment/`
│     freeze eca-verl · fresh eca-verl-vexact (official VeXact pins; not clone eca-verl)
│     ✅ Env A0 → ✅ 2-Q smoke → ✅ frozen-20 Gate A1-Exact (max |δ|=0)
│     old HF = continuity reference; VeOmni/FSDP actor = authoritative reference
│     ✅ Gate A1 → ✅ A2 loop → ✅ A3/Gate B → ✅ A4 32×4 → NOW Boundary@50
│     Boundary GRPO step10: REVIEW/STOP (global search bias; exact stack PASS)
│     ✅ Phase19 fixed-policy 640 + four-estimator attribution
│     ❌ RF++ baseline@10 → global internal (exact/system gates PASS)
│     ❌ GRPO-no-std@10 → global internal (exact/system gates PASS)
│     optimizer/normalization sweep CLOSED; neither continues to step25
│     ❌ Root-Pivot RP-0: balanced route-only moved both classes internal
│     calibrate fixed beta from initial gradient scales; no dev coefficient sweep
│     9/9 branches completed; beta=6.10e-5; route-only NSΔ=-.273, NeedΔ=-.278
│     per-class signs correct but shared update is global; cross-job cosine invalid
│     formal Root-Pivot @10 LOCKED; register a new conditional-separation plan
│     Boundary@50 sentinel: exact 2-Q at steps 0/10/25/50
│     PASS → Candidate ECA-v1 → one Boundary refresh → short @50 → Full Corpus
│     FAIL branches only by signature: REINFORCE | SAPO | Root Branch/BPO
│     VeXact hold after 2 effective working days → auto HFExact (same exact gate)
│     abstract RolloutBackend only after Gate A1 PASS
│     reward/table frozen until Exact backend proven
│
├── 🟡 CUR-0 causal diagnostic → `results/22_cur/`
│     fresh random 128; same canonical prompt; do(search) vs do(internal), N=4/arm
│     predict delta-F1 and search cost separately; lambda only at deployment
│     Gate 0A outcome bidirectionality → 0B frozen margin → 0C h18/h27/h36 probes
│     linear first; MLP/uncertainty only behind preregistered insufficiency gates
│     ✅ 1632 rows after N=8 refinement; margin rejected; primary linear modest
│     standalone Layer-27 MLP@128 CANCELLED (low information value)
│
├── 🟡 CUR-1 one-shot model-building phase → `docs/CUR1_PLAN.md`
│     128 pilot + fresh train640×N1 + val128×N4 + test128×N8
│     one expensive acquisition → one offline baseline matrix → one fresh test
│     primary: L18 PCA64 semantic + cross-layer dynamics → potential outcomes
│     metrics: Spearman, RMSE, regret, Recovery@25/50/75, quality-cost frontier
│     ✅ fresh train/val capture PASS (2304 rows); test SEALED; B0–B6 next
│     ❌ Validation Unlock FAIL: B3 Recovery@50=.380, F1@50=.423
│
├── ❌ DSSR Safe-Skip routing phase → `docs/SK_CUR_PLAN.md`
│     one greedy tool-free short probe → self-knowledge → SkipRegret
│     Train: reuse Search N1 + Probe N1; new Val2: Probe1 + Search4
│     SELF_KNOWLEDGE_ROUTER_FAIL; Test sealed
│
├── 🟡 Phase25 Step-Level Adaptive Retrieval → `docs/STEP_LEVEL_ADAPTIVE_RETRIEVAL_PLAN.md`
│     move decision from root to explicit reasoning checkpoints
│     ✅ S0 Train32 contract + Train8 exact replay PASS
│     NEXT: Train640 counterfactuals → fresh Val3 → sealed Test once
│
├── ⬜ Candidate ECA PASS → freeze
├── ⬜ 3E Full-Corpus (passage BM25 + rerank)
│     └── gold-evidence / evidence-use audit → CIPO only if bottleneck
│
└── ⬜ Phase 4
      GRPO vs REINFORCE on final frozen Candidate-ECA reward
      Root Branching only if Exact+Boundary still lacks counterfactuals
      optional CIGPO ablation (read first; implement later)
```

### 3C-GEN pass / fail (pre-registered)

**PASS** (held-out val-200 Agent):

- Answer: 3C ≥ SFT-v1 Agent **and** ≥ 3B  
- Evidence F1: 3C ≫ 3B  
- Search: recovers from 3B no-search (search_rate / count ↑ vs 3B)  
- Prefer paired Wrong→Right > Right→Wrong  

→ 3D0/3D1 may start on smoke/small pool; enlarge train for formal later.

**FAIL** (train128 high, val≈SFT):

- Mechanism still credited on-train; **enlarge train (~1k–2k) becomes hard prerequisite** before complex 3D2.  
- Do **not** overturn 3C closeout.

### 3D2 trigger (pre-registered; option B)

After 3D1 λ∈{0, 0.05, 0.10, 0.20, 0.30}, require a Pareto point with:

1. Search cost ↓ substantively (e.g. ≥20% relative)  
2. Answer/Evidence not collapsed (Answer loss ≲2–3pp)  
3. Stratified: high-capability / Direct✓ → less search  
4. Search-required → search stays high  

If no such point (only global bias knob) → **trigger 3D2**.

### 3D2 capability protocol (when triggered)

- **Not** every-step online; **periodic refresh** (e.g. every 25/50 steps), freeze labels in window  
- tool-free n=4, temp aligned with policy; success = **normalized EM**  
- \(p_{int}\in\{0,.25,.5,.75,1\}\)  
- Phase2 Direct labels = **audit/reference only** (3D0 sanity), not final 3D2 reward labels  
- Evidence gated by \((1-p_{int})\); cost gated by \(p_{int}\) — do not only raise λ  

### CIPO / CIGPO / REINFORCE / retrieval extras

| Item | Role |
|------|------|
| CIPO evidence-use | After Full-Corpus **audit** shows gold SF brittle or unused evidence |
| CIGPO info-gain | Phase4 **candidate** ablation; read now, do not implement in 3D |
| REINFORCE vs GRPO | Phase4; **same** final Candidate-ECA reward (3D1 or 3D2 if that won) |
| AgentIR / ReSeek / SAPO | Condition on measured bottleneck / ISDR |
| AdaCoM / memory | **Out** (horizon too short) |
| Multimodal 5M | After text mainline ([§B](#b-multimodal-branch-phase-5m-deferred)) |

---

## B. Multimodal branch (Phase 5M — deferred)

Unchanged: M1 text-actor + frozen VLM tools → M2 Qwen-VL. **Not in 3D.**

---

## C. Reading list (docs only; PDF not blocking GEN)

```text
11 AutoSearch
12 How to Train Your Deep Research Agent / Search-R1++
13 CIGPO          (Hotpot + Qwen2.5-3B + GRPO zero-advantage — must read)
14 CIPO
15 Revisiting Text Ranking in Deep Research
16 AgentIR

conditional: 17 ReSeek  18 SAPO
```

Priority for *understanding* current failures: **AutoSearch → CIGPO → Search-R1++ → CIPO**.

---

## D. Explicit non-goals (now)

- Grind 3C to 500/1000  
- Write 3D2 / CIPO code before GEN (+ 3D1 gate)  
- Resume ablation 3D from 3C@400  
- Claim smoke128 answer≈0.61 as val EM  
- Open-web or multimodal as 3D blockers  
