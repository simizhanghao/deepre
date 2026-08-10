# Roadmap — Evidence-Cost-Aware Deep Research Agent (v2)

> Frozen **2026-08-09 (v2)**. Problem-driven tech only.  
> Board: [RESULTS_BOARD.md](RESULTS_BOARD.md) · Actions: [NEXT_STEPS.md](NEXT_STEPS.md)

## Principle

```text
Do NOT insert multimodal into Phase 3D.
Do NOT add AutoSearch / CIPO / REINFORCE “because papers are new”.
Tech enters only via pre-registered failure gates.
```

Causal story:

```text
3B  sparse answer     → no-search
3C  + evidence        → always-search
3D1 uniform cost      → quality–cost baseline
3D2 capability cost   → ONLY if uniform cost fails routing gate
```

---

## A. Text ECA mainline

```text
NOW → 3C-GEN only
│
├── ✅ 3B CLOSED @100
├── ✅ 3C CLOSED @400
│
├── ⬜ 3C-GEN                    ← ONLY active work item
│     FSDP → HF merge
│     frozen val-200 Agent (Candidate-BM25, max_search=2)
│     SFT-v1 | 3B@100 | 3C@400
│     Gates below → decide small-pool 3D vs enlarge train first
│
├── ✅ 3D0   offline λ on calib-512 → λ_s=0.40 (strict 0.50)
│     NOTE: 0.05–0.30 cannot stop Evid farming on I
│
├── ❌ 3D1 Uniform Cost λ=0.40  FAIL @~250
│     search→0 after step5; KL→~0.58; no Pareto
│
├── ✅ 3D1b online λ phase diagram   CLOSED
│     no stable Pareto (0.05 always-search; ≥0.10 collapse)
│     → formally triggers 3D2
│
├── ✓ 3D2-v0 Capability-only Cost  @50 CLOSED (routing FAIL)
├── ◐ 3D2b Search-Boundary Stage-II  from 3C@400 (NEXT)
│     no Uniform extinction; Δ_route FAIL / late search≈1
│     HOLD segmented@400 until routing diagnosis
│     R = R_A + λ_e(1−p_int)R_E + λ_f R_F − λ_s p_int 1[N_s>0]
│
├── ⬜ 3E Full-Corpus (passage BM25 + rerank)
│     └── gold-evidence / evidence-use audit → CIPO only if bottleneck
│
└── ⬜ Phase 4
      larger train + matched-step
      GRPO vs REINFORCE on final frozen Candidate-ECA reward
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
