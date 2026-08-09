# Next Steps — Text ECA mainline

> Updated **2026-08-09**. Full roadmap: [ROADMAP.md](ROADMAP.md) · Results: [RESULTS_BOARD.md](RESULTS_BOARD.md)  
> **Multimodal = Phase 5M only** — do not insert into 3D.

## Done

- [x] SFT-v1 freeze (`outputs/sft_qwen25_3b_coldstart_v1_merged`)
- [x] 3A agent rollout smoke
- [x] 3B Answer-only GRPO CLOSED @100 (no-search pathology)
- [x] 3C Evidence GRPO CLOSED @400 (always-search; mechanism OK)

## Now (in order)

### 1) 3C-GEN gate ⬜

Frozen val-200 **Agentic** Candidate-BM25 eval (not train-window metrics).

- [ ] FSDP `global_step_*` → HF / eval-loadable checkpoint for 3B@100 & 3C@400
- [ ] Same protocol: Candidate-BM25, `max_search_turns=2`, fixed decode
- [ ] Run: **SFT-v1 | 3B@100 | 3C@400**
- [ ] Log: Answer EM/F1, Evid F1, `search_rate`, **`search_count` + P(0/1/2)**, finish, obs tokens
- [ ] Write pass/fail vs pre-declared gates (3C ≥ SFT & ≥ 3B on answer; evid ≫ 3B; search > 3B)
- [ ] Doc: `docs/PHASE3C_GEN.md` + `results/phase3c_gen_val200_*/`

If val barely moves → expand train pool becomes a hard Phase-4 / long-3D prerequisite.  
If val clearly rises → 3D may stay on smoke/small pool for mechanism, then scale.

### 2) 3D0 — Cost prep ⬜

- [ ] Audit 3C trajectories: `search_count` mean / hist (expect ~1.0, not 2)
- [ ] Offline λ_search sweep (~0.05–0.40, center 0.1–0.3) on cached rollouts
- [ ] Ranking check: internal✓ / necessary search✓ / wasteful search↓
- [ ] Keep Duplicate secondary unless double-search appears
- [ ] Doc: `docs/PHASE3D0.md`

### 3) 3D1 — Cost-aware GRPO ⬜

- [ ] **Fresh from SFT-v1** (ablation line; do not resume 3C@400)
- [ ] \(R = R_A + 0.5 R_E + 0.1 R_F - \lambda_s C_{search}\)
- [ ] Metrics triad: Answer · Evidence · Search cost (+ zero_std)
- [ ] Success ≠ search↓ alone if quality collapses

### 4) After 3D mechanism ⬜

- [ ] 3D2 Pareto vs 3B/3C (+ GEN)
- [ ] 3E Full-Corpus Wikipedia (parallel index prep OK anytime)
- [ ] Phase 4: larger train / matched-step / frozen eval pack

### 5) Later only — Phase 5M multimodal ⬜

See [ROADMAP.md §B](ROADMAP.md). M1 frozen-VLM tools before Qwen-VL end-to-end RL.

## Do not do next

- Continue 3C training to 500/1000  
- Swap base model to Qwen-VL for 3D  
- Make open-web or multimodal a 3D blocker  
- Quote smoke128 0.61 as val EM
