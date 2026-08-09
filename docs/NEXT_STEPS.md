# Next Steps — Text ECA (v2)

> [ROADMAP.md](ROADMAP.md) · [RESULTS_BOARD.md](RESULTS_BOARD.md)  
> **NOW = 3C-GEN only.** No 3D0/3D2 code until GEN numbers exist.

## Done

- [x] SFT-v1 · 3A · 3B@100 · 3C@400 closed

## 1) 3C-GEN ⬜ ← execute now

```bash
# see docs/PHASE3C_GEN.md and scripts/run_phase3c_gen.sh
```

Checklist:

- [ ] Merge FSDP → HF: 3B@100, 3C@400  
- [ ] Agent val-200: SFT-v1 | 3B@100 | 3C@400 (Candidate-BM25, max_search=2, T=0)  
- [ ] Metrics: EM/F1, Evid F1, search_rate, search_count + P(0/1/2), finish, obs tokens  
- [ ] Apply PASS/FAIL gates in ROADMAP  
- [ ] Write `results/phase3c_gen_val200_*/` + freeze note in PHASE3C_GEN.md  

**Do not** start 3D0 λ sweep until this table exists.

## 2) After GEN

- PASS → 3D0 → 3D1 Uniform Cost → Pareto gate → (maybe) 3D2  
- FAIL → enlarge train ~1k–2k before heavy Cost / 3D2  

## 3) Later

3E Full-Corpus · CIPO if audit says so · Phase4 matched-step + GRPO vs REINFORCE · 5M multimodal  

## Do not

- Continue 3C training  
- Implement 3D2/CIPO/REINFORCE this week  
- Swap to Qwen-VL  
