# Next Steps — Text ECA (v2)

> [ROADMAP.md](ROADMAP.md) · [RESULTS_BOARD.md](RESULTS_BOARD.md)

## Done

- [x] 3C CLOSED @400 · **3C-GEN PASS** (dev-200)
- [x] **3D0 offline λ sweep** → **λ_s=0.40** ([PHASE3D0.md](PHASE3D0.md))

## NOW — 3D1 Uniform Cost GRPO ⬜

```bash
STEPS=400 ECA_SEARCH_COST_WEIGHT=0.40 bash scripts/tmux_grpo_cost.sh
```

Details: [PHASE3D1.md](PHASE3D1.md)

- Fresh SFT-v1, smoke128, 400 steps, matched to 3C budget  
- **Do not** resume from 3C@400  
- After train: merge HF → Agent eval on **dev-200** → Pareto / routing gate  

## After 3D1

- PASS routing → skip 3D2 mainline  
- FAIL (global bias only) → 3D2 capability-aware  
- Then 3E Full-Corpus · Phase4 larger train + final held-out  

## Naming

`hotpotqa_200` = **dev-200** (selection). Phase4 needs a disjoint final test.
