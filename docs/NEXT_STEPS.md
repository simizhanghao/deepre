# Next Steps — Text ECA (v2)

> [ROADMAP.md](ROADMAP.md) · [RESULTS_BOARD.md](RESULTS_BOARD.md)

## Done

- [x] Evidence GRPO CLOSED @400 · GEN PASS (dev-200)
- [x] Offline cost λ → λ_s=0.40
- [x] Uniform cost FAIL → no stable Pareto → trigger Boundary
- [x] Capability-cost window-1 @50 CLOSED (routing FAIL; HOLD 400)
- [x] Artifact cleanup: numbered `results/` / `outputs/rl/` / English reward modules

## NOW — Boundary-aware Stage-II

1. Boundary table @ **Evidence@400 HF** (`outputs/rl/04_table_search_boundary`)  
2. Stage-II GRPO from Evidence@400 + `src/rl/rewards_boundary.py`  
3. Gate on \(\Delta_{\mathrm{boundary}}\), NoSearch↓ / NeedSearch↑ search rates  
4. Entry: `scripts/run_grpo_boundary.sh` / `scripts/tmux_grpo_boundary.sh`

Do **not**: blind 400 · Uniform λ 微扫 · CIGPO/CIPO · REINFORCE (unless boundary OK but mixed-action groups still rare).

## Later

Full-Corpus · Phase4 · multimodal

## Naming

`hotpotqa_200` = **dev-200** (selection).  
Train pool: `data/rl/train_smoke_128`.  
SFT: `outputs/00_sft_v1_merged`.
