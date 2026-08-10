# Next Steps — Text ECA (v2)

> [ROADMAP.md](ROADMAP.md) · [RESULTS_BOARD.md](RESULTS_BOARD.md)

## Done

- [x] 3C CLOSED @400 · **3C-GEN PASS** (dev-200)
- [x] **3D0** → λ_s=0.40 offline ([PHASE3D0.md](PHASE3D0.md))
- [x] **3D1 λ=0.40 FAIL** — search extinction ([PHASE3D1.md](PHASE3D1.md))
- [x] **3D1b CLOSED** — no stable Uniform Pareto → trigger 3D2 ([PHASE3D1B.md](PHASE3D1B.md))
- [x] **3D2-v0 window-1 @50 DONE** — SOFT_PASS stability / **FAIL routing**; **HOLD 400** ([PHASE3D2.md](PHASE3D2.md))
- [x] Papers: SAAS / Search-R1-algo / CIGPO / CIPO → `papers/`

## NOW — 3D2b Search-Boundary-Aware Stage-II ⬜

1. Close 3D2-v0 (no continue→400)  
2. Bootstrap boundary table @ **3C@400 HF**: 4×disabled + 4×enabled, δ=2  
3. Stage-II GRPO from 3C@400 + `rewards_3d2b` · **50 steps**  
4. Gate on \(\Delta_{\mathrm{boundary}}\), NoSearch↓ / NeedSearch↑ search rates  

Do **not**: blind 400 · Uniform λ 微扫 · CIGPO/CIPO · REINFORCE (unless boundary OK but mixed-action groups still rare).  
Docs: [PHASE3D2B.md](PHASE3D2B.md)

## Later

3E Full-Corpus · Phase4 · 5M multimodal

## Naming

`hotpotqa_200` = **dev-200** (selection).
