# Next Steps — Text ECA (v2)

> [ROADMAP.md](ROADMAP.md) · [RESULTS_BOARD.md](RESULTS_BOARD.md)

## Done

- [x] Evidence GRPO CLOSED @400 · GEN PASS (dev-200)
- [x] Offline cost λ → λ_s=0.40
- [x] Uniform cost FAIL → no stable Pareto → trigger Boundary
- [x] Capability-cost window-1 @50 CLOSED (routing FAIL; HOLD 400)
- [x] Artifact cleanup: numbered `results/` / `outputs/rl/` / English reward modules
- [x] Boundary Stage-II smoke routing FAIL (`Δ_boundary≈0`)
- [x] Routing Exploration HF smoke → `NATURAL_EXPLORATION_OK`
- [x] Training-parity SGLang 32×4 @ T=0.9/1.3 → `TRAINING_PARITY_EXPLORATION_FAIL`

## NOW — Dual-arm / fix rollout mismatch (not Mixed-action)

Real `EcaSearchAgentLoop` (veRL+SGLang) shows **p_internal=0**, **mixed=0** on both T=0.9 and T=1.3.  
HF smoke exploration does **not** transfer to the training worker.

1. Prefer **dual-arm / forced search↔internal** counterfactuals in the real agent loop  
   OR diagnose HF `react_loop` vs `EcaSearchAgentLoop` protocol gap  
2. Acceptance later: tools-enabled eval `Δ_boundary`↑, `sr_no`↓, `sr_need` high  
3. Mixed-action GRPO only if parity gate flips to OK (or dual-arm proves groups)  
4. REINFORCE only after mixed groups exist + preference OK + GRPO still Δ≈0

Do **not**: Mixed-action GRPO first · blind 400 · Uniform λ · CIGPO/CIPO · new `phase*` names.

## Later

Full-Corpus · Phase4 · multimodal

## Naming

`hotpotqa_200` = **dev-200** (selection).  
Train pool: `data/rl/train_smoke_128`.  
SFT: `outputs/00_sft_v1_merged`.  
Parity dumps: `results/16_audit_routing_exploration/parity_sglang_32x4/` (gitignored).
