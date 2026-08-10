# Next Steps — Text ECA (v2)

> [ROADMAP.md](ROADMAP.md) · [RESULTS_BOARD.md](RESULTS_BOARD.md)

## Done

- [x] Evidence GRPO CLOSED @400 · GEN PASS (dev-200)
- [x] Offline cost λ → λ_s=0.40
- [x] Uniform cost FAIL → no stable Pareto → trigger Boundary
- [x] Capability-cost window-1 @50 CLOSED (routing FAIL; HOLD 400)
- [x] Artifact cleanup: numbered `results/` / `outputs/rl/` / English reward modules
- [x] Boundary Stage-II smoke routing FAIL (`Δ_boundary≈0`)
- [x] Routing Exploration Audit **smoke** → `NATURAL_EXPLORATION_OK`  
      (`results/16_audit_routing_exploration/`, T=1.3: P(internal|NoSearch)=0.50, mixed=0.375)

## NOW — Mixed-action GRPO (design → tiny train)

Exploration is **not** saturated: NoSearch still yields internal at T=0.9/1.3.  
Next makes GRPO groups contain search↔internal counterfactuals (still Evidence@400 init + `rewards_boundary`).

1. Prefer extending `scripts/run_grpo_boundary.sh` / rollout path — **no new reward file**  
2. Acceptance on **tools-enabled** eval: `Δ_boundary`↑, `sr_no`↓, `sr_need` stays high  
3. Optional: enlarge Routing Exploration beyond smoke (32×4) before locking train hyperparams  
4. REINFORCE only if mixed-action groups exist + preference OK + GRPO still Δ≈0

Do **not**: Dual-arm first · blind 400 · Uniform λ · CIGPO/CIPO · new `phase*` names.

## Later

Full-Corpus · Phase4 · multimodal

## Naming

`hotpotqa_200` = **dev-200** (selection).  
Train pool: `data/rl/train_smoke_128`.  
SFT: `outputs/00_sft_v1_merged`.
