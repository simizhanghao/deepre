# Project Map — Evidence-Cost-Aware Deep Research Agent

> 状态：SGLang parity **FAIL** → **Dual-arm / fix rollout mismatch**。见 [NEXT_STEPS.md](NEXT_STEPS.md) · [ROADMAP.md](ROADMAP.md)

## 路线

```text
SFT-v1 → Answer-only → Evidence → Uniform Cost → Boundary
  → HF Routing smoke (OK) → SGLang parity 32×4 (FAIL)
  → Dual-arm / fix HF↔train loop mismatch → …
```

## Pipeline

```text
Question → agent loop → Candidate-BM25 tool → reward (evidence/boundary)
        → eval / results/{run}/
```

## Directory Layout

| Path | Purpose |
|------|---------|
| `configs/sft/` | `sft_v1_lora.yaml` · `sft_v1_merge.yaml` · `dataset_info_sft_v1.json` |
| `configs/rl/` | `grpo_smoke128.yaml` · tool / agent_loop |
| `data/sft/` | `coldstart_v1.jsonl` + llamafactory ShareGPT |
| `data/rl/train_smoke_128/` | GRPO parquet + BM25 index |
| `data/rl/calib_cost_lambda_512/` | offline λ calib |
| `data/eval/` | hotpotqa id lists |
| `src/rl/` | `rewards_evidence.py` · `rewards_boundary.py` · agent loop · tools |
| `src/sft/` | coldstart builders |
| `src/eval/` | metrics · protocol · trace |
| `scripts/` | train / eval / boundary / parity entrypoints |
| `results/` | numbered audits `01_`…`16_` |
| `outputs/` | `00_sft_v1_merged` · `rl/01_`…`06_` |
| `logs/` | numbered train/eval logs |
| `docs/` | ROADMAP · RESULTS_BOARD · NEXT_STEPS · PROJECT_MAP |
| `external/` | **read-only** third-party refs |
| `papers/` | reading notes |

## Key scripts

| Script | Role |
|--------|------|
| `scripts/run_grpo_evidence.sh` | Evidence GRPO |
| `scripts/run_grpo_boundary.sh` | Boundary GRPO (default 8 GPU) |
| `scripts/tmux_grpo_boundary.sh` | detach-safe Boundary launch |
| `scripts/build_search_boundary_table.py` | boundary table bootstrap |
| `scripts/audit_routing_exploration.py` | HF Routing Exploration Audit |
| `scripts/audit_routing_parity_sglang.py` | veRL/SGLang training-parity audit |
| `scripts/recreate_eca_verl_gpus.sh` | recreate `eca-verl` with `--gpus all` |
| `scripts/run_eval_val200_gen.sh` | val-200 GEN compare |
| `scripts/launch_grpo_main.py` | veRL entry + metrics patch |

## Data Flow

1. HotpotQA → `data/sft/coldstart_v1.jsonl` → SFT → `outputs/00_sft_v1_merged`
2. `train_smoke_128` → Evidence / Boundary GRPO → `outputs/rl/03_*` / `06_*`
3. Eval → `results/10_eval_grpo_evidence_val200/`
4. Parity → `results/16_audit_routing_exploration/parity_sglang_32x4/`
