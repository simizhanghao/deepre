# Phase 3C-GEN — Held-out Agent generalization gate

> Status: **PASS** (2026-08-09)  
> Artifacts: `results/phase3c_gen_val200_20260809_120709/`  
> Purpose: smoke128 train-window metrics ≠ val EM. Measure transfer on frozen val-200.

## Protocol (locked)

| Knob | Value |
|------|--------|
| Eval | `data/eval/hotpotqa_200.jsonl` |
| Loop | Agentic Candidate-BM25 (`run_agent_rollout_smoke.py`) |
| max_search_turns / top_k / T | 2 / 5 / 0.0 |
| Models | SFT-v1 · 3B HF@100 · 3C HF@400 |

## Results (n=200)

| Model | Answer EM | Token F1 | Evid F1 | search_rate | search_count | P0 / P1 / P2 | finish |
|-------|----------:|---------:|--------:|------------:|-------------:|-------------:|-------:|
| SFT-v1 | 0.475 | 0.609 | 0.566 | 0.88 | 0.88 | 0.12 / 0.88 / 0.00 | 0.955 |
| 3B@100 | 0.190 | 0.249 | 0.251 | **0.09** | 0.09 | **0.91** / 0.09 / 0.00 | 0.900 |
| **3C@400** | **0.540** | **0.666** | **0.667** | **1.00** | **1.00** | 0.00 / **1.00** / 0.00 | **1.000** |

HF merges: `outputs/rl/hf_merged/grpo_sftv1_{smoke_step100,evidence_3c_step400}`.

## Gate verdict: **PASS**

| Criterion | Result |
|-----------|--------|
| 3C Answer ≥ SFT and ≥ 3B | **0.540 ≥ 0.475 ≥ 0.190** |
| 3C Evid ≫ 3B | **0.667 ≫ 0.251** |
| Search recovers from 3B no-search | **1.00 vs 0.09** (P0: 0 vs 0.91) |

**Interpretation:** Evidence GRPO transfers to held-out val-200 — not only smoke128 memorization. Gap train-window (~0.61) → val EM (0.54) is expected. Always-search (P1=1.0) also transfers → Cost (3D) still needed.

**Next (per ROADMAP v2):** 3D0 offline λ sweep → 3D1 Uniform Cost (still may use small train for mechanism; enlarge later for formal). Do **not** grind more 3C steps.
