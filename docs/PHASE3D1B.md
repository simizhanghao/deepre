# Phase 3D1b — Online Uniform Cost λ phase diagram

## Status: **CLOSED → trigger 3D2**

## Purpose

Decide whether Uniform Cost has a stable mid-λ Pareto window online.

## Protocol (executed)

| Item | Value |
|------|--------|
| Init | fresh SFT-v1 |
| λ | 0.05 / 0.10 / 0.15 / 0.20 (+ prior 0.40) |
| Budget | ≤60 steps + early-stop |
| Throughput | BATCH=16, GPU_MEM_UTIL=0.60, MICRO_BATCH=2, n=4 |

## Phase diagram (online)

| λ | max_step | last10 search | last10 answer | stop | reading |
|--:|---------:|--------------:|--------------:|------|---------|
| 0.05 | 55 | 0.99 | 0.00 | always_search | too weak |
| 0.10 | 52 | 0.03 | 0.18 | search_collapse | collapse (slower) |
| 0.15 | 11 | 0.15→0 | 0.09 | search_collapse | fast collapse |
| 0.20 | 10 | 0.15→0 | 0.12 | search_collapse | fast collapse |
| 0.40 | ~5 (3D1) | 0 | ~0.25 late | FAIL | extinction |

Artifacts:

- `results/phase3d1b_online_lambda_20260809_160340/` (λ=0.05)
- `results/phase3d1b_online_lambda_20260809_171129/` (λ=0.10/0.15/0.20 + summary)
- merged: `results/phase3d1b_merged_phase_diagram.json`

## Verdict

> **NO_STABLE_UNIFORM_WINDOW.**  
> Search responds to uniform λ as a **nonlinear phase transition** (always-search ↔ never-search), not a smooth quality–cost tradeoff.  
> **Formally trigger Phase 3D2 Capability-Aware Cost.** Do **not** micro-tune λ∈{0.12,…} or long-train Uniform@400.

Also confirmed earlier: offline λ ranking (3D0) **cannot** replace online policy dynamics.

## Explicit non-followups

- resume λ=0.40 / Cost warmup / KL hacks / continue from 3C@400
