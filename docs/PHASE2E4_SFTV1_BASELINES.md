# Phase 2E4 — SFT-v1 answer baselines (frozen val-200)

> **Status: closed.** Answer baselines + protocol trio done. Phase 2 frozen → see `docs/PHASE2_CLOSED.md`.  
> Model: `outputs/sft_qwen25_3b_coldstart_v1_merged` · n=200 HotpotQA distractor validation (frozen).

## Capability table (Exact Match)

| Setting | Base | SFT-v0 | **SFT-v1** | Δ v1−v0 | Δ v1−Base |
|---------|-----:|-------:|----------:|--------:|----------:|
| Direct | 0.180 | 0.170 | **0.175** | +0.005 | −0.005 |
| Candidate-BM25 | 0.435 | 0.470 | **0.485** | **+0.015** | **+0.050** |
| Oracle | 0.595 | 0.650 | **0.660** | **+0.010** | **+0.065** |

Token F1 (v1): Direct 0.248 / Candidate 0.599 / Oracle 0.772.  
Format valid rate: **1.0** on all three.

## Artifacts

| Run | Path |
|-----|------|
| Direct | `results/baseline_direct_n200_20260808_152524_phase2e4_sftv1_n200/` |
| Oracle | `results/baseline_oracle_n200_20260808_152523_phase2e4_sftv1_n200/` |
| Candidate | `results/baseline_candidate_bm25_n200_20260808_152524_phase2e4_sftv1_n200/` |
| LoRA | `outputs/sft_qwen25_3b_lora_coldstart_v1` |
| Merged | `outputs/sft_qwen25_3b_coldstart_v1_merged` |
| Train data | `data/sft/coldstart_v1.jsonl` (4550) → ShareGPT `eca_coldstart_v1_*` |

## Training note (2E3)

- From **Base** with identical LoRA hparams as v0 (r32 / α64 / lr 1e-4 / 2 ep / cutoff 4096 / 4×GPU).
- Train loss 0.082 · eval loss 0.042 (dev ShareGPT; not val-200 EM).
- Docker: `lf-sft` (`pytorch:2.5.1-cu124` → commit `lf-sft:ready` recommended).

## Interpretation

1. **v1 > v0 on all three answer settings** — gains are small but consistent; largest absolute lift vs Base remains Oracle (+6.5pp) and Candidate (+5.0pp).
2. **Candidate +1.5pp vs v0** matches the BM25 hard-neg evidence mix (P1); noise robustness moved slightly.
3. **Oracle +1.0pp vs v0** is consistent with Kimi grounded rationale on hard multi-hop (P2), but not a large jump — expected: only ~400/4550 rows are teacher reasoning.
4. **Direct flat (~0.175)** — routing calibration (P0) is **not** measurable from Direct/Oracle/Candidate alone; need protocol `routing` eval (internal% / search%).
5. Do **not** treat these EM deltas as GRPO-ready proof; next gate is protocol evidence F1 + routing mix vs v0 (internal 29% / search 71%).

## Protocol trio (phase2e4c) — done

| Metric | v0 | **v1** |
|--------|---:|------:|
| Evid F1 Oracle | 0.818 | **0.835** |
| Evid F1 Candidate | 0.665 | **0.725** |
| internal / search | 29%/71% | **12%/88%** |
| protocol valid routing | 1.0 | **1.0** |

Artifacts: `results/protocol_*_n200_20260808_153215_phase2e4c/`.

## Next

Phase 3A Search Agent rollout — `docs/PHASE3A_ROLLOUT.md` (no more SFT).
