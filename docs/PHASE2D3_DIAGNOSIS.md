# Phase 2D3 — Base vs SFT Diagnosis (frozen val-200)

> **Status: closed.** Answer baselines (2D3-B), taxonomy/paired audit (2D3-D), and protocol evals (2D3-C: evidence_oracle / evidence_candidate / routing) are filled.  
> Model: `outputs/sft_qwen25_3b_coldstart_v0_merged` · git `b5a9f8a` · n=200 HotpotQA distractor validation (frozen).

## Capability table

| Capability | Base → SFT / note |
| --------------------- | -------------: |
| Oracle Answer EM | 0.595 → **0.650** (Δ +0.055) |
| Candidate Answer EM | 0.435 → **0.470** (Δ +0.035) |
| Direct Answer EM | 0.180 → 0.170 (Δ −0.010) |
| Evidence F1 Oracle | **0.818** (P 0.845 / R 0.820) |
| Evidence F1 Candidate | **0.665** (P 0.690 / R 0.664) |
| Protocol valid (evidence) | Oracle **0.905** / Candidate **0.890** |
| Protocol valid (routing) | **1.000** |
| Internal / Search rate | **0.29 / 0.71** (none 0.0) |
| Routing mean EM (routing-only) | **0.09** — *not* agent final EM; search branch stops without answer |
| Conditional internal EM (proxy) | ≈ **0.31** on the 58 internal-routed items |
| C taxonomy | **38.0% → 33.0%** (Δ −5.0 pp) |
| Oracle W→R / R→W | **24 / 13** (net +11) |
| Candidate W→R / R→W | **25 / 18** (net +7) |

### Routing conditionals (protocol routing)

| Slice | n | internal | search |
|-------|--:|--------:|-------:|
| Base Direct correct | 36 | **50%** | 50% |
| Base Direct wrong ∧ Oracle correct | 88 | 26.1% | **73.9%** |

Proxy routing view (Direct-correct → should internal; else → should search):

|  | Pred internal | Pred search |
|--|-------------:|------------:|
| Direct correct | 18 | 18 |
| Direct wrong | 40 | 124 |

- Routing proxy accuracy ≈ **(18+124)/200 = 71%**
- Internal precision / recall ≈ **31% / 50%**
- Search precision / recall ≈ **87% / 76%**

## Taxonomy (EM-based)

| Label | Base | SFT | Δ |
|------:|-----:|----:|--:|
| A | 28.5% | 31.5% | +3.0 |
| B | 15.5% | 18.5% | +3.0 |
| **C** | **38.0%** | **33.0%** | **−5.0** |
| D | 10.0% | 11.5% | +1.5 |
| E | 7.5% | 4.0% | −3.5 |
| O | 0.5% | 1.5% | +1.0 |

Base-C (n=76) under SFT labels: stay C=55, →B=12, →A=7, →D=2.

## Paired significance (n=200)

| Method | ΔEM | McNemar p | Bootstrap 95% CI |
|--------|----:|----------:|------------------|
| Oracle | +0.055 | 0.099 | [−0.005, +0.115] |
| Candidate | +0.035 | 0.360 | [−0.030, +0.095] |
| Direct | −0.010 | 0.774 | [−0.045, +0.025] |

Oracle improvement is directionally clear but **not** yet p&lt;0.05 under exact McNemar; treat as promising, not definitive.

## Artifacts (local; `results/` gitignored)

| Stage | Path |
|-------|------|
| QA baselines (SFT) | `results/baseline_*_phase2d3_sft_n200/` |
| CPU audit (2D3-D) | `results/audit_base_vs_sft_n200_20260807_181820_phase2d3d/` |
| Evidence oracle (2D3-C) | `results/protocol_evidence_oracle_n200_20260807_182426_phase2d3c/` |
| Evidence candidate (2D3-C) | `results/protocol_evidence_candidate_n200_20260807_182427_phase2d3c/` |
| Routing (2D3-C) | `results/protocol_routing_n200_20260807_182427_phase2d3c/` |
| Logs | `logs/protocol_phase2d3c/` |

## Reading (final for v0)

1. **Protocol ✅** — routing valid 100%; evidence valid ~90%. Cold-start action space is stable; no routing collapse (not 100% search / 100% internal).
2. **Evidence mostly ✅, noise-sensitive** — Oracle Evid F1 0.82 vs Candidate 0.67 → model *can* extract evidence on clean docs; BM25 hard negatives still hurt (~15 F1). Not situation B (broken extractor).
3. **Routing preliminary ✅, calibration ❌** — search rises from 50% (Direct-correct) to 73.9% (Direct-wrong∧Oracle-correct). But over-search on easy items (only 50% internal) and false-internal on hard items (26.1%). Internal precision ~31%.
4. **Routing EM=0.09 must not be misread** — routing-only eval stops after `<search>`; 71% of items never emit `<answer>` under this scorer. Compare answer capability via Direct/Oracle/Candidate tables instead.
5. **C shrinks but is unsolved** — 38%→33%; 55/76 original C remain. Together with strong Oracle evidence but remaining C, multi-hop reasoning (`template_v0`) is still a gap.
6. **Paired transitions** — Oracle net +11 (24 W→R vs 13 R→W): mostly new capability, some instability.

**One-line verdict:** Cold-start SFT v0 is a **qualified RL/SFT starting policy** (stable non-collapsed protocol + partial competence sensing + usable evidence extractor), **not** a mature routing or evidence-reasoning policy.

## Coldstart v1 priorities (do not retune LoRA / do not enter GRPO yet)

| Priority | Fix | Why |
|----------|-----|-----|
| **P0 Routing calibration** | More Direct✅→internal positives; false-confidence hard negatives (Direct❌∧Oracle✅ but model would internal); slightly raise internal mix above ~15% | Internal P≈31%, over-search + false-internal |
| **P1 BM25 hard-neg evidence** | Replace/augment random distractors with BM25 Top-K hard negatives in evidence trajectories | Oracle Evid F1 0.82 → Candidate 0.67 |
| **P2 Grounded reasoning (subset)** | Teacher rationale only on remaining-C / hard multi-hop slices; keep most of 3k | C still 33%; template_v0 weak |

Stay out of Phase 3 GRPO until P0–P1 (and preferably a targeted P2 smoke) land — otherwise search→bad evidence→wrong answer → sparse / zero-std groups.

## Reproduce

```bash
# 2D3-D (CPU)
python scripts/audit_base_vs_sft.py \
  --base-direct results/baseline_direct_n200_20260807_154900_phase1_final_n200/metrics.json \
  --base-oracle results/baseline_oracle_n200_20260807_154912_phase1_final_n200/metrics.json \
  --base-bm25 results/baseline_candidate_bm25_n200_20260807_154918_phase1_final_n200/metrics.json \
  --sft-direct results/baseline_direct_n200_20260807_180530_phase2d3_sft_n200/metrics.json \
  --sft-oracle results/baseline_oracle_n200_20260807_180636_phase2d3_sft_n200/metrics.json \
  --sft-bm25 results/baseline_candidate_bm25_n200_20260807_180727_phase2d3_sft_n200/metrics.json

# 2D3-C (GPU) — or use scripts/run_protocol_eval_tmux.sh
MERGED=outputs/sft_qwen25_3b_coldstart_v0_merged
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/run_protocol_eval.py --mode evidence_oracle \
  --model-path $MERGED --max-samples 200 --run-tag phase2d3c
CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/run_protocol_eval.py --mode evidence_candidate \
  --model-path $MERGED --max-samples 200 --run-tag phase2d3c
CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/run_protocol_eval.py --mode routing \
  --model-path $MERGED --max-samples 200 --run-tag phase2d3c
```
