# Phase 2 CLOSED — Freeze SFT-v1 as RL initialization

> **Status: closed 2026-08-08.** Do not start SFT-v2. Next: Phase 3A Search Agent rollout.

## Frozen checkpoint

```text
outputs/sft_qwen25_3b_coldstart_v1_merged
```

Git (docs/data/configs): `f419d2f`+ · Policy from Base + coldstart_v1 (4550) LoRA.

## Why freeze now

SFT finished its job: protocol-stable cold-start with basic evidence + initial routing.  
v0→v1 answer gains are small (+1.0~1.5pp); Candidate Evidence F1 +6pp.  
Routing still over-searches (88%) — that is a **policy tradeoff for cost-aware RL**, not an SFT ticket.

## Gate results (val-200)

### Answer baselines

| Setting | Base | v0 | **v1** |
|---------|-----:|---:|------:|
| Direct EM | 0.180 | 0.170 | **0.175** |
| Candidate EM | 0.435 | 0.470 | **0.485** |
| Oracle EM | 0.595 | 0.650 | **0.660** |

### Protocol

| Metric | v0 | **v1** |
|--------|---:|------:|
| Evid F1 Oracle | 0.818 | **0.835** |
| Evid F1 Candidate | 0.665 | **0.725** |
| protocol valid (oracle/cand/routing) | 0.905 / 0.890 / 1.0 | **0.920 / 0.895 / 1.0** |
| internal / search | 29% / 71% | **12% / 88%** |
| Direct✓ → internal | 50% | 33% |
| D✗∧O✓ → search | 73.9% | **95.5%** |
| internal precision | 31% | **50%** |

Artifacts: `results/baseline_*_phase2e4_sftv1_n200/`, `results/protocol_*_phase2e4c/`.  
Details: `docs/PHASE2E4_SFTV1_BASELINES.md`.

## Explicit non-goals (stop list)

- SFT-v2 / more epochs / LoRA retune / more Teacher data for routing
- Optimizing Direct EM via SFT
- Treating routing-only EM as Agent EM

## Handoff to Phase 3

```text
Phase 3A  Search Agent rollout (Candidate-BM25, max_search_turns=2) — no train
Phase 3B  Search-R1 answer-only GRPO baseline
Phase 3C  + Evidence Reward
Phase 3D  + Cost / duplicate penalty
Phase 3E  Internal/external routing via RL
```

See `docs/PHASE3A_ROLLOUT.md`.
