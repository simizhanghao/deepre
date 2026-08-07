# Run Log

One line per smoke/full run. Format:

```
YYYY-MM-DD | stage | command | output_dir | pass/fail | notes
```

---

2026-08-07 | phase2d3-b | `run_baseline.py` Direct/Oracle/Candidate on merged coldstart_v0 | `results/baseline_*_phase2d3_sft_n200/` | pass | EM 0.170 / 0.650 / 0.470
2026-08-07 | phase2d3-d | `audit_base_vs_sft.py` taxonomy + paired | `results/audit_base_vs_sft_n200_20260807_181820_phase2d3d/` | pass | C 38%→33%; Oracle W→R/R→W 24/13
2026-08-07 | phase2d3-c | `run_protocol_eval.py` evidence_oracle/candidate + routing | `results/protocol_*_20260807_18242*_phase2d3c/` | pass | Evid F1 0.818/0.665; routing internal 29%/search 71%; protocol valid 1.0
2026-08-07 | phase2d3-docs | freeze `docs/PHASE2D3_DIAGNOSIS.md` | — | pass | v1 priorities: routing cal > BM25 hard-neg evid > subset teacher rationale
2026-08-07 | phase2e1-direct | `label_direct_train.py` n8000 4-shard Base Direct | `results/phase2e1_direct_label_n8000_20260807_202826_phase2e1/` | pass | direct_correct 1500/8000 (18.8%)
2026-08-07 | phase2e1-base-oracle | `run_baseline.py` oracle on train shards | `results/phase2e1_base_oracle_n8000_20260807_205154/merged/` | pass | EM 0.6812 (5445/7993); search-req pool ~4151
2026-08-07 | phase2e1-sftv0-oracle | `run_baseline.py` oracle w/ coldstart_v0_merged | `results/phase2e1_sftv0_oracle_n8000_20260807_211627/merged/` | pass | EM 0.7375 (+5.6pp vs Base); hard pool 2098
2026-08-07 | phase2e1-docs | freeze `docs/PHASE2E1_LABELING.md` | — | pass | 2E1 GPU labeling closed; next 2E2 coldstart_v1 builder
2026-08-07 | phase2e2-scaffold | add teacher + coldstart_v1 builder scripts | `src/sft/teacher_reasoning.py`, `coldstart_v1_builder.py`, `scripts/generate_teacher_reasoning.py`, `build_sft_coldstart_v1.py` | pass | Kimi endpoint for think-only P2
2026-08-07 | phase2e2-teacher-smoke20 | `generate_teacher_reasoning.py --max-samples 20` | `results/teacher_reasoning_n20_20260807_213753_smoke20/` | fail | 0/20; Kimi `10.16.137.2:8000` timeout/500 then connection refused — retry when service up

