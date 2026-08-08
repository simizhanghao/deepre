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
2026-08-08 | phase2e2-teacher-concurrent | add `--concurrency` + live progress flush | `scripts/generate_teacher_reasoning.py` | pass | default workers via env `KIMI_CONCURRENCY`
2026-08-08 | phase2e2-teacher-smoke20-c16 | `generate_teacher_reasoning.py --max-samples 20 --concurrency 16` | `results/teacher_reasoning_n20_20260808_113323_smoke20_c16/` | fail | API OK but 0/20 accept; missing `<think>` wrapper (meta-planning raw text)
2026-08-08 | phase2e2-teacher-io-v2 | rewrite Teacher contract: JSON rationale + code `<think>` wrap; modes A/B/C; API metadata | `src/sft/teacher_reasoning.py`, `scripts/generate_teacher_reasoning.py` | pass | stop asking LLM for XML; thinking disabled by default
2026-08-08 | phase2e2-teacher-smoke-abc | `generate_teacher_reasoning.py --mode abc --max-samples 5` | `results/teacher_reasoning_n5_*_smoke_abc_mode{A,B,C}/` | pass | A/B/C all parse=1.0 accept=0.8; choose mode A
2026-08-08 | phase2e2-teacher550 | `generate_teacher_reasoning.py --mode A --n-persistent 440 --n-other 110` | `results/teacher_reasoning_n550_20260808_115020_teacher550_modeA/` | pass | parse=1.0 accept=513/550; keep 400 in mix
2026-08-08 | phase2e2-build-v1 | `build_sft_coldstart_v1.py` | `data/sft/coldstart_v1.jsonl` + `results/phase2e2_coldstart_v1_20260808/` | pass | 4550 rows; kimi=400; val overlap 0
2026-08-08 | phase2e3-export | `export_coldstart_sharegpt.py --prefix eca_coldstart_v1` | `data/sft/llamafactory/eca_coldstart_v1_*.jsonl` | pass | train/dev/smoke 4322/228/80
2026-08-08 | phase2e3-sft | LF docker LoRA train + merge | `outputs/sft_qwen25_3b_lora_coldstart_v1` → `..._v1_merged` | pass | train_loss 0.082 eval_loss 0.042; from Base
2026-08-08 | phase2e4-baselines | `run_baseline.py` Direct/Oracle/Candidate v1 | `results/baseline_*_phase2e4_sftv1_n200/` | pass | EM 0.175 / 0.660 / 0.485 (vs v0 0.170/0.650/0.470)
2026-08-08 | phase2e4c-protocol | `run_protocol_eval.py` evidence_oracle/candidate + routing v1 | `results/protocol_*_20260808_153215_phase2e4c/` | pass | EvidF1 0.835/0.725; route 12%/88%; freeze SFT-v1
2026-08-08 | phase2-closed | freeze SFT-v1 as RL init | `docs/PHASE2_CLOSED.md` | pass | no SFT-v2; handoff Phase 3A
2026-08-08 | phase3a-scaffold | Search Agent rollout loop + Candidate-BM25 tool + smoke CLI | `src/agents/react_loop.py`, `src/tools/candidate_bm25.py`, `scripts/run_agent_rollout_smoke.py` | pass | max_search_turns=2; no train
2026-08-08 | phase3a-n8 | `run_agent_rollout_smoke.py --max-samples 8` | `results/agent_rollout_n8_20260808_163047_phase3a_n8/` | pass | finish=1.0 search_count=1.0 obs_mask=1.0
2026-08-08 | phase3a-n32 | `run_agent_rollout_smoke.py --max-samples 32` | `results/agent_rollout_n32_20260808_163356_phase3a_n32/` | pass | finish=0.969; 3A closed; handoff 3B GRPO plan

