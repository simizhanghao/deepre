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
2026-08-08 | phase3b-plan | lock veRL+SGLang docker GRPO decisions | `docs/PHASE3B_GRPO.md`, `docs/PHASE3B_SETUP.md` | pass | EM+0.1fmt; Candidate+sample_id; 4×GPU; no lf-sft reuse
2026-08-08 | phase3b0-scaffold | Candidate BaseTool + smoke128 parquet + mask audit + GRPO launcher | `src/rl/*`, `configs/rl/*`, `docs/PHASE3B0.md` | pass | synthetic mask PASS; train=128; no GRPO steps yet
2026-08-08 | phase3b1-smoke5 | `run_grpo_smoke.sh` STEPS=5 n=4 4GPU EM+0.1fmt | `outputs/rl/grpo_sftv1_smoke/global_step_5` | pass | step1 score≈0.225 loss≈0.118; step5 score≈0.191 loss≈0.389; ~4m43s; fixed min_global_steps + sgl055 stop_token_ids
2026-08-08 | phase3b1-ops | tmux resume launcher + tensorboard logger + docker smokeok tag | `scripts/tmux_grpo_smoke.sh`, `eca-verl:sgl055-smokeok-20260808` | pass | resume_mode=auto; TB under outputs/rl/tensorboard/
2026-08-08 | phase3b2-start | resume GRPO baseline step5→50 (n=4, EM+0.1fmt, frozen knobs) | `outputs/rl/grpo_sftv1_smoke` + TB | superseded | early legs hit step16 false-stop
2026-08-08 | phase3b2-resume15 | docker exec -d resume step15→50 after SIGTERM / total_epochs=1 false-stop | logs/grpo_sftv1_smoke_to50_resume15_* | fail→fixed | rootcause: (1) client-attached exec SIGTERM (2) total_epochs=1 ⇒ exit @16
2026-08-08 | phase3b2-step50 | STEPS=50 TOTAL_EPOCHS=50 resume15→50; hard audit | `outputs/rl/grpo_sftv1_smoke/global_step_50` + `results/phase3b2_grpo_sftv1_smoke_step50_20260808/` | pass | score 0.245→0.286 (late); kl_loss→0.012; format late OK; no NaN; split metrics missing; **conditional →100**
2026-08-08 | phase3b2-metrics-hook | file-patch TaskRunnerV1.run → apply() in Ray actor | `scripts/patch_verl_phase3b_metrics.py` | pass | fixes driver-only monkeypatch miss; no algo knobs touched
2026-08-08 | phase3b2-to100 | STEPS=100 resume50→100 formal baseline + diagnostics | `outputs/rl/grpo_sftv1_smoke` | running | hook live @61+; resume from ckpt60 after LinkedList fix; early signal search=0 zero_std↑ — watch to 100
2026-08-09 | phase3b2-step100 | hard audit close 3B | `global_step_100` + `results/phase3b2_grpo_sftv1_baseline_step100_20260809/` | pass→close | score 0.23→0.29; search=0 (61–100); zero_std≈0.77; format/finish≈1; **next 3C from SFT-v1**
2026-08-09 | phase3c0-scaffold | Evidence F1 reward + breakdown + SF parquet + metrics + launchers | `src/rl/rewards_3c.py`, `docs/PHASE3C.md` | pass | λ_e=0.5; Cost weights reserved
2026-08-09 | phase3c0-offline | `offline_reward_replay_3c.py` | `results/phase3c_offline_reward_replay/` | pass | perfect=1.0 none=0; sim zero_std=0; group_std≈0.21
2026-08-09 | phase3c1-to500 | STEPS=500 SAVE_FREQ=25 from SFT-v1 (not 3B ckpt) | `outputs/rl/grpo_sftv1_evidence_3c` | partial | disk full @~324; incomplete 325 dropped
2026-08-09 | phase3c-cleanup | drop non-final 0/1/2 LoRA + mid 3B/3C ckpts; SAVE_FREQ→50 | kept SFT merged + 3B@100 + 3C@300 | pass | /data1 ~73% free
2026-08-09 | phase3c-resume300 | STEPS=500 SAVE_FREQ=50 resume auto from 300 | same OUT_DIR | pass→stop@400 | user stop after global_step_400
2026-08-09 | phase3c-step400 | hard audit close 3C | `global_step_400` + `results/phase3c_grpo_sftv1_evidence_step400_20260809/` | pass→close | late answer≈0.61 evid≈0.62 search≈1 zero_std≈0.58; **next 3D Cost from SFT-v1**
2026-08-09 | docs-board | freeze RESULTS_BOARD + PHASE3C closeout | `docs/RESULTS_BOARD.md`, `docs/PHASE3C.md` | pass | push GitHub
2026-08-09 | docs-roadmap | freeze text ECA mainline + defer multimodal 5M | `docs/ROADMAP.md`, `docs/NEXT_STEPS.md` | pass | 3C-GEN→3D0→3D; MM after text
2026-08-09 | docs-roadmap-v2 | Pareto-gated 3D2 + capability gating + CIPO conditional | `docs/ROADMAP.md` v2 | pass | 3C-GEN only active work
2026-08-09 | phase3c-gen-export | FSDP→HF merge 3B@100 + 3C@400 | `outputs/rl/hf_merged/` | running | then val-200 Agent
