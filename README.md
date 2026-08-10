# DeepResearch-Agent-RL

Evidence-Cost-Aware Deep Research Agent：HotpotQA 上的 Search-R1 风格 Agentic RL（SFT 冷启动 → GRPO）。

## 状态

当前主线：**Rollout Alignment Recovery**（硬结论 `SGLANG_ROUTE_TOKEN_LOGIT_TIM` → VeXact / HFExact）。  
结果板：[docs/RESULTS_BOARD.md](docs/RESULTS_BOARD.md) · 待办：[docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)

| 阶段 | 状态 | 产物 |
|------|------|------|
| SFT-v1 | frozen | `outputs/00_sft_v1_merged` |
| Answer-only GRPO | closed @100 | `outputs/rl/02_hf_answer_only_step100` |
| Evidence GRPO | **closed @400** | `outputs/rl/03_hf_evidence_step400` |
| Evidence GEN | **PASS** | `results/10_eval_grpo_evidence_val200/` |
| Offline cost λ | **DONE** | λ_s=**0.40** · `data/rl/calib_cost_lambda_512` |
| Uniform cost | **FAIL** | `results/12_audit_uniform_cost_fail/` |
| Boundary table | **PASS** | `outputs/rl/04_table_search_boundary` |
| Boundary GRPO | routing **FAIL** | `outputs/rl/06_ckpt_grpo_boundary` · `src/rl/rewards_boundary.py` |
| Routing / TIM audit | **CLOSED** | `results/16_.../worker_mismatch/` · `SGLANG_ROUTE_TOKEN_LOGIT_TIM` |
| **Rollout Alignment Recovery** | **NOW** | `results/17_rollout_alignment/` · VeXact → Gate A → budget → Boundary@50 |

### Evidence 一句话结果（train_smoke_128）

相对 Answer-only（search≈0）：Evidence 将 **search→~1**，**Evidence F1≈0.62**，**Answer≈0.61**；后期必搜 → Boundary。  
Boundary 学不动的主因现已定位为 **SGLang route-token TIM**（非单纯 reward/GRPO）。

## 方法概要

- **范式:** 数据 → SFT 冷启动 → 多轮检索 GRPO（**须 trainer-aligned rollout**）
- **检索:** Candidate-BM25 + `sample_id`（非开放网页）
- **Reward:** Answer+Format → +Evidence F1 → Boundary-aware cost（对齐前不改）
- **框架:** veRL；历史基线 SGLang（`eca-verl`）；**对齐主线** VeXact / HFExact（`eca-verl-vexact`）
- **模型:** Qwen2.5-**3B**-Instruct → SFT-v1 → Evidence@400

## 文档

| Doc | 内容 |
|-----|------|
| [ROADMAP.md](docs/ROADMAP.md) | 主线与门禁 |
| [NEXT_STEPS.md](docs/NEXT_STEPS.md) | 当前待办 |
| [RESULTS_BOARD.md](docs/RESULTS_BOARD.md) | 实验结果总表 |
| [PROJECT_MAP.md](docs/PROJECT_MAP.md) | 仓库地图 |

## 关键路径

```text
outputs/00_sft_v1_merged
outputs/rl/03_hf_evidence_step400
outputs/rl/04_table_search_boundary/boundary_latest.json
data/rl/train_smoke_128/
src/rl/rewards_evidence.py
src/rl/rewards_boundary.py
configs/sft/sft_v1_lora.yaml
configs/rl/grpo_smoke128.yaml
```

## Cursor Agent Skills

本项目使用 `.cursor/skills/` 下的专用 skills（勿装 marketplace skills）。

## 工程约束

- 大权重 / `outputs/` / 多数 `results/` 默认 gitignore
- `external/` 只读第三方参考，勿改
- 详见 `.cursor/rules/00-deepresearch-guardrails.mdc`
