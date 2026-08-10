# DeepResearch-Agent-RL

Evidence-Cost-Aware Deep Research Agent：HotpotQA 上的 Search-R1 风格 Agentic RL（SFT 冷启动 → GRPO）。

## 状态

当前主线：**Boundary-aware GRPO**（从 3C@400 HF 继续）。  
结果板：[docs/RESULTS_BOARD.md](docs/RESULTS_BOARD.md) · 待办：[docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)

| 阶段 | 状态 | 产物 |
|------|------|------|
| SFT-v1 | frozen | `outputs/00_sft_v1_merged` |
| Answer-only GRPO | closed @100 | `outputs/rl/02_hf_answer_only_step100` |
| Evidence GRPO | **closed @400** | `outputs/rl/03_hf_evidence_step400` |
| Evidence GEN | **PASS** | `results/10_eval_grpo_evidence_val200/` |
| Offline cost λ | **DONE** | λ_s=**0.40** · `data/rl/calib_cost_lambda_512` |
| Uniform cost | **FAIL** | `results/12_audit_uniform_cost_fail/` |
| Boundary GRPO | **active** | `outputs/rl/06_ckpt_grpo_boundary` · `src/rl/rewards_boundary.py` |

### Evidence 一句话结果（train_smoke_128）

相对 Answer-only（search≈0）：Evidence 将 **search→~1**，**Evidence F1≈0.62**，**Answer≈0.61**；后期必搜 → Boundary / Cost。

## 方法概要

- **范式:** 数据 → SFT 冷启动 → 多轮检索 GRPO
- **检索:** Candidate-BM25 + `sample_id`（非开放网页）
- **Reward:** Answer+Format → +Evidence F1 → Boundary-aware cost
- **框架:** veRL + SGLang（容器 `eca-verl`）
- **模型:** Qwen2.5-**3B**-Instruct → SFT-v1

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
