# DeepResearch-Agent-RL

基于开源数据的 DeepResearch / Search-R1 风格 Agentic RL 后训练系统。

## 状态

🟡 工程初始化中 — skills 与 guardrails 已就绪，代码模块待建。

## 方法概要

- **范式:** WebDancer 四阶段（数据 → SFT 冷启动 → RL 泛化）
- **RL:** Search-R1 多轮搜索 + retrieved token masking
- **Reward:** R-Search 多奖励 + R1-Searcher++ 搜索成本控制
- **框架:** veRL / DeepResearch-R1（待集成）
- **模型:** Qwen2.5-7B-Instruct

## 目录结构

见 [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md)

## Cursor Agent Skills

本项目使用 `.cursor/skills/` 下的 8 个专用 skills，不要安装第三方 marketplace skills。

| Skill | 用途 |
|-------|------|
| `repo-orientation` | 仓库结构与模块定位 |
| `data-contract-validation` | 数据 schema 与校验 |
| `search-tool-protocol` | ReAct 格式与搜索工具 |
| `sft-coldstart-training` | SFT 冷启动 |
| `rl-reward-grpo` | RL reward 与 GRPO |
| `eval-ablation` | 评测与消融 |
| `experiment-smoke-test` | Smoke 测试门禁 |
| `report-readme-writing` | 文档与报告 |
| `theory-paper-reading` | 10 篇论文精读导读 |

## 下一步

见 [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)

## 工程约束

- 所有产物写入 `outputs/{run_name}/`
- Smoke 默认 ≤8 samples，长训需明确批准
- 详见 `.cursor/rules/00-deepresearch-guardrails.mdc`
