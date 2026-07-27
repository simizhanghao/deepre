# Next Steps — 最小可运行闭环

按顺序执行，每步完成后更新本文件。

## Step 0 ✅ Skills & Guardrails

- [x] `.cursor/rules/00-deepresearch-guardrails.mdc`
- [x] 8 个项目 skills
- [x] `.gitignore` + `docs/PROJECT_MAP.md`

## Step 1 — 仓库骨架（不写训练逻辑）

**调用 skills:** `repo-orientation` + `experiment-smoke-test`

创建空模块目录与占位 `__init__.py`，一个顶层 `README.md` 骨架。

交付：`agent/`, `tools/`, `reward/`, `eval/`, `configs/`, `scripts/`, `outputs/.gitkeep`

## Step 2 — 数据 schema + validator

**调用 skills:** `data-contract-validation` + `experiment-smoke-test`

- 定义 SFT / trajectory schema
- `scripts/validate_data.py --max-samples 20`
- 用 2–3 条手写样本跑通 validator

交付：`outputs/smoke_validator_*/` + pass/fail 统计

## Step 3 — Parser + mock search + trace logger

**调用 skills:** `search-tool-protocol` + `experiment-smoke-test`

- `agent/parser.py` — XML 标签解析
- `tools/search_bm25.py` — debug 模式返回固定 mock
- `agent/react_loop.py` — ≤8 题 smoke
- 单元测试：normal / malformed / no-search

交付：`outputs/smoke_agent_*/trajectories.jsonl`

## Step 4 — Baseline eval（不训练）

**调用 skills:** `eval-ablation` + `experiment-smoke-test`

- Base / RAG / ReAct prompt 三种 baseline
- ≤50 题 smoke eval
- `outputs/smoke_baseline_*/eval/results.csv`

## Step 5 — SFT smoke

**调用 skills:** `sft-coldstart-training` + `experiment-smoke-test`

- 等 DeepResearch-9K 下载后再做
- 8 样本 LoRA smoke，5 steps

## Step 6 — Reward offline test

**调用 skills:** `rl-reward-grpo` + `experiment-smoke-test`

- `reward/` 模块化 reward
- 5 条 handcrafted trajectory 排名验证
- **不启动 RL**

## Step 7 — RL smoke（需用户批准）

**调用 skills:** `rl-reward-grpo` + `experiment-smoke-test`

- 8 prompt, 2 turn, 1 GPU GRPO smoke
- 通过后再考虑 full training

---

## 驱动 Agent 的推荐 Prompt 模板

```text
使用 search-tool-protocol + experiment-smoke-test skills，
实现 parser、mock search、trace logger 和单元测试。
不要接真实搜索，不要训练。输出只写到 outputs/smoke_*。
```
