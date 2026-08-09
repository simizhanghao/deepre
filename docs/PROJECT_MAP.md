# Project Map — Evidence-Cost-Aware Deep Research Agent

> 状态：**3C CLOSED @400**；下一执行项见 [NEXT_STEPS.md](NEXT_STEPS.md)；全路线见 [ROADMAP.md](ROADMAP.md)

## 路线

```text
Search-R1 (3B) → Evidence (3C) → Cost (3D) → Full-Corpus (3E) → Phase4
                                                      ↘ Phase5M Multimodal (later)
```

Do **not** put multimodal inside 3D. Open-web = L3 after text ECA.
## Pipeline Overview

```text
Question → Planner/Reasoner → src/tools (retriever) → src/env (search_env)
        → Evidence Extractor → Answer → src/rewards/* → src/eval → results/{run}/
```

## Directory Layout (方案 A)

| Path | Status | Purpose |
|------|--------|---------|
| `.cursor/rules/` | ✅ | 常驻 guardrails |
| `.cursor/skills/` | ✅ | 10 个 skill（6 个待对齐新路线） |
| `configs/` | ✅ 空 | sft / grpo / eval yaml |
| `data/raw/` | ✅ 空 | 原始下载数据（只读） |
| `data/processed/` | ✅ 空 | 统一格式 JSONL |
| `data/sft/` | ✅ 空 | SFT 轨迹数据 |
| `data/eval/` | ✅ 空 | 评测数据 |
| `src/tools/` | ✅ 空 | retriever, bm25, dense_retriever, reranker |
| `src/env/` | ✅ 空 | search_env |
| `src/agents/` | ✅ 空 | direct, rag, react, evidence agent |
| `src/rewards/` | ✅ 空 | answer, evidence, citation, format, cost |
| `src/eval/` | ✅ 空 | metrics, failure_analysis, trace_viewer |
| `src/train/` | ✅ 空 | sft_train, grpo_train, data_collator |
| `scripts/` | ✅ 空 | 运行入口脚本 |
| `results/` | ✅ 空 | 所有实验产物（gitignore） |
| `docs/` | ✅ | PROJECT_MAP, NEXT_STEPS, RUN_LOG, reading-notes |
| `papers/` | ✅ | 10 份精读资料 |
| `outputs/` | ⚠️ legacy | 旧布局遗留，待移除，勿写新产物 |

## Data Flow

1. **Input:** HotpotQA, 2Wiki（先） → `data/raw/`
2. **Process:** validate + filter → `data/processed/*.jsonl`
3. **SFT:** trajectory（含 evidence/internal）→ `data/sft/` → `src/train/sft_train.py` → `results/sft_*/`
4. **RL:** question + gold_answer → rollout → reward → `results/rl_*/`
5. **Eval:** all methods → `results/{run}/` + trace.jsonl

## Known Missing Pieces

- [ ] trace schema（Task 1，下一步地基）
- [ ] 全部 src/ 代码模块
- [ ] FlashRAG baseline 接入
- [ ] Search-R1 最小 GRPO 闭环
- [ ] evidence / cost / routing reward
- [ ] 单元测试与 smoke 脚本

## Skills Reference

| Skill | 状态 |
|-------|------|
| `task-scoped-execution` | ✅ 保留（执行纪律） |
| `experiment-smoke-test` | ✅ 保留 |
| `theory-paper-reading` | ✅ 保留（论文已读） |
| `repo-orientation` | ⚠️ 待对齐 src/ |
| `search-tool-protocol` | ⚠️ 待加 evidence/internal |
| `sft-coldstart-training` | ⚠️ 待对齐新轨迹格式 |
| `rl-reward-grpo` | ⚠️ 待加多 reward |
| `eval-ablation` | ⚠️ 待加 evidence/cost 指标 |
| `report-readme-writing` | ⚠️ 待改主线 |
| `data-contract-validation` | ⚠️ 待对齐 trace schema |
| （新增）`trace-schema-and-failure-analysis` | ⬜ 计划新增 |
| （新增）`evidence-cost-aware-design` | ⬜ 计划新增 |
