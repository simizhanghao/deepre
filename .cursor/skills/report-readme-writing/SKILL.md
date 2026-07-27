---
name: report-readme-writing
description: >-
  Writes README, technical report, experiment summary, resume bullets, and
  interview talking points for the DeepResearch/Search-R1 project. Use when
  documenting results or preparing presentation materials.
---

# Report and README Writing Skill

## When to Use

- Writing or updating project README
- Preparing `docs/technical_report.md`
- Drafting resume bullets or interview script

## Report Structure

1. **Motivation** — Why agentic search RL, not plain RAG
2. **Method** — WebDancer 4-stage (simplified), Search-R1 RL, multi-reward, cost control
3. **Pipeline** — data → tool → SFT → RL → eval (with diagram or bullet flow)
4. **Results** — one main table + one curve if available + one good/bad case
5. **Ablation** — link to `docs/ablation_table.md`
6. **Limitations** — small data, local BM25 not real web, no SOTA claim
7. **Interview script** — what I built, what was hard, what improved, next steps

## Writing Rules

- README readable in 3 minutes
- Include exact commands to reproduce main result
- Do not create multiple report versions (`report_final.md`, `report2.md`)
- Update existing docs in place
- Figures live in `outputs/{final_run}/figures/`; reference by relative path in docs

## Resume Bullets (Template)

> 基于 Qwen2.5-7B 与 veRL 构建 DeepResearch Agentic RL 后训练系统，支持 search/read/final 多轮工具调用；使用 DeepResearch-9K 轨迹 SFT 冷启动，在 HotpotQA/2Wiki/MuSiQue 上进行 Search-R1-style GRPO。

> 复现 multi-turn search interaction 与 retrieved token masking；设计复合 reward 优化答案、格式、证据支持与搜索成本。

> 构建 Base/RAG/ReAct/SFT/SFT+RL 对比实验，评估 EM/F1、valid action rate、search count、evidence support 等指标。

## Done Criteria

- README has: install, data prep, smoke command, eval command
- No new markdown files beyond `docs/` unless user requests
- Claims match actual numbers in `outputs/*/eval/results.csv`
