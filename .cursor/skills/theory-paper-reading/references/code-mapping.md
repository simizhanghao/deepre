# 理论层 → 代码模块映射

## Layer 1: RAG

```
query → retriever → documents → generator → answer
```

| 模块 | 文件 |
|------|------|
| One-shot baseline | `eval/one_shot_rag.py` |
| Retriever | `tools/search_bm25.py` (top-k once) |

**局限:** 只检索一次，不会根据中间发现继续搜索。

---

## Layer 2: ReAct

```
Thought → Action → Observation → ... → Final
```

| 模块 | 文件 |
|------|------|
| Agent loop | `agent/react_loop.py` |
| Action parser | `agent/parser.py` |
| Trajectory log | `agent/trajectory.py` |
| Prompts | `agent/prompts.py` |
| Baseline eval | `eval/run_eval.py` (ReAct prompt mode) |

---

## Layer 3: Tool Use

| 工具 | 文件 |
|------|------|
| search(query) | `tools/search_bm25.py` |
| read(doc_id) | `tools/read_doc.py` |
| Query cache | `tools/cache.py` |

---

## Layer 4: Search-R1

| 概念 | 文件 |
|------|------|
| Multi-turn search | `agent/react_loop.py` |
| Action format parse | `agent/parser.py` |
| Retrieved token masking | SFT/RL training scripts (obs tokens label=-100) |
| Outcome reward | `reward/answer_reward.py` |
| GRPO training | `training/rl/grpo_search.py` |

---

## Layer 5: WebDancer 四阶段

| 阶段 | 文件 |
|------|------|
| 1. Data | `data/raw/`, `scripts/prepare_data.sh` |
| 2. Trajectory | `data/processed/`, `agent/trajectory.py` |
| 3. SFT | `training/sft/`, `configs/sft_qwen7b_lora.yaml` |
| 4. RL | `training/rl/`, `configs/grpo_search_r1.yaml` |

---

## Layer 6: DeepResearch-9K / R1

| 用途 | 文件 |
|------|------|
| SFT 主数据 | `data/processed/sft_*.jsonl` |
| Trajectory filter | `scripts/prepare_data.sh` |
| Held-out eval | `eval/run_eval.py` |
| Rule reward | `reward/*.py` |

---

## 实验版本 → 代码

| Version | 实现 |
|---------|------|
| Baseline 0: Base | `eval/run_eval.py --mode base` |
| Baseline 1: RAG | `eval/one_shot_rag.py` |
| Baseline 2: ReAct | `eval/run_eval.py --mode react` |
| Version 1: SFT | `training/sft/` |
| Version 2: SFT+RL | `training/rl/` |
| Version 3: Multi-reward | `reward/evidence_reward.py` 等 |
| Version 4: Cost control | `reward/cost_reward.py` |

---

## R-Search + R1-Searcher++ → Reward 模块

| Reward 组件 | 文件 |
|-------------|------|
| answer | `reward/answer_reward.py` |
| format | `reward/format_reward.py` |
| evidence | `reward/evidence_reward.py` |
| search cost | `reward/cost_reward.py` |
