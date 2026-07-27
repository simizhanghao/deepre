# 10 资料精读地图

> 一个月内只读这 10 个，不再扩散。

---

## 资料 1：RAG（1–2 小时）

**文件:** `papers/01-RAG.pdf` | arXiv: [2005.11401](https://arxiv.org/abs/2005.11401)

**学会:**
- parametric memory = 模型参数里的知识
- non-parametric memory = 外部文档库
- retriever → generator 流水线

**代码对应:** `eval/one_shot_rag.py`

**不必学:** DPR 训练、自训 retriever

**读完后回答:**
1. RAG 和 fine-tuning 解决的是同一类问题吗？
2. 为什么 one-shot RAG 不够做多跳研究？

---

## 资料 2：ReAct（必读）

**文件:** `papers/02-ReAct.pdf` | arXiv: [2210.03629](https://arxiv.org/abs/2210.03629)

**学会:**
- Thought = 内部推理
- Action = 调用工具
- Observation = 环境返回
- Final = 结束回答
- Trajectory = 完整路径

**代码对应:** `agent/react_loop.py`, `agent/parser.py`, `agent/trajectory.py`

**读完后必须能手写:**
```python
while not done:
    model_output = model(prompt + history)
    action = parse_action(model_output)
    observation = tool(action)
    history.append(model_output, observation)
```

**读完后回答:**
1. Reasoning trace 和 action 各更新什么状态？
2. 为什么 prompt ReAct 不稳定，需要 SFT？

---

## 资料 3：Toolformer（概念即可）

**文件:** `papers/03-Toolformer.pdf` | arXiv: [2302.04761](https://arxiv.org/abs/2302.04761)

**学会:**
- 模型何时调用工具、调用哪个、传什么参数
- 工具结果如何继续用于生成

**代码对应:** `tools/search_bm25.py`, `tools/read_doc.py`

**不必学:** 完整公式推导

---

## 资料 4：Search-R1（精读 · 项目主干）

**文件:** `papers/04-Search-R1.pdf` | arXiv: [2503.09516](https://arxiv.org/abs/2503.09516)

**五个必答问题:**
1. Action 格式是什么？
2. Retrieved documents 如何插入？
3. 为什么 retrieved token masking？
4. Reward 怎么算？
5. 如何评估 Base / RAG / RL？

**代码对应:**
- `agent/parser.py`
- `tools/search_bm25.py`
- `reward/answer_reward.py`
- `training/rl/grpo_search.py`
- retrieved token mask 逻辑

**核心论点:** prompting 搜索不最优；RL 让模型学会与搜索引擎最优交互。

---

## 资料 5：WebDancer（只抓四阶段）

**文件:** `papers/05-WebDancer.pdf` | arXiv: [2505.22648](https://arxiv.org/abs/2505.22648)

**四阶段 → 本项目:**

| WebDancer | 本项目简化 |
|-----------|-----------|
| browsing data construction | 开源数据 DeepResearch-9K / HotpotQA |
| trajectory sampling | 已有轨迹 + baseline rollout |
| SFT cold start | LoRA SFT |
| RL for generalization | GRPO |

**面试用途:** 解释"为什么这样设计项目"

---

## 资料 6：DeepResearch-9K / DeepResearch-R1

**文件:** `papers/06-DeepResearch-9K.pdf` | arXiv: [2603.01152](https://arxiv.org/abs/2603.01152)

**Repo:** [DeepResearch-R1](https://github.com/Applied-Machine-Learning-Lab/DeepResearch-R1)

**搞清楚:**
- question / trajectory / reasoning chain / final answer 字段
- L1/L2/L3 难度定义
- SFT 怎么跑、GRPO/PPO 怎么配
- rule-based vs LLM-as-judge reward

**代码对应:** `data/`, `training/sft/`, `training/rl/`

---

## 资料 7：veRL Agentic RL 文档

**文件:** `papers/07-verl-agentic-rl.md`（链接汇总，非 PDF）

**学会:**
- rollout 是什么
- multi-turn tool call 怎么接
- reward function 怎么传
- GRPO/PPO 在哪配置

**不必:** 一开始读 veRL 全部源码

---

## 资料 8：R-Search（强化版依据）

**文件:** `papers/08-R-Search.pdf` | arXiv: [2506.04185](https://arxiv.org/abs/2506.04185)

**学会:**
- 为什么只用最终答案 reward 不够
- multi-stage / multi-type reward
- 如何奖励轨迹质量

**本项目简化 reward:**
- answer + format + search validity + evidence support + search cost penalty

---

## 资料 9：R1-Searcher++（成本意识）

**文件:** `papers/09-R1-Searcher++.pdf` | arXiv: [2505.17005](https://arxiv.org/abs/2505.17005)

**学会:**
- 为什么不能无脑搜索
- SFT cold-start + RL 两阶段
- internal knowledge utilization 思想

**本项目简化:**
- search 次数惩罚、重复 query 惩罚、efficiency bonus

---

## 资料 10：LangChain Open Deep Research（产品参考）

**文件:** `papers/10-open-deep-research.md`

**只看:** 产品形态、报告结构、search/read/summarize 串联

**禁止:** 做成 LangChain 应用项目

---

## 建议阅读顺序

```
ReAct (2) → RAG (1) → Toolformer (3) → Search-R1 (4) → WebDancer (5)
→ DeepResearch-9K (6) → veRL (7) → R-Search (8) → R1-Searcher++ (9)
→ Open Deep Research (10, 可选)
```

Search-R1 可在 ReAct 之后立即精读。
