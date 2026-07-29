# Search-R1 to ECA Mapping

> Search-R1 的原始数据、rollout 标签、检索注入、token mask、reward,逐项映射到本项目
> (Evidence-Cost-Aware Deep Research Agent,下称 ECA)的统一 trace schema。
>
> 依据:`src/eval/trace_schema.py`(schema v0.1)、`src/eval/metrics.py`、`docs/searchR1_read.md`。

## 1. Purpose and Scope

本文档是后续 parser、adapter、SFT 数据转换、RL rollout 和评测的**接口依据**:
Phase 0.2 起所有"Search-R1 侧数据/轨迹 → TraceRecord"的转换代码必须遵循本映射,不得自由发挥。

本阶段(Phase 0.1D)**只定义映射语义,不实现任何代码,不定义 reward 权重与公式**。

## 2. Input Data Field Mapping

Search-R1 训练数据字段(见 `external/Search-R1/scripts/data_process/nq_search.py`,
解读见 `docs/searchR1_read.md` 第 1 节)映射如下:

| Search-R1 字段 | ECA 位置 | 说明 |
|----------------|----------|------|
| `data_source` | `TraceRecord.metadata["data_source"]` | 数据集名(nq / hotpotqa / ...) |
| `prompt` | `TraceRecord.question` | **仅提取用户问题本身**,见下方注意事项 |
| `ability` | `TraceRecord.metadata["ability"]` | 任务类型标签,原样保留 |
| `reward_model.ground_truth` | `TraceRecord.gold_answers` | 统一转为 `List[str]`(单个字符串包成列表) |
| `extra_info.index` | `TraceRecord.sample_id` | ⭐ 同题多 rollout 的分组依据,见下 |
| `extra_info` 其余字段(如 `split`) | `TraceRecord.metadata["extra_info"]` | 原样保留 |

**sample_id / trace_id 约定**:

- `extra_info.index` 是 int,`sample_id` 是 str,且 index 仅在单个数据集内唯一。
  构造规则:`sample_id = f"{data_source}_{split}_{index}"`(如 `nq_train_1024`),跨数据集不撞号。
- 同一道题的多条 rollout **共用 sample_id**(未来 GRPO 按它分组算组内均值/方差/advantage,
  对应 Search-R1 中 `uid = index` 的用法,见 `docs/searchR1_read.md` 7.3 节);
  每条具体 rollout 使用独立 `trace_id`,如 `nq_train_1024_grpo_0`、`nq_train_1024_grpo_1`。

**prompt → question 的注意事项(parser 必须遵守)**:

1. Search-R1 的 `prompt` = 指令模板(声明 `<think>/<search>/<information>/<answer>` 用法)+ 用户问题。
   映射到 `question` 时必须**剥掉模板,只保留问题文本**;模板属于 agent 的 prompt 构造,不属于数据。
2. 模板内含示例 `<answer> Beijing </answer>`。Search-R1 的打分器因此要求完整序列中
   `<answer>` 出现 ≥2 次才认为模型真的作答(`qa_em.py::extract_solution`)。
   parser 解析 rollout 时**必须跳过模板中的示例标签**,只解析模型生成部分,
   否则会把示例误当成 answer step。

## 3. Rollout Step Mapping

Search-R1 rollout 文本标签(见 `docs/searchR1_read.md` 第 2 节)映射到 `TraceStep.step_type`:

| Search-R1 表示 | ECA step_type | 内容(`content`) |
|----------------|---------------|------------------|
| `<think>...</think>` | `think` | 显式推理文本 |
| `<search>...</search>` | `search` | 搜索 query |
| `<information>...</information>` | `observation` | 检索器返回并注入上下文的内容 |
| `<answer>...</answer>` | `answer` | 最终答案 |
| (ECA 新增)`<evidence>...</evidence>` | `evidence` | 模型主动选择/输出的证据 |
| (ECA 新增)`<internal>...</internal>` | `internal` | 模型决定依赖参数知识、不检索的显式路由输出 |

三种概念轨迹(结构示例,不实现 parser):

```text
Direct:
  question → answer

Search-R1 style:
  question → think → search → observation → think → answer

ECA full trace:
  question → think → (internal 或 search) → [observation → evidence] → think → answer
```

说明:

- `internal` 与 `search` 是**互斥的路由分支**,不要求同时出现:
  模型判断参数知识足够 → 走 `internal`;需要外部知识 → 走 `search`。
- `observation`、`evidence` 仅在走了 `search` 分支后出现;一条轨迹可有多轮 search。
- schema validator 要求恰好一个 `answer` 且位于末尾(未完成 rollout 的处理留待 Phase 0.2,
  见 `trace_schema.py` 已知限制)。

## 4. Document, Observation and Evidence

三者是不同的东西,不得混用:

| 概念 | 定义 | 存放位置 |
|------|------|----------|
| `Document` | 检索器返回的结构化 passage,带 `document_id/title/text/rank/score` | `TraceRecord.documents` |
| `observation` step | 把一个或多个 Document **格式化后注入模型上下文**的环境内容 | `TraceRecord.steps`(step_type=observation) |
| `Evidence` | 模型从 Document/Observation 中**主动选择**的支持答案片段 | `TraceRecord.evidences` |

引用关系:

```text
Retriever
    ↓ 返回结构化结果
Document[]                      (documents 列表,带稳定 document_id)
    ↓ 格式化并注入上下文
observation step                (step.document_ids 指向来源 Document)
    ↓ 模型选择/压缩
Evidence[]                      (Evidence.document_id 指向来源 Document)
    ↓
evidence step / answer citation (step.evidence_ids 指向 Evidence)
```

关键约定:

- **Evidence 是模型的输出行为,不是检索结果的别名**。检索器返回的东西永远是 Document;
  只有模型主动挑出、压缩或引用的片段才是 Evidence。
- **为什么需要稳定 ID**:citation reward 要验证"模型引用的证据确实来自某篇检索文档",
  没有 `document_id` 就无法回溯。Search-R1 现状是格式化检索结果时只保留 title/text、
  丢弃 doc_id(`generation.py::_passages2string`,见 `docs/searchR1_read.md` 第 2 节坑点),
  ECA 的 adapter 必须在 Document 层保留检索器返回的原始 id。

## 5. Loss Mask Mapping

Search-R1 的 retrieved token masking(`info_mask`,完整链路见 `docs/searchR1_read.md`
7.6 节)在 ECA schema 中抽象为 step 级 `loss_mask`,规则固化于
`trace_schema.py::EXPECTED_LOSS_MASK`,validator 强制校验:

| step_type | loss_mask | 原因 |
|-----------|----------:|------|
| think | True | 模型生成 |
| internal | True | 模型生成的路由决策 |
| search | True | 模型生成的工具调用 |
| observation | **False** | 环境注入 |
| evidence | True | 模型主动选择并输出 |
| answer | True | 模型生成 |

语义:**attention 可见 ≠ 参与训练 loss**。

- observation 会进入上下文,模型可以注意到它;但其 token 不是模型生成的动作,
  不进入 SFT target、policy loss、KL loss(对应 Search-R1 中 info_mask 同时作用于
  KL 与 policy loss 两处)。
- evidence 即使内容来源于 observation,也是模型主动选择/重新输出的,**不 mask**。
- 当前 schema 只保存 **step 级** mask;展开为 token 级 mask 是未来 data collator 的职责
  (本阶段不实现 collator)。

## 6. Metrics and Reward Extension

当前已实现的基础 metrics(`src/eval/metrics.py`):

| metric | 含义 |
|--------|------|
| `exact_match` | 归一化后与任一 gold 完全相等(0/1) |
| `token_f1` | 归一化分词后的 token 重叠 F1,多 gold 取最大 |
| `search_count` | search step 数量(从 steps 现场重算) |
| `duplicate_query_count` | 词面归一化后重复的 query 次数 |
| `format_valid` | 结构校验是否通过 |

未来 reward 组件(字段已定义于 `trace_schema.py::RewardBreakdown`,计算未实现):

| reward 组件 | 对应 Search-R1 现状 |
|-------------|---------------------|
| `answer_reward` | 即 Search-R1 唯一的 EM outcome reward(`qa_em.py`) |
| `evidence_reward` | 无,ECA 新增 |
| `citation_reward` | 无,ECA 新增 |
| `format_reward` | 有接口(`format_score`)但默认关闭 |
| `cost_reward` | 无,ECA 新增 |
| `duplicate_query_penalty` | 无,ECA 新增 |

**metric 与 reward 的边界**:

- metrics **报告事实**(如 `search_count = 3`),用于评测与分析,不参与反向传播;
- reward 把事实**转换为训练信号**(如未来 `rewards/cost.py` 按配置把 search_count
  折算成 cost_reward),可能含权重、裁剪、惩罚;
- 本阶段不定义任何权重与公式;原始分数与权重将在后续 reward config 中分离。
- 层次对应:Search-R1 的 EM 既是 metric 也是 reward(二者合一);ECA 把两层拆开,
  `metrics.exact_match` 是评测口径,`RewardBreakdown.answer_reward` 是训练信号,
  即使数值来源相同也分开存放。

## 7. Search-R1 Gaps and ECA Extensions

| Search-R1 当前能力 | ECA 改造 |
|--------------------|----------|
| 只优化最终答案(单一 EM outcome reward) | multi-reward:answer + evidence + citation + format + cost |
| retrieved information 被 mask(info_mask) | 保留相同原则(observation loss_mask=False) |
| 无显式 evidence step | 新增 `evidence` step + `Evidence` 结构 |
| 无显式 internal routing | 新增 `internal` step(内外知识路由) |
| 搜索次数约束弱(仅 max_turns 截断) | cost metrics(已实现)→ cost reward(未来) |
| 重复 query 无独立统计 | `duplicate_query_count` metric(已实现)→ penalty(未来) |
| 格式化检索结果时丢弃 doc_id | Document 保留稳定 `document_id` |
| 只看 answer 对错,无过程分析 | full trace 分析(steps + documents + evidences + cost) |
| 稀疏 0/1 reward 易触发 GRPO zero-std | dense multi-reward 缓解组内全同(见 `searchR1_read.md` 7.2) |

## 8. Adapter Contract for Later Phases

未来的 Search-R1 → TraceRecord adapter(Phase 0.2+ 实现)必须完成以下职责:

1. 从原始样本解析问题文本(剥掉指令模板)与 gold answers(统一为 `List[str]`);
2. 按第 2 节规则构造 `sample_id`(`{data_source}_{split}_{index}`)与 `trace_id`;
3. 按第 3 节映射解析 rollout 文本为有序 `TraceStep` 列表(跳过模板示例标签,
   step_id 从 0 连续编号);
4. 保留检索器返回的结构化 `Document`(含原始 doc_id);
5. 建立 Evidence → Document、step → document/evidence 的 ID 引用;
6. 按第 5 节规则设置每个 step 的 `loss_mask`;
7. 汇总填充 `CostInfo`(search_count 等,与 steps 保持一致);
8. 产出每条 TraceRecord 后调用 `validate_trace_record`,校验不通过的记录
   必须显式上报,不得静默丢弃。

本文档只定义契约,不编写 adapter 代码。

## 9. Non-Goals

本阶段(Phase 0.1)不做:

- parser / adapter 实现
- 检索器接入
- Agent 实现(Direct / RAG / ReAct / evidence agent)
- reward 计算与权重定义
- token 级 collator
- SFT / RL 训练
- 语义级证据验证(evidence 是否真支持答案、query 语义判重等)

## 10. Phase 0.1 Acceptance Checklist

- [x] schema 映射完整(第 2 节:6 个数据字段全部有落点,sample_id 规则明确)
- [x] rollout 映射完整(第 3 节:4 个原生标签 + 2 个新增标签,三种概念轨迹)
- [x] mask 语义明确(第 5 节:6 种 step 的 loss_mask 及理由,step 级 → token 级的分工)
- [x] reward 扩展边界明确(第 6 节:metric 报告事实 / reward 转训练信号,不定义权重)
- [x] 没有新增实现代码(本文档为纯 markdown)
- [x] 没有超出单文件范围(仅创建 `docs/searchR1_to_eca_mapping.md`)
