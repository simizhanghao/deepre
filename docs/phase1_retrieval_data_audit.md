# Phase 1A0 — 真实数据与检索基础设施审计（决策合同）

> **性质：只做决策，不做实现。**  
> 本文件固定 Phase 1 后续所有实验的基础设施口径。  
> 不安装依赖、不下载数据、不建索引、不跑模型、不实现 retriever / Agent / reward / SFT / GRPO。

依据：`docs/PROJECT_MAP.md`、`docs/searchR1_read.md`、`docs/searchR1_to_eca_mapping.md`、  
`src/eval/trace_schema.py`、`src/eval/metrics.py`、`scripts/smoke_direct.py`、  
`scripts/smoke_rag.py`、`external/FlashRAG` README、以及 Phase 0 只读环境审计结果。

---

## 0. 环境审计快照（已确认事实）

| 项 | 状态 |
|----|------|
| Phase 0 | ✅ TraceRecord + metrics + Direct/toy-RAG smoke 已通 |
| Generator | ✅ 本地 `Qwen2.5-3B-Instruct`（`/data1/hcc/.hf_home/...`） |
| `deepresearch` env | ✅ torch `2.11.0+cu128`，transformers `5.14.1`（刚整理干净） |
| FlashRAG 源码 | ✅ `external/FlashRAG`，commit `e0e7339` |
| FlashRAG `datasets/` / `indexes/` / `models/` | ❌ 空 |
| conda 是否安装 `flashrag` | ❌ |
| HotpotQA / NQ / Wikipedia corpus | ❌ 尚未下载 |
| Java | ❌ 未安装 |
| `bm25s` / `pyserini` / `faiss` | ❌ 均未安装 |
| `/data1` 磁盘 | ✅ 约 1.1 TB 可用 |

**结论（环境）：** 不得对当前 `deepresearch` env 执行 `pip install flashrag-dev[full]`。

---

## 1. Phase 1 目标：为什么现在不能进入 SFT / RL

Phase 0 只证明了工程链路能跑：

```text
手工问题 → 真实模型 → Direct / toy RAG → TraceRecord → metrics → 落盘
```

它**不能**回答：真实数据上 RAG 是否有用、BM25 能否找到证据、错误来自检索还是生成、检索成本是否值得。

Phase 1 的唯一目标：

> 在**同一批真实问题**上，用**同一个 Qwen2.5-3B-Instruct**，比较 Direct / Oracle RAG / BM25 RAG，  
> 把答案质量、检索质量、成本全部写入统一 TraceRecord，形成可信 baseline 与失败诊断。

因此 Phase 1 **不做**：

- SFT / GRPO / 多步自主搜索 / `<search>` 生成  
- evidence reward / cost reward / internal-external router 训练  

没有 Phase 1 的诊断表，Phase 2 不知道该教什么，Phase 3 不知道该奖励什么。

---

## 2. 第一 benchmark：HotpotQA

**拍板：Phase 1 第一真实 benchmark = HotpotQA。**

理由：

1. 项目地图已定：HotpotQA、2Wiki 优先（`PROJECT_MAP`）。
2. 多跳 + sentence-level supporting facts，天然支持 retrieval eval、evidence、citation、后续 multi-hop。
3. 比单跳 QA 更贴近 Deep Research Agent 目标。
4. FlashRAG 已预处理 HotpotQA（wiki 域；train/dev 规模见其数据集表），便于后续复用资源口径。  
   参考：[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)、[HotpotQA](https://arxiv.org/abs/1809.09600)。

**已知局限（提前写死）：**

> HotpotQA 多数题偏“需要外部证据”，适合评测 evidence / search，**不适合单独训练** internal/external routing。  
> Routing 数据需在后续阶段混入小模型可直接回答的简单题。

---

## 3. HotpotQA 数据合同（Phase 1A1 必须固定的字段）

Phase 1A1 将把原始样本转为项目 eval JSONL。合同字段如下（实现阶段不得自由发挥）：

| 字段 | 要求 |
|------|------|
| `sample_id` | `hotpotqa_{split}_{index}`，与 mapping 的 `{data_source}_{split}_{index}` 一致 |
| `question` | 纯问题文本，不含指令模板 |
| `gold_answers` | `List[str]`，多答案全部保留 |
| `supporting_facts` | `[{title, sentence_id}, ...]`，保留 sentence-level 金标 |
| `contexts`（如有） | 题包自带 context（title + sentences）；用于 Oracle / Candidate-BM25 |
| `document_id` | 稳定构造规则（建议：`{title}` 或 `{title}#{sent_id}`，1A1 定一种并全局沿用） |
| `metadata` | 至少含 `dataset=hotpotqa`、`split`、原始 `level`/`type`（若有） |

**待 1A1 明确拍板（本文件先登记，不在本轮下载数据）：**

1. **split**：默认优先 `validation` / FlashRAG 预处理后的 `dev`（以实际文件名为准，1A1 写死）。
2. **Oracle 喂给模型的粒度**：整篇 gold context vs supporting sentences only —— 推荐默认 **gold supporting documents（按 title 聚合的金标段落/文章）**；若只用 supporting sentences 必须在 run_info 标注 `oracle_granularity=sentences`。
3. **BM25 检索范围**：Candidate（题内 context）vs Full-Corpus（全 Wiki）—— 见第 4 节，**禁止混名**。

---

## 4. 三种 RAG 设定：命名必须区分

| 名称 | 检索输入 | 考查什么 | Phase 1 角色 |
|------|----------|----------|--------------|
| **Oracle RAG** | 数据集提供的标准 supporting / gold docs，无检索器 | 证据完全正确时，3B 能否读懂并答对（生成上界） | 必做 |
| **Candidate-BM25 RAG** | BM25 仅在**该题自带 contexts** 内排序 Top-K | 低成本诊断：排序/阅读是否正常 | 诊断版（可先做） |
| **Full-Corpus BM25 RAG** | BM25 在**Wikipedia 全语料**上 Top-K | 接近 Search-R1 的真实 search 环境 | 正式检索 baseline |

**硬性规则：**

- 不得把 Candidate-BM25 报告为 “BM25 RAG” 或 “真实检索”。
- 实验表与 `run_info` 必须写全称：`oracle_rag` / `candidate_bm25_rag` / `full_corpus_bm25_rag`。
- Full-Corpus 在语料/索引就绪前可标记为 `planned`；未就绪时不得伪造数字。

诊断框架：

```text
Direct → Oracle          = 外部知识理论上限收益
Oracle → Full-Corpus BM25 = 检索器造成的损失
Direct → Full-Corpus BM25 = 真实检索净收益是否抵得过成本
```

---

## 5. BM25s vs Pyserini

| 维度 | BM25s | Pyserini |
|------|-------|----------|
| 依赖 | Python 原生，轻量（FlashRAG 亦将其标为 Pyserini 的轻量替代） | Java + Lucene，安装复杂 |
| 本机现状 | 未装，但**可装且无 Java 前置** | 未装，且 **Java 不存在** |
| Phase 1 必要性 | ✅ 足够做 Top-K、保留 doc_id/score/rank | ❌ 当前不解决已有问题 |
| 与 Search-R1 对齐 | 语义同为稀疏 BM25；实现栈不同 | 更接近大规模 Wiki / 部分官方索引生态 |
| FlashRAG 态度 | README：因 Pyserini 安装复杂引入 BM25s；`bm25_backend=bm25s` | 仍支持，但官方倾向 BM25s 作为更易用路径 |

参考：FlashRAG changelog（2024-09-18）明确因 Pyserini 安装限制引入轻量 `BM25s`。

---

## 6. 当前推荐（拍板）

| 决策项 | 拍板 |
|--------|------|
| Retriever baseline | **BM25s** |
| 完整 FlashRAG 安装 | **暂不**；不执行 `flashrag-dev[full]` |
| Pyserini | **Phase 1 暂缓**；Full-Corpus / Search-R1 严格对齐时再评估 |
| Generator | **Qwen2.5-3B-Instruct**（与 Phase 0 一致，不换 7B） |
| FlashRAG 角色 | 参考实现 + 数据/索引资源口径；**不是**主 pipeline |

---

## 7. 检索与生成解耦（工程合同）

Phase 1 固定两段式，禁止“生成进程内临时检索且不落盘”：

```text
# Retrieval side（可独立环境）
eval JSONL
  → BM25s
  → retrieval_results.jsonl
       每条至少含：
       sample_id, query, document_id, rank, score, title, text
       (+ method: candidate_bm25 | full_corpus_bm25, top_k, index_id)

# Generation side（deepresearch env）
eval JSONL + retrieval_results.jsonl（或 Oracle 金标 docs）
  → Qwen2.5-3B
  → TraceRecord（documents[] + observation + answer）
  → metrics / summary
```

价值：

1. **复现**：检索缓存固定后，换 prompt / checkpoint 不改变检索侧。  
2. **环境隔离**：BM25 依赖不污染已稳定的 `deepresearch` torch/transformers。  
3. **Failure analysis**：可单独检查“正确文档是否进 Top-K” vs “进了为何仍答错”。

与 Search-R1「HTTP retrieval server ↔ 训练进程」同构，只是 Phase 1 用 jsonl 缓存代替服务。

映射到已有 schema：`retrieval_results` → `TraceRecord.documents: Document[]`  
（`document_id/title/text/rank/score` 已在 `trace_schema.py` 定义）。

---

## 8. FlashRAG 当前角色

```text
FlashRAG =
  参考实现（BM25s backend、corpus/index 构建脚本、数据集字段习惯）
  + 数据/索引资源来源（HotpotQA 预处理、wiki corpus、可选预建 index）

deepresearch =
  TraceRecord / metrics / cost / failure analysis
  + 后续 Agent rollout / reward / SFT / RL
```

- `external/FlashRAG` 保持只读参考，不把评测体系交给 FlashRAG pipeline。  
- 若需跑 FlashRAG 官方脚本，**单独 conda env**，不并入 `deepresearch`。  
- 固定参考 commit：`e0e7339`（升级需另开决策，不得静默更换）。

---

## 9. Phase 1 固定 baseline 方法

| method_id | 说明 |
|-----------|------|
| `direct` | 不检索，Question → Model → Answer |
| `oracle_rag` | 金标证据 → observation → Answer |
| `candidate_bm25_rag` | 题内 context 上 BM25s Top-K → observation → Answer（诊断） |
| `full_corpus_bm25_rag` | 全语料 BM25s Top-K → observation → Answer（正式） |

统一约束（所有方法）：

- 同一 `eval` subset（同一 `sample_id` 列表）  
- 同一模型路径与解码超参（写入 `run_info.json`）  
- 同一答案 prompt 风格（Direct / RAG 仅差是否注入 context，差异必须文档化）  
- 同一 `basic_metrics` + 同一 TraceRecord schema  
- 产物目录：`results/{run_name}/`（`trace.jsonl` + `metrics.json` + `run_info.json` + 可选 `summary.json`）

---

## 10. 样本规模阶梯

| N | 目的 |
|---|------|
| **8** | pipeline smoke（真实 HotpotQA，非手工 Madison） |
| **50** | 第一次趋势（Direct / Oracle / BM25 方向是否合理） |
| **200** | 正式 baseline 小实验（面试/报告主表） |
| 完整 dev | 系统稳定后再考虑；**非 Phase 1 默认** |

子集构造要求（Phase 1B）：固定 `seed`、固定 `sample_id` 列表文件，三方法共用，禁止各跑各的随机抽题。

---

## 11. Phase 1 必须报告的指标

### Answer

- `exact_match`
- `token_f1`
- （聚合）`mean_em` / `mean_token_f1` / `format_valid_rate`

### Retrieval（Oracle 可记 recall=1.0 或标 “—”；Direct 标 “—”）

- Document Recall@1 / @3 / @5（相对 gold supporting titles / docs）  
- Supporting-fact coverage（金标 supporting facts 被 Top-K 覆盖的比例；具体操作定义在 1A1/1E 固化）

### Cost（对齐 `CostInfo` + 聚合）

- `retrieved_document_count`
- `prompt_tokens`
- `observation_tokens`
- `generated_tokens`
- `latency_ms`（及均值）

### Structure

- `format_valid`（`validate_trace_record`）

第一张正式表至少包含：

| 方法 | 检索方式 | EM | F1 | Recall@K | Prompt tokens | 延迟 |
|------|----------|---:|---:|---------:|--------------:|-----:|
| Direct | 无 | | | — | | |
| Oracle RAG | 金标证据 | | | 1.0 | | |
| Full-Corpus BM25 | BM25s Top-K | | | | | |

Candidate-BM25 可作为附录诊断行，不得替代 Full-Corpus 主行（若 Full-Corpus 尚未就绪，主表注明 pending）。

---

## 12. Phase 1F Failure Taxonomy

按**同一 `sample_id`** 在 Direct / Oracle / BM25 上的对错归类（BM25 默认指当时启用的正式检索设定，run 中写明 candidate 或 full-corpus）：

| 类别 | 模式 | 解读 |
|------|------|------|
| A | Direct❌ Oracle✅ BM25✅ | 检索真正有帮助 |
| B | Direct❌ Oracle✅ BM25❌ | 检索器未找到正确证据 |
| C | Direct❌ Oracle❌ | 生成器不会用证据 / 推理不足 |
| D | Direct✅ RAG✅ 但成本更高 | 应走 internal；搜索浪费 |
| E | Direct✅ RAG❌ | 检索噪声误导 |

Phase 0 的 James Madison 烟测属于 **D 的原型**；Phase 1 要统计真实数据中各类比例。

---

## 13. 实施顺序（本轮之后）

```text
Phase 1A0  本文件：数据/检索/环境决策合同          ← 完成即停
    ↓
Phase 1A1  固定 HotpotQA eval schema + retrieval cache contract（仍可不下载完整 Wiki）
    ↓
Phase 1B   构造 8 → 50 → 200 真实 eval subset（只数据处理）
    ↓
Phase 1C   批量 Direct baseline（开始用 GPU；建议单卡即可）
    ↓
Phase 1D   Oracle RAG
    ↓
Phase 1E   BM25s retrieval cache → Candidate / Full-Corpus RAG
    ↓
Phase 1F   对比表 + failure taxonomy
```

**GPU 说明：** 4×GPU（4567）对 Phase 1A0–1B **不需要**。  
Phase 1C 起再用；baseline 推理默认单卡足够，多卡留待后续吞吐优化，不作为 Phase 1 阻塞项。

---

## 14. Non-goals（Phase 1A0）

- ❌ 安装 `bm25s` / `pyserini` / `faiss` / `flashrag`  
- ❌ 下载 HotpotQA / Wikipedia / 预建 index  
- ❌ 创建或修改除本文件外的任何代码/数据  
- ❌ 运行模型或占用 GPU  
- ❌ 实现 retriever / Agent / evidence reward  
- ❌ SFT / GRPO  

---

## 15. 决策摘要（一页纸）

```text
Benchmark:     HotpotQA
Scale:         8 → 50 → 200
Generator:     Qwen2.5-3B-Instruct（Phase 0 不变）
Retriever:     BM25s
Env:           检索与生成解耦；不污染 deepresearch env；FlashRAG 可另开 env
FlashRAG:      参考实现 / 数据资源；非主 pipeline；commit e0e7339
Pyserini:      Phase 1 暂缓
Baselines:     Direct | Oracle RAG | Candidate-BM25 | Full-Corpus BM25
Output:        TraceRecord + retrieval_results.jsonl + metrics summary
Next:          Phase 1A1（eval schema + retrieval cache contract）
```

**Phase 1A0 验收：** 上述四问已回答——数据用哪个 setting 的合同入口、检索器用 BM25s、环境解耦、三（四）方法同题同模同指标可比。

---

*文档版本：Phase 1A0 / 2026-08-07*
