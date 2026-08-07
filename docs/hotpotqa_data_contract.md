# HotpotQA Data and Retrieval Contract

> Phase **1A1** 数据合同。只定义接口，不下载数据、不实现 converter / retriever / runner。  
> 上级决策见 `docs/phase1_retrieval_data_audit.md`；轨迹落点见 `src/eval/trace_schema.py`。

---

## 1. Purpose

保证下列四条实验线使用**同一批问题**和**同一套数据接口**，仅检索上下文不同：

| method_id | 上下文来源 |
|-----------|------------|
| `direct` | 无检索 |
| `oracle_rag` | supporting_facts 涉及的金标文档 |
| `candidate_bm25_rag` | 题内 contexts 上 BM25s Top-K |
| `full_corpus_bm25_rag` | 固定 Wiki 全语料 BM25s Top-K |

没有本合同，四条线会各自发明字段名 / id / Oracle 定义，导致 EM/F1 与 failure taxonomy 不可比。

---

## 2. Eval Sample Schema

每条评测样本为一行 JSON（未来 JSONL）。**必填 / 可选**如下。

| 字段 | 类型 | 必填 | 语义 |
|------|------|------|------|
| `sample_id` | `str` | ✅ | 问题级稳定 ID（见 §3） |
| `question` | `str` | ✅ | 纯问题文本，无指令模板 |
| `gold_answers` | `List[str]` | ✅ | 可接受答案列表（见 §4） |
| `supporting_facts` | `List[obj]` | ✅ | 金标证据身份（见 §5） |
| `contexts` | `List[obj]` | ✅* | 题包自带候选文档（见 §6）；*Full-Corpus 生成侧可不读，但 eval 文件仍保留以便 Oracle/Candidate |
| `metadata` | `Dict` | ✅ | 至少含 `dataset`、`split` |

### `supporting_facts[]` 元素

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `title` | `str` | ✅ | 与某条 context 的 `title` 对齐 |
| `sentence_id` | `int` | ✅ | 该 title 下 0-based 句子下标 |
| `sentence` | `str` | 可选 | 若源数据可稳定得到则保留；**不是** gold identity |

### `contexts[]` 元素

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `document_id` | `str` | ✅ | 样本内稳定 ID（见 §6） |
| `title` | `str` | ✅ | 原始标题 |
| `sentences` | `List[str]` | ✅ | 原始句子顺序，不得打乱 |
| `text` | `str` | ✅ | 由 `sentences` 按序拼接派生（见 §6） |

### `metadata` 最小集

| 键 | 说明 |
|----|------|
| `dataset` | 固定 `"hotpotqa"` |
| `split` | ECA 规范名：优先 `"validation"`（若源文件叫 `dev`，此处仍写 `validation`，并另存 `source_split`） |
| `source_split` | 原始文件/发布名（如 `dev` / `validation`） |
| `source` | 数据来源标签（1B 选定后写入，如 `flashrag` / `hotpotqa_official` / `hf:...`） |
| `raw_id` | 源数据中的稳定序号或 `_id` 字符串 |
| `level` | 若有：`easy` / `medium` / `hard` |
| `type` | 若有：`bridge` / `comparison` 等 |

### 完整示例

```json
{
  "sample_id": "hotpotqa_distractor_validation_5a8b57f25542995d1e6f1371",
  "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
  "gold_answers": ["Arthur's Magazine"],
  "supporting_facts": [
    {"title": "Arthur's Magazine", "sentence_id": 0},
    {"title": "First for Women", "sentence_id": 0}
  ],
  "contexts": [
    {
      "document_id": "hotpotqa_distractor_validation_5a8b57f25542995d1e6f1371_ctx_0",
      "title": "Arthur's Magazine",
      "sentences": ["Arthur's Magazine was an American literary periodical published in Philadelphia.", "..."],
      "text": "Arthur's Magazine was an American literary periodical published in Philadelphia. ..."
    }
  ],
  "metadata": {
    "dataset": "hotpotqa",
    "split": "validation",
    "source_split": "validation",
    "source": "hf:hotpotqa/hotpot_qa",
    "config": "distractor",
    "raw_id": "5a8b57f25542995d1e6f1371",
    "level": "hard",
    "type": "comparison"
  }
}
```

---

## 3. sample_id Contract

| 规则 | 要求 |
|------|------|
| 构造 | `hotpotqa_{config}_{split}_{raw_id}` |
| `config` | HF 配置名，Phase 1 固定 `distractor` |
| `split` | ECA 规范名：`validation` |
| `raw_id` | HotpotQA 原始稳定字符串 ID（写入 `metadata.raw_id`） |
| 跨方法 | Direct / Oracle / Candidate-BM25 / Full-Corpus **必须相同** |
| 禁止 | 含 method 名；运行时随机 UUID；用“当前行号”当唯一身份 |

示例：`hotpotqa_distractor_validation_5a8b57f25542995d1e6f1371`

**子集抽样（Phase 1B）：** 对完整 validation 用固定 `seed=42` 做确定性 permutation，再取前缀  
`8 ⊂ 50 ⊂ 200`。子集成员由 `sample_id` 列表定义，不依赖原始文件行序。

### `trace_id`（不属于 eval sample）

生成/rollout 侧才出现，允许带 method / run / rollout：

```text
{sample_id}_{method_id}_{run_tag}_{rollout_idx}
例：hotpotqa_validation_000123_direct_r0_0
```

`sample_id` 永不因 method 改变。

---

## 4. Gold Answer Contract

| 规则 | 说明 |
|------|------|
| 类型 | 永远 `List[str]` |
| 单答案 | `["唯一答案"]`，长度 1 |
| 转换阶段 | **不做** `normalize_answer` |
| 归一化 | 仅在 `src/eval/metrics.py` 打分时发生 |
| 原文 | 不改写、不截断、不合并同义答案（除非源数据本身提供多 gold） |

---

## 5. Supporting Fact Contract

Gold evidence **身份** = `(title, sentence_id)`，不是自由文本匹配。

| 用途 | 如何用 |
|------|--------|
| supporting evidence coverage | Top-K 文档 title 是否覆盖 gold titles；进一步可查 sentence 是否落在返回文档内 |
| evidence evaluation（后续） | 模型选出的 evidence 是否命中 gold `(title, sentence_id)` |
| citation evaluation（后续） | 引用是否指向真实 Document 且与 gold 可对齐 |

可选字段 `sentence` 仅便于人工验收；缺失不视为合同违约。  
`sentence_id` 必须与对应 `contexts[].sentences` 的下标一致（0-based）。

---

## 6. Context / Document Contract

HotpotQA 一条 context ≈ 一个 ECA 文档级对象。

| 规则 | 拍板 |
|------|------|
| 粒度 | **每个 title 一条 context / 一个 Document**（不是每个 sentence 一篇） |
| `sentences` | 保留原始顺序与边界 |
| `text` | `" ".join(sentences)`（单空格拼接；converter 实现时写死并测一轮） |
| observation | 可只序列化 `title`+`text`，但 **eval JSONL 与 cache 不得丢 `sentences`** |
| 样本内唯一 | 同一 `sample_id` 下 `document_id` 唯一；`title` 在 HotpotQA 题包内通常唯一，若冲突以 `document_id` 为准并在 metadata 记录 |

### `document_id`（题包 context）

```text
{sample_id}_ctx_{k}
```

`k` = 该样本 `contexts` 列表中的 0-based 顺序。

示例：`hotpotqa_distractor_validation_5a8b57f25542995d1e6f1371_ctx_0`

### Full-Corpus 文档 ID（预告，本阶段不实现）

全语料命中使用**语料自身稳定 ID**（如 wiki dump 行号 / FlashRAG corpus `id`），写入 retrieval cache；**不要**强行改写成 `_ctx_` 形式。  
在 `retriever.corpus_id` / `index_id` 中记录语料与索引版本。

### 进入 `TraceRecord.Document` 时

| eval / cache 字段 | `Document` 字段 |
|-------------------|-----------------|
| `document_id` | `document_id` |
| `title` | `title` |
| `text` | `text` |
| `rank` / `score`（若来自检索） | `rank` / `score` |
| `sentences` | 放 `Document.metadata["sentences"]`（生成可不用，评测/验收保留） |

---

## 7. Oracle RAG Contract

| 项 | 拍板 |
|----|------|
| 选取规则 | `supporting_facts` 中出现的 **unique titles** |
| 文档内容 | 上述 title 在 `contexts` 中对应的**完整文档**（整份 `sentences`/`text`） |
| 禁止 | 只塞 gold 那几句 sentence 当作 Oracle（除非未来显式开 `oracle_granularity=sentences` 实验） |
| 禁止 | 把题目自带全部 candidate contexts 当作 Oracle |
| 排序 | 按 title 在 `supporting_facts` 中首次出现顺序；同 title 不重复 |
| 目的 | 测「正确 supporting documents 已到手」时的生成上界 |

Oracle **可以不经过** `retrieval_results.jsonl`（无检索器），但若落盘为 cache 以便统一 generation 入口，则：

```json
{
  "sample_id": "hotpotqa_validation_000123",
  "query": "...",
  "retriever": {
    "name": "oracle",
    "scope": "oracle_supporting_docs",
    "top_k": null
  },
  "documents": [ "...完整金标文档..." ]
}
```

`scope` 必须为 `oracle_supporting_docs`，不得标成 `candidate` / `full_corpus`。

---

## 8. Candidate-BM25 Contract

| 项 | 拍板 |
|----|------|
| 范围 | **仅**当前 sample 的 `contexts` |
| query | 默认 = 原始 `question`（与其它方法相同） |
| 输出 | 带 `document_id` / `title` / `text` / `rank` / `score` |
| `retriever.name` | `"bm25s"` |
| `retriever.scope` | `"candidate"` |
| 角色 | **diagnostic retrieval** |
| 表述 | 禁止称为 full-corpus / 真实 Wiki search |

---

## 9. Full-Corpus BM25 Contract

| 项 | 拍板 |
|----|------|
| query | 与 Candidate 相同的 `question` |
| 范围 | 预先固定的 Wikipedia corpus + BM25s index |
| 版本 | 必须记录 `corpus_id` 与 `index_id`（路径或内容哈希/发布名） |
| 输出结构 | 与 Candidate **同一 schema**（§10） |
| `retriever.scope` | `"full_corpus"` |
| 本阶段 | **不实现、不下载、不建索引** |

未就绪时实验表可写 `pending`，不得用 Candidate 结果冒充本行。

---

## 10. `retrieval_results.jsonl` Schema

一行一个 sample 的一次检索结果。

### 顶层

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `sample_id` | `str` | ✅ | 与 eval sample 一致 |
| `query` | `str` | ✅ | 实际用于检索的字符串 |
| `retriever` | `obj` | ✅ | 见下 |
| `documents` | `List[obj]` | ✅ | 按 `rank` 升序；可为空列表但需显式写出 |

### `retriever` 对象

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | `str` | ✅ | 如 `bm25s` / `oracle` |
| `scope` | `str` | ✅ | `candidate` \| `full_corpus` \| `oracle_supporting_docs` |
| `top_k` | `int` \| `null` | ✅ | Oracle 可用 `null` |
| `corpus_id` | `str` | full_corpus 必填 | 语料版本标识 |
| `index_id` | `str` | full_corpus 必填 | 索引版本标识 |
| `config` | `obj` | 可选 | 额外超参（k1、b 等） |

### `documents[]` 元素

| 字段 | 类型 | 必填 |
|------|------|------|
| `document_id` | `str` | ✅ |
| `title` | `str` | ✅ |
| `text` | `str` | ✅ |
| `rank` | `int` | ✅（从 1 开始） |
| `score` | `float` \| `null` | ✅（Oracle 可用 `null`） |
| `metadata` | `obj` | 可选（可含 `sentences`） |

### 示例（Candidate）

```json
{
  "sample_id": "hotpotqa_validation_000123",
  "query": "Which magazine was started first, Arthur's Magazine or First for Women?",
  "retriever": {
    "name": "bm25s",
    "scope": "candidate",
    "top_k": 5
  },
  "documents": [
    {
      "document_id": "hotpotqa_validation_000123_ctx_0",
      "title": "Arthur's Magazine",
      "text": "Arthur's Magazine was an American literary periodical...",
      "rank": 1,
      "score": 12.73
    }
  ]
}
```

---

## 11. Retrieval Cache Rules

| 规则 | 说明 |
|------|------|
| 边界 | cache = Retriever → Generator 的稳定接口 |
| 生成侧 | **不得**在 generator 进程内静默重跑检索（除非新开 retrieval 实验并写新 cache） |
| 复用 | 同一 `retriever` 配置 + 同一 eval subset → 可复用同一 cache 文件 |
| 禁止内容 | 模型答案、`gold_answers` 改写、reward、trace steps |
| 可含 | `sample_id`、`query`、检索文档、retriever 配置/版本 |
| 落盘位置（预告） | `results/{run_name}/retrieval_results.jsonl` 或 `data/processed/retrieval/...`；具体路径在实现 retrieval runner 时选定，本阶段不定代码 |

---

## 12. Mapping to TraceRecord

| 来源 | TraceRecord |
|------|-------------|
| eval.`question` | `question` |
| eval.`gold_answers` | `gold_answers` |
| eval.`sample_id` | `sample_id`（不变） |
| cache.`documents` 或 Oracle 文档 | `documents: Document[]` |
| 格式化后的检索文本 | `steps` 中 `observation`（`loss_mask=False`） |
| eval.`supporting_facts` | `metadata["supporting_facts"]`（及后续 gold evidence 评测输入） |
| method / run | 仅进入 `trace_id` 与 `metadata`，不改 `sample_id` |

**retrieval cache 本身不是 TraceRecord。**  
TraceRecord 在 generation 之后产生；cache 可在无 GPU 时单独生成。

Direct：无 cache，无 `documents` / 无 observation。  
Oracle / BM25：generation 读 cache（或等价 Oracle 构造）→ 填 `documents` + observation → answer。

---

## 13. Fair Comparison Contract

四条方法必须同时满足：

| 维度 | 要求 |
|------|------|
| 样本集 | 同一 subset 文件（同一 `sample_id` 列表） |
| ID | 同一 `sample_id` |
| 模型 | 同一 generator checkpoint（Phase 1：Qwen2.5-3B-Instruct） |
| 生成配置 | 同一 decoding / max_new_tokens / seed 策略（差异写入 `run_info` 则必须声明不公平） |
| 答案指标 | 同一 `exact_match` / `token_f1` / `format_valid` |
| 唯一差异 | 注入的检索上下文（无 / Oracle docs / Candidate Top-K / Full-Corpus Top-K） |

禁止：不同 baseline 使用不同抽题集合。

---

## 14. Dataset Scale Plan

| N | 用途 |
|---|------|
| 8 | pipeline validation（真实 HotpotQA，非手工题） |
| 50 | first trend |
| 200 | formal small baseline |

扩大 N 时：**在同一有序 `sample_id` 列表上取前缀**（前 8 ⊂ 前 50 ⊂ 前 200），禁止按方法分别重采样。

子集清单文件（Phase 1B 产出，本阶段不创建）预告命名：

```text
data/eval/hotpotqa_validation_ids_n200.txt
```

或等价 JSONL；实现时二选一，但必须可复现。

---

## 15. Validation Rules for Future Converter

未来 converter / validator 至少检查：

| # | 检查 |
|---|------|
| 1 | `sample_id` 非空；文件内唯一 |
| 2 | `question` 非空 |
| 3 | `gold_answers` 为非空 list，且元素均为非空 `str` |
| 4 | 样本内 `contexts[].document_id` 唯一 |
| 5 | 每个 `supporting_facts[].title` 能在 `contexts[].title` 中找到（当前 HotpotQA distractor setting 下应成立；找不到 → 记 invalid） |
| 6 | `sentence_id >= 0` 且 `< len(对应 title 的 sentences)` |
| 7 | `text` 与 `" ".join(sentences)` 一致（或由 converter 重算覆盖） |
| 8 | **不**在转换阶段调用 `normalize_answer` |

失败行写入本次 run 的 invalid 清单，不得静默丢弃且继续宣称全集有效。

---

## 16. Non-Goals

本阶段（1A1）不：

- 下载 HotpotQA / Wikipedia  
- 实现 converter / retriever / index  
- 创建任何 eval JSONL 或 retrieval cache 文件  
- 运行 generation / 占用 GPU  
- 修改 `trace_schema.py` / `metrics.py`  
- 定义 reward  
- 进入 SFT / RL / Phase 1B 实现  

---

## 17. Next Phase（仅建议，待确认）

**Phase 1B（建议拆步）：**

1. 选定 HotpotQA 数据来源与具体 split（FlashRAG 预处理 vs 官方/HF 原始）  
2. 获取原始数据  
3. 按本合同实现唯一 converter  
4. 先产出 **8 条**真实 eval subset  
5. 人工验收 8 条（字段、Oracle titles、sentence_id）  

然后才进入 Phase 1C：8 条真实 HotpotQA → Qwen Direct baseline。

**本文件完成后停止；不开始 Phase 1B。**

---

## 拍板摘要

```text
sample_id:     hotpotqa_{config}_{split}_{raw_id}
               例 hotpotqa_distractor_validation_5a8b57f25542995d1e6f1371
source:        hf:hotpotqa/hotpot_qa  config=distractor  split=validation
gold_answers:  List[str]，转换期不 normalize
supporting:    (title, sentence_id) 为金标身份
context doc:   每 title 一篇；id = {sample_id}_ctx_{k}；保留 sentences + 派生 text
Oracle:        supporting unique titles 的完整文档；非全候选、非仅金句
BM25 scope:    candidate | full_corpus（缓存内强制标记）
cache:         retrieval_results.jsonl → 再生成 → TraceRecord
fairness:      同 subset / 同 sample_id / 同模型 / 同指标
scale:         seed=42 permutation 后 8 ⊂ 50 ⊂ 200 前缀子集
```

*文档版本：Phase 1A1 / 2026-08-07*
