# 资料 10：LangChain Open Deep Research（产品参考）

> **只当产品形态参考，不当训练主线。** 禁止做成 LangChain 应用项目。

## 仓库

- https://github.com/langchain-ai/open_deep_research

## 看什么

1. **DeepResearch 产品最终长什么样** — 输入问题 → 多步搜索 → 结构化报告
2. **报告结构** — 摘要、分节、引用、结论
3. **工具串联** — search → read → summarize → synthesize

## 不做什么

- 不用 LangChain 框架作为本项目训练底座
- 不复制其 MCP / 多 provider 架构
- 不把 LangGraph workflow 当 RL 替代

## 对本项目的启发

| 产品特性 | 本项目简化版 |
|----------|-------------|
| 多轮 web search | `search(query)` + BM25 |
| 读全文 | `read(doc_id)` |
| 报告生成 | `<final>` 含 answer + evidence |
| 成本控制 | search cost penalty |

## 读完后回答

1. 产品和训练系统的边界在哪？
2. 为什么我们要做 RL 而不是只做 workflow？
