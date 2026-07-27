# 论文库 — DeepResearch-Agent-RL

10 份精读资料，对应 `.cursor/skills/theory-paper-reading/`。

## PDF 清单

| # | 文件 | 论文 | arXiv |
|---|------|------|-------|
| 1 | [01-RAG.pdf](./01-RAG.pdf) | Retrieval-Augmented Generation | [2005.11401](https://arxiv.org/abs/2005.11401) |
| 2 | [02-ReAct.pdf](./02-ReAct.pdf) | ReAct: Synergizing Reasoning and Acting | [2210.03629](https://arxiv.org/abs/2210.03629) |
| 3 | [03-Toolformer.pdf](./03-Toolformer.pdf) | Toolformer | [2302.04761](https://arxiv.org/abs/2302.04761) |
| 4 | [04-Search-R1.pdf](./04-Search-R1.pdf) | Search-R1 | [2503.09516](https://arxiv.org/abs/2503.09516) |
| 5 | [05-WebDancer.pdf](./05-WebDancer.pdf) | WebDancer | [2505.22648](https://arxiv.org/abs/2505.22648) |
| 6 | [06-DeepResearch-9K.pdf](./06-DeepResearch-9K.pdf) | DeepResearch-9K | [2603.01152](https://arxiv.org/abs/2603.01152) |
| 7 | [07-verl-agentic-rl.md](./07-verl-agentic-rl.md) | veRL 文档链接 | — |
| 8 | [08-R-Search.pdf](./08-R-Search.pdf) | R-Search | [2506.04185](https://arxiv.org/abs/2506.04185) |
| 9 | [09-R1-Searcher++.pdf](./09-R1-Searcher++.pdf) | R1-Searcher++ | [2505.17005](https://arxiv.org/abs/2505.17005) |
| 10 | [10-open-deep-research.md](./10-open-deep-research.md) | LangChain 产品参考 | GitHub |

## 来源说明

- `04-Search-R1.pdf` 复制自 `/data1/hcc/agentic-rec/papers/`
- 其余 PDF 从 arXiv 下载（2026-07-03）
- Windows 路径 `C:\Users\HanChengcheng\Desktop\科研\agentic-RL` 在 Linux 服务器不可直接访问；若本地有标注版 PDF，请 scp 覆盖对应文件

## 同步本地论文（可选）

```bash
# 在 Windows 本地执行，上传到服务器
scp "C:/Users/HanChengcheng/Desktop/科研/agentic-RL/*.pdf" \
  hanchengcheng@<server>:/data1/hcc/deepresearch/papers/
```

## 阅读方式

对 Cursor Agent 说：

```text
使用 theory-paper-reading skill，带我读资料 2 ReAct。
每次只讲一个概念，最后给我 1 道自测题。
```
