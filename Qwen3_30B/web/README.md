# web（暂不执行）

Web 线在最终 30B checkpoint 冻结后启动。原则是只替换 tool backend，不改 Agent policy：

```text
<search>query</search>
  → WebSearchAdapter
  → search API + page reader
  → 统一 observation
  → 现有 Agent loop
```

计划交付 BrowseComp-Plus、BrowseComp / GAIA-text 评测，以及展示 trace、evidence、answer 和 sources 的 Web Demo。

