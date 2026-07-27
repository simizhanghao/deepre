---
name: theory-paper-reading
description: >-
  Guides structured reading of the 10 core papers for DeepResearch-Agent-RL.
  Maps 6 theory layers (RAG, ReAct, Tool Use, Search-R1, WebDancer,
  DeepResearch-9K) to code modules. Use when the user asks to read papers,
  learn theory, understand concepts, or before implementing a new module.
---

# Theory & Paper Reading Skill

## When to Use

- User says: "读论文", "讲清楚 X", "这层理论对应什么代码"
- Before implementing: `react_loop`, `search`, `reward`, `sft`, `grpo`
- Interview prep: explain why this project is designed this way

## Agent Behavior (Strict)

1. **One paper / one concept per session** — do not dump all 10 at once
2. **Read from local PDF** in `papers/` when available; cite section/page if possible
3. **Always end with**: 3 bullet takeaways + 1 code file mapping + 1 quiz question
4. **Do not expand reading list** beyond the 10 materials below
5. **Do not start coding** during a reading session unless user asks

## Six Theory Layers (Project Scope Only)

| Layer | Core Idea | Project Mapping |
|-------|-----------|-----------------|
| 1. RAG | parametric + non-parametric memory; query→retriever→docs→generator | Baseline 1: `eval/one_shot_rag.py` |
| 2. ReAct | Thought→Action→Observation loop | Baseline 2: `agent/react_loop.py`, `parser.py` |
| 3. Tool Use | when/what/how to call tools | `tools/search_bm25.py`, `tools/read_doc.py` |
| 4. Search-R1 | multi-turn search RL + retrieved token masking + outcome reward | Version 2: `training/rl/`, `reward/` |
| 5. WebDancer | 4-stage: data→trajectory→SFT→RL | Overall pipeline |
| 6. DeepResearch-9K/R1 | SFT data + training framework | `data/`, `training/sft/` |

## Reading Order (10 Materials — Do Not Expand)

| # | Material | Time | PDF / Ref |
|---|----------|------|-----------|
| 1 | RAG | 1–2h | `papers/01-RAG.pdf` |
| 2 | ReAct | must-read | `papers/02-ReAct.pdf` |
| 3 | Toolformer | concepts only | `papers/03-Toolformer.pdf` |
| 4 | Search-R1 | **精读** | `papers/04-Search-R1.pdf` |
| 5 | WebDancer | 4-stage only | `papers/05-WebDancer.pdf` |
| 6 | DeepResearch-9K/R1 | data + framework | `papers/06-DeepResearch-9K.pdf` + repo |
| 7 | veRL Agentic RL | framework docs | `papers/07-verl-agentic-rl.md` |
| 8 | R-Search | multi-reward | `papers/08-R-Search.pdf` |
| 9 | R1-Searcher++ | cost control | `papers/09-R1-Searcher++.pdf` |
| 10 | LangChain Open Deep Research | product ref only | `papers/10-open-deep-research.md` |

Full per-paper checklists: [references/reading-map.md](references/reading-map.md)

## Guided Reading Workflow

When user picks paper N:

```
Step 1: State "今天要读：资料 N — {title}"
Step 2: List 3–5 questions user must answer after reading (from reading-map.md)
Step 3: Walk through paper section by section (concise, not full translation)
Step 4: Map to code modules (from references/code-mapping.md)
Step 5: Mini quiz — ask user 1 question; wait for answer
Step 6: Write summary to docs/reading-notes/{NN}-{slug}.md (only if user confirms)
```

## ReAct Loop (Must Know by Heart)

```python
while not done:
    model_output = model(prompt + history)
    action = parse_action(model_output)
    observation = tool(action)
    history.append(model_output, observation)
```

## Search-R1 Five Questions (Paper 4)

After reading Search-R1, user must answer:

1. Action format是什么？
2. Retrieved documents 如何插入 context？
3. 为什么需要 retrieved token masking？
4. Reward 怎么算？
5. 如何评估 Base / RAG / RL？

## Stop Conditions

- User confused → simplify to one layer, one diagram, one example
- User wants to skip → allow skip only for Toolformer formulas & LangChain product details
- User asks unrelated papers → redirect to reading-map.md list

## Done Criteria (Per Session)

- User can explain concept in own words
- User knows which code file will implement it
- User knows what this layer does NOT cover (limitations)
