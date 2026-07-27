---
name: search-tool-protocol
description: >-
  Implements ReAct/Search-R1 XML action format, local BM25 search tool,
  observation injection, parser, and search_count metrics. Use when building
  agent loop, tool wrappers, or trace logging.
---

# Search Tool Protocol Skill

## When to Use

- Implementing `agent/react_loop.py`, `agent/parser.py`
- Building `tools/search_bm25.py`, mock search for debug
- Fixing format errors or search_count mismatches

## Action Format (Fixed)

```text
<think>...</think>
<search>query text</search>
<observe>...</observe>          # injected by environment, not model-generated
<think>...</think>
<final>answer + evidence</final>
```

Alternate tag `<answer>` acceptable in parser if consistently mapped to `final`.

## Implementation Requirements

1. **Parser** extracts search queries robustly; handles malformed / partial tags
2. **Search tool** returns deterministic mock results in `--debug` mode (no web)
3. Count every `<search>` as one search action
4. **Cache** repeated queries within a trajectory (`tools/cache.py`)
5. **Limits:** top_k 3–5, max searches 3–5, snippet 200–400 tokens

## Trace Log Schema (per turn)

Save to `outputs/{run_name}/trajectories.jsonl`:

```json
{
  "id": "string",
  "question": "string",
  "model_output": "string",
  "parsed_searches": ["query"],
  "observations": ["text"],
  "final_answer": "string",
  "search_count": 0,
  "parser_errors": []
}
```

## Common Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| Real web in debug | Slow, non-reproducible, blocked | Mock BM25 only until eval phase |
| Tool output as list breaks chat template | Template render error | Pass tool results as plain strings |
| Observation in policy gradient | Wrong RL signal | Retrieved token masking — obs not in loss |
| Parser too strict | >5% format errors on smoke | Relax + log errors, fix prompt |
| No stop condition | Infinite search loop | Hard cap max_turns; penalize in reward |

## Stop Conditions

- Parser fails on >5% of smoke examples → fix format before training
- Output has unclosed tags → count as format error, do not silently truncate

## Done Criteria

- Unit tests: normal trajectory, malformed tags, no-search direct final, max-turn exceeded
- Smoke run on ≤8 questions produces valid `trajectories.jsonl`
- No network calls in `--debug` mode
