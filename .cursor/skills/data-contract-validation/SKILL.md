---
name: data-contract-validation
description: >-
  Defines and validates JSONL schemas for SFT samples, search trajectories,
  RL rollouts, and eval files. Use when creating, converting, filtering, or
  debugging datasets in this DeepResearch project.
---

# Data Contract and Validation Skill

## When to Use

- Converting DeepResearch-9K / HotpotQA / 2Wiki to project format
- Building SFT JSONL or trajectory logs
- Debugging "invalid sample" or length overflow errors

## Schemas

### SFT Sample

```json
{
  "id": "string",
  "question": "string",
  "messages": [
    {"role": "system", "content": "string"},
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ],
  "answer": "string",
  "source": "string",
  "meta": {}
}
```

### Search Trajectory

```json
{
  "id": "string",
  "question": "string",
  "trajectory": [
    {"type": "thought", "text": "string"},
    {"type": "search", "query": "string"},
    {"type": "observation", "text": "string"},
    {"type": "final", "text": "string"}
  ],
  "gold_answer": "string",
  "search_count": 0,
  "meta": {}
}
```

## Required Checks

1. JSONL parseable line-by-line
2. Required fields present; empty question/answer forbidden
3. `search_count` matches actual `<search>` actions in trajectory
4. Print length stats: question, trajectory, observation tokens/chars
5. Invalid rows → `outputs/{run_name}/invalid_rows.jsonl` (not repo root)

## DeepResearch-9K Filtering Rules

**Keep:** has question, final answer, search trajectory, search_count 1–6, extractable answer, L1/L2/L3 mix.

**Drop:** no final, no search, empty query, trajectory too long for context, mixed-language garbage.

**Curriculum ratios:**
- SFT: L1 40% / L2 40% / L3 20%
- RL: L1 20% / L2 40% / L3 40%

## Common Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| Duplicate IDs across sources | Eval leakage | Prefix ids: `hotpot_`, `dr9k_` |
| Observation in SFT loss | Model memorizes retrieval text | Mask observation tokens in loss |
| Unbounded JSONL writes | Disk fills | `--max-samples` in all converters |
| Raw data overwritten | Irrecoverable | Write only to `data/processed/` |

## Done Criteria

- Validator script exists under `scripts/` or `data/`
- Runs on ≤20 sample smoke file; prints pass/fail counts
- No new data files outside `data/` or `outputs/`
