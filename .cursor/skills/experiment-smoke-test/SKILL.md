---
name: experiment-smoke-test
description: >-
  Runs tiny reproducible smoke tests before any training, eval, parser, reward,
  or data job. Use before every GPU run or after changing training/inference/
  reward code in this project.
---

# Experiment Smoke Test Skill

## When to Use

- **Always** before first GPU job of the day
- After changing parser, reward, agent loop, or data converter
- When user says "quick test" or "verify pipeline"

## Required CLI Flags (Every Runnable Script)

```
--config PATH
--seed INT
--output-dir PATH      # must be under outputs/
--max-samples INT      # smoke: 2–8
--debug                # enables mock search, minimal steps
```

## Smoke Checklist (In Order)

```
[ ] 1. Data validator on ≤20 rows
[ ] 2. Parser unit tests (pytest or inline)
[ ] 3. Reward unit tests on 5 handcrafted trajectories
[ ] 4. Agent loop: ≤8 questions, mock search, write trajectories.jsonl
[ ] 5. Tiny SFT or tiny eval (≤8 samples, ≤5 steps)
[ ] 6. Append command + result to docs/RUN_LOG.md (one line per run)
```

## Output Discipline

- Smoke output dir: `outputs/smoke_{component}_{YYYYMMDD}/`
- Do **not** create `test_results/`, `debug_out/`, or timestamped dirs at repo root
- After successful smoke, do not re-run same smoke unless code changed
- Never launch multi-hour GPU jobs without: successful smoke + user approval

## Common Smoke Failures

| Failure | Likely Cause | Next Step |
|---------|--------------|-----------|
| Parser 100% fail | Wrong tag format / prompt | Fix parser + prompt together |
| Reward all zero | Answer extract broken | Fix final answer parser first |
| OOM on 8 samples | Full finetune / long context | Reduce max_length, use LoRA |
| Hang >10 min | Real web search / SGLang | Switch to mock + vLLM |
| Empty trajectories.jsonl | Agent loop exception swallowed | Check logs in output dir |

## Done Criteria

- One command reproduces "pipeline is alive"
- Output confined to single `outputs/smoke_*` directory
- RUN_LOG.md updated with exact command and pass/fail
