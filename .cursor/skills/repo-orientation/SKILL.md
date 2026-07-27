---
name: repo-orientation
description: >-
  Orients the agent to this DeepResearch/Search-R1 repo structure, locates
  modules, and plans changes. Use when starting a new task, exploring the
  codebase, or before any code edit in this project.
---

# Repo Orientation Skill

## When to Use

- First task of a session
- Before creating new files or modules
- When unsure where data / agent / reward / eval live

## Workflow

1. Inspect top-level tree (max depth 2). Do not `find` the entire repo.
2. Map existing folders to pipeline stages:

   | Folder | Stage |
   |--------|-------|
   | `data/` | raw → processed → splits |
   | `agent/` | ReAct loop, parser, prompts |
   | `tools/` | BM25 search, read, cache |
   | `reward/` | answer, format, evidence, cost |
   | `training/sft/`, `training/rl/` | SFT & GRPO |
   | `eval/` | metrics, run_eval |
   | `configs/` | yaml configs |
   | `scripts/` | shell entrypoints |
   | `outputs/` | **only** place for run artifacts |

3. Read `README.md`, `docs/PROJECT_MAP.md`, and relevant config if present.
4. Update `docs/PROJECT_MAP.md` only when structure changes (not every task).
5. Extend existing modules — never fork a second parser/trainer/eval.

## Anti-Patterns (Do Not Repeat)

| Trap | Fix |
|------|-----|
| Agent creates `src/`, `lib/`, `core/` parallel to planned layout | Use planned dirs only |
| Scattered `test_output/`, `results/`, `runs/` | Consolidate to `outputs/{run_name}/` |
| Duplicate entrypoints (`train.py`, `train_v2.py`) | One script + `--debug` flag |
| Reading entire codebase into context | Read only files needed for current task |

## Done Criteria

- Can answer: where does data enter, where is inference, where is reward, where do eval results go?
- No new top-level directories created without user approval.
