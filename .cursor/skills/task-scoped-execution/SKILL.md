---
name: task-scoped-execution
description: >-
  Enforces user-controlled, single-task execution discipline for the
  Evidence-Cost-Aware Deep Research Agent project. Use at the start of EVERY
  work request to scope the task, restrict file writes, define acceptance
  criteria, and stop for user confirmation before the next step.
---

# Task-Scoped Execution Skill

The user controls the roadmap. The agent executes ONE small, verifiable task at
a time and never plans or advances phases on its own.

## Skills First (Hard — every reply)

Before answering the user or calling other tools for work:

1. Re-read this file with the Read tool (do not rely on memory alone).
2. Re-read `.cursor/skills/artifact-naming-and-cleanup/SKILL.md`.
3. Re-read any other matching project skill for the request.
4. Then apply the 5-point contract below.

## Contract for Every Task (5 Points)

Before doing anything, restate and confirm:

1. **本轮目标** — one concrete task, nothing more
2. **允许 / 禁止改的文件** — default read-only; writes only if explicitly allowed
3. **完成后必须输出什么** — the exact deliverable (report, schema, notes...)
4. **如何验证成功** — the acceptance criteria
5. **停下等确认** — do NOT decide or start the next step

If any of the 5 is unclear, ask before acting.

## Default Restrictions

Unless the user explicitly authorizes otherwise, do NOT:
- create, modify, or delete files
- create new directories or scaffold structure
- install dependencies
- clone repositories
- run training or long experiments
- write RL reward / training code ahead of its phase

## Phase Order (Never Skip)

```
Phase 0 scaffold + trace schema
Phase 1 baselines (no training)
Phase 2 SFT trajectory construction
Phase 3 SFT cold-start (3B first)
Phase 4 GRPO / RLOO (not PPO)
Phase 5 ablation + packaging
```

Gate: do not enter RL until baseline, retrieval, eval, and trace are verified.

## Hard "Do Not" List (Current Stage)

- No PPO, no 7B first, no real web search
- No full WebDancer / browser agent
- No auto-generating project structure
- No dumping many scripts into one file

## Anti-Sprawl Rules

- Prefer editing existing files over creating new ones
- Max 1 new file per task unless the user asks for more
- All experiment artifacts go under the project's designated results/outputs dir
- No `tmp/`, `backup/`, `old/`, `copy_of_*`, `*_v2` files

## End-of-Task Report (Always)

- files changed (or "read-only, none")
- commands run
- results / findings
- how it meets the acceptance criteria
- proposed next step — **as a suggestion only, awaiting user approval**
- if this task ran train/eval/long jobs: also follow
  `artifact-naming-and-cleanup` (cleanup commands in the same turn)

## Done Criteria

- Exactly one task addressed
- No unauthorized writes
- Deliverable + verification provided
- Explicitly stopped for user confirmation
