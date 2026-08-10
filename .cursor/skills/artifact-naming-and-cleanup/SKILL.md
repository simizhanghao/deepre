---
name: artifact-naming-and-cleanup
description: >-
  Enforces numbered English artifact naming and mandatory post-run GPU/process
  cleanup prompts for the DeepResearch ECA project. Use whenever creating or
  renaming files under outputs/, results/, logs/, data/, configs/, scripts/,
  src/; after any training, eval, tmux, docker, Ray, SGLang, or long script run;
  or when the user asks about naming, cleanup, or killing leftover processes.
---

# Artifact Naming and Post-Run Cleanup

Hard skill. Do not invent `phase*` / `*_v0` / timestamp-soup names for new
artifacts. After every train/eval/long job, **immediately** give the user
copy-paste cleanup commands — do not wait for them to ask.

## When to Use

- Creating any new path under `outputs/`, `results/`, `logs/`, `data/`, `configs/`
- Adding scripts or reward modules under `scripts/`, `src/`
- After GRPO / SFT / eval / boundary bootstrap / tmux / docker jobs finish or fail
- User mentions 命名、改名、清理进程、杀进程、GPU 占用

## Naming Rules (New Artifacts Only)

### Directory prefixes

| Area | Pattern | Examples |
|------|---------|----------|
| `outputs/` top models | `NN_descriptive_english` | `00_sft_v1_merged` |
| `outputs/rl/` | `NN_role_descriptive` | `01_ckpt_grpo_evidence_fsdp`, `03_hf_evidence_step400`, `04_table_search_boundary`, `06_ckpt_grpo_boundary` |
| `results/` | `NN_verb_object` | `09_audit_grpo_evidence_step400`, `10_eval_grpo_evidence_val200` |
| `logs/` | `NN_role_*.log` (+ `_latest` symlink) | `01_grpo_evidence_train.log`, `06_grpo_boundary_latest.log` |
| `data/rl/` | short English, no phase | `train_smoke_128`, `calib_cost_lambda_512` |
| `configs/sft/` | `sft_v1_*.yaml` / `dataset_info_sft_v1.json` | |
| `configs/rl/` | role yaml, no phase | `grpo_smoke128.yaml`, `candidate_bm25_tool.yaml` |
| `src/rl/` rewards | `rewards_<role>.py` | `rewards_evidence.py`, `rewards_boundary.py` |
| `src/rl/` metrics | `grpo_metrics.py` | not `phase3*_metrics.py` |
| `scripts/` | verb_object English | `run_grpo_boundary.sh`, `run_eval_val200_gen.sh`, `build_calib_cost_lambda.py` |

### Forbidden for **new** names

- `phase*`, `3d1b_ls*`, `*_v0_*` as the primary name
- `grpo_grpo_*`, double prefixes
- Long timestamp dumps as the **directory** name (ok inside a numbered parent)
- Parallel `tmp/`, `backup/`, `*_v2` directories
- Writing new experiment trees outside `results/` / `outputs/` / `logs/`

### Numbering

- Pick the next free `NN_` in that folder; do not reuse closed numbers for unrelated work.
- Prefer stable names for “current” tables/models (`boundary_latest.json`, `p_int_latest.json` symlinks).
- Subdirs under a numbered parent may be short English (`gen_sft`, `gen_3b`, `gen_3c`, `by_step.csv`).

### Code modules

- Extend existing files before creating new ones.
- Max 1 new file per task unless user asks for more (see task-scoped-execution).
- Never edit `external/`.

## Post-Run Cleanup (Mandatory)

Whenever a training, eval, tmux, docker, Ray, SGLang, retrieval server, or other
long-running job **finishes, fails, or is stopped**, the agent MUST end the
turn with:

1. One-line status (pass/fail + where outputs landed)
2. **Immediate** cleanup prompt + copy-paste commands (user runs them)

### Default cleanup block (adapt ports/session names)

```bash
# 1) Who holds GPUs?
nvidia-smi

# 2) Stop tmux train session if still up (example names)
tmux ls 2>/dev/null
# tmux kill-session -t eca-grpo-3d2b
# tmux kill-session -t eca-grpo-3c

# 3) Inside eca-verl: Ray / SGLang leftovers
docker exec eca-verl bash -lc 'ray stop --force 2>/dev/null; pkill -9 -f sglang.launch_server 2>/dev/null; pkill -9 -f launch_grpo_main.py 2>/dev/null; exit 0'

# 4) Host: stray python train/eval / retriever
pkill -f 'scripts/run_grpo_|scripts/launch_grpo_main|scripts/run_agent_rollout|build_search_boundary_table' 2>/dev/null || true
# Retriever on :8001 (only if no longer needed)
# pkill -f 'start_candidate_retrieval_server.py' 2>/dev/null || true

# 5) Confirm GPU free
nvidia-smi
```

### Rules for the cleanup prompt

- Always show the block after train/eval; do not bury it.
- Customize session name / ports from the job just run.
- Do **not** auto-run destructive kills unless the user asked and rules allow shell; prefer pasteable commands (this repo: user executes commands).
- If a job is still supposed to run, say so and **skip** kill commands; only offer `nvidia-smi` / `tmux ls` status.

## Interaction with Other Skills

- `task-scoped-execution`: still one task + stop for confirm.
- `experiment-smoke-test`: smoke runs also get cleanup prompts after finish.
- Prefer this naming over older `outputs/smoke_*` guidance when paths conflict;
  put smokes under `results/smoke_*` or `outputs/rl/` with a clear `NN_` / `smoke_` prefix agreed in-task.

## Done Criteria

- New paths match the tables above
- No new `phase*` artifact directories
- After any long job: cleanup commands were shown in the same turn
