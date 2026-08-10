---
name: acceptance-commands
description: >-
  Formats every shell/terminal command block the user must run as 验收命令（请你执行）.
  Use whenever proposing bash, docker, pytest, train, eval, ray, cleanup, or any
  copy-pasteable command. Agent never runs Shell; user executes and pastes output.
---

# Acceptance Commands（验收命令）

Hard skill. The user runs all shell commands. The agent only proposes them.

## When to Use (Mandatory)

Read and follow this skill **before** any reply that includes a runnable command:
train, eval, docker, ray, pytest, python, git, install, cleanup, `nvidia-smi`, etc.

If this skill was not Read in the current turn, do not emit commands yet — Read it first.

## Hard Rules

1. **Never** use the Shell tool to execute these commands.
2. **One acceptance block per step** (or one tightly related batch). Wait for paste-back.
3. Commands must be **copy-pasteable**: absolute `cd`, real paths, no `<placeholder>` unless labeled.
4. After the user pastes output: interpret, then give the **next** 验收命令 block if needed.
5. Do **not** assume success without paste-back.

## Required Output Format (verbatim structure)

Use exactly this shape (Chinese headers fixed):

```markdown
验收命令（请你执行）— <短标题：本步在验什么>

\`\`\`bash
cd <绝对路径工作目录>
<命令1>
<命令2>
...
\`\`\`

执行后把终端输出贴回。

期望：
- <可检查的成功条件1>
- <可检查的成功条件2>
```

### Field rules

| 字段 | 要求 |
|------|------|
| 标题 | `验收命令（请你执行）—` + 短名（任务/阶段/门禁） |
| 代码块 | 语言标记 `bash`；首行通常 `cd` 到绝对路径 |
| 贴回句 | 固定写：`执行后把终端输出贴回。` |
| 期望 | 条列、可判定（数字、exit code、文件存在、gate 字符串等） |

### Allowed inside the bash block

- `cd` + sequential commands the user can paste once
- env prefixes: `PYTHONPATH=...`, `CUDA_VISIBLE_DEVICES=...`, `STEPS=1 ...`
- `docker exec ...`, `bash scripts/...`, `python scripts/...`
- a final one-liner that prints the key metric/json for easy paste

### Forbidden formats

- Do **not** use the old template alone (`用途:` / `目录:` / `命令:`) as the primary form
- Do **not** bury commands in prose without the 验收命令 header
- Do **not** say “你可以跑一下” without a full block + 期望

## Cleanup commands

Post-train/eval cleanup also uses this format, e.g. title
`验收命令（请你执行）— 清理 Ray/GPU 残留`. Follow
`artifact-naming-and-cleanup` for *what* to kill; this skill for *how* to present it.

## Minimal example

```markdown
验收命令（请你执行）— parity smoke T=0.9 dump 行数

\`\`\`bash
cd /data1/hcc/deepresearch
wc -l results/16_audit_routing_exploration/parity_sglang_32x4/rollouts_t0.9.jsonl
nvidia-smi
\`\`\`

执行后把终端输出贴回。

期望：
- DUMP 行数约为 32（或与 N×batch 一致）
- 相关 GPU 显存已释放或仅剩预期进程
```

## DeepResearch notes

- Host repo default cwd: `/data1/hcc/deepresearch`
- Container paths often `/workspace/deepresearch` inside `eca-verl`
- Long jobs: still one 验收命令 block; 期望里写日志路径与成功标志（如 `DUMP_LINES=`、`gate=`）
