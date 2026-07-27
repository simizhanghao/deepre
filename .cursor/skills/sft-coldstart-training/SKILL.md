---
name: sft-coldstart-training
description: >-
  Prepares SFT data, LoRA configs, and cold-start training for format learning
  before RL. Use when building SFT pipelines, loss masking, or SFT evaluation
  in this DeepResearch project.
---

# SFT Cold Start Training Skill

## When to Use

- Converting trajectories to SFT JSONL
- Writing LoRA training scripts
- Evaluating format accuracy after SFT

## Goal

Teach the model correct `<search>` / `<final>` format **before** any RL. SFT fixes stability; RL fixes strategy.

## Workflow

1. Load validated SFT JSONL from `data/processed/`
2. Build prompt template (system + user question)
3. **Loss mask:** observation / retrieved text tokens → label = -100
4. Train LoRA on Qwen2.5-7B-Instruct (3B for debug)
5. Log: train loss, eval loss, format accuracy, answer proxy, avg length
6. Save to `outputs/{run_name}/checkpoint/` + 5 sample generations

## Smoke Mode (Default)

```
--max-train-samples 8
--max-eval-samples 8
--max-steps 5
--debug
no distributed training
```

Full mode requires explicit user approval.

## Config Defaults

| Param | Value |
|-------|-------|
| method | LoRA / QLoRA |
| data | 2k–10k filtered trajectories |
| epochs | 1–2 |
| lr | 1e-5 ~ 2e-5 |
| max_length | 8192 |

## Common Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| Observation in loss | Model copies retrieval verbatim | Mask obs tokens |
| Train on L3 only | Format ok but search strategy poor | Curriculum L1/L2/L3 mix |
| Skip SFT → RL | RL reward hacking on format | Always SFT before RL |
| Checkpoint sprawl | Many `checkpoint-*` dirs | One output dir per run_name |
| Full finetune on 7B | OOM | LoRA rank 8–64 first |

## Done Criteria

- `format_accuracy` improves on held-out 8-sample eval
- ≥5 generations saved to `outputs/{run_name}/samples.jsonl`
- Config yaml saved alongside checkpoint
