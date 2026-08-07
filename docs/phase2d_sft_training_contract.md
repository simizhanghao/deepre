# Phase 2D0 — SFT Training Contract (Hard Gates)

> Before any LoRA run. Fixes protocol drift from Phase 2C export.

## 1. Tag unification: `<reasoning>` → `<think>`

| Layer | Canonical tag |
|-------|----------------|
| TraceRecord `step_type` | `think` |
| Search-R1 / rollout surface | `<think>...</think>` |
| Cold-start SFT targets | **`<think>` only** |

Phase 2C `coldstart_v0.jsonl` historically used `<reasoning>`.  
Exporter `scripts/export_coldstart_sharegpt.py` **rewrites** to `<think>` for all train files.  
Do not train on `<reasoning>`.

## 2. Observation must not enter assistant loss

| Content | Learnable? |
|---------|------------|
| `<internal>` / `<search>` / `<evidence>` / `<think>` / `<answer>` | Yes |
| Retrieved docs / `<observation>` body | **No** |

ShareGPT layout for `search_format`:

```text
system
human: Question
gpt: <search>...</search>          # learned
observation: retrieved docs        # NOT learned (LF observation role)
gpt: <evidence>...<think>...<answer>...   # learned
```

Never pack observation into a single assistant target string for vanilla SFT.

## 3. Training backend

| Env | Role |
|-----|------|
| `deepresearch` | data / eval / baselines |
| LlamaFactory (separate env) | LoRA SFT |

## 4. Outputs of 2D0

- `data/sft/llamafactory/eca_coldstart_v0_{train,dev,smoke}.jsonl`
- `configs/sft/qwen25_3b_lora_coldstart_{smoke,v0}.yaml`
- Register datasets in LlamaFactory `data/dataset_info.json`

## 5. Non-goals

No GRPO, no full FT first, no Teacher upgrade, no touching frozen val-200 as train.
