# Phase 2E2 — Coldstart v1 construction

> **Status: closed.** `coldstart_v1.jsonl` built; SFT-v1 trained + val-200 answer baselines in `docs/PHASE2E4_SFTV1_BASELINES.md`.

## Design

| Mix | Target n | Signal |
|-----|--------:|--------|
| internal calibration | 950 | Base Direct✓ |
| search-required | 850 | Direct✗ ∧ Base-Oracle✓ + BM25 obs |
| BM25 hard-neg evidence | 1150 | Candidate Top-5 context → gold `<evidence>` |
| evidence + reasoning | 1200 | ~400 Kimi hard rationale + template fill |
| search_format protocol | 400 | gold-covered BM25 |
| **total** | **≈4550** | |

### Teacher I/O contract (v2)

```text
Question + Gold Evidence + Gold Answer
        ↓
Kimi (thinking disabled by default)
        ↓
{"reasoning": "..."}          ← Teacher owns content
        ↓
semantic validator + quality_score (0–5)
        ↓
code wrap: <think>...</think> ← protocol is deterministic
```

Do **not** ask Kimi to emit `<think>` XML. Do **not** train on `reasoning_content`.

Default call (mode **A**):

- `thinking: {type: disabled}`
- `response_format: json_schema` (`{"reasoning": string}`)
- `max_tokens: 512`, `temperature: 0.3`

## Deliverables

| Path | Role |
|------|------|
| `data/sft/coldstart_v1.jsonl` | 4550-row mixture |
| `data/sft/llamafactory/eca_coldstart_v1_{train,dev,smoke}.jsonl` | ShareGPT for LlamaFactory |
| `configs/sft/qwen25_3b_lora_coldstart_v1.yaml` | LoRA train (same hparams as v0) |
| `configs/sft/qwen25_3b_lora_coldstart_v1_merge.yaml` | merge export |
| `results/phase2e2_coldstart_v1_20260808/` | build audit (local) |
| `results/teacher_reasoning_n550_*_teacher550_modeA/` | Teacher cache (local) |

## Teacher scale-up

| Run | parse | accept | notes |
|-----|------:|-------:|-------|
| smoke_abc (5×3) | 1.0 | 0.8 | chose mode A |
| smoke20_A | 1.0 | 0.95 | 19/20 |
| teacher550 | 1.0 | **0.933** (513/550) | used top 400 in mix |

## Build audit (4550)

- kimi2.6 reasoning: **400**
- template_v0 reasoning fill: 800
- val-200 overlap: **0**
- rejected rows: **0**

## Non-goals

- Do not continue-train from SFT-v0 adapter.
- Do not change LoRA rank/lr/epochs for v1.
- Do not let Teacher rewrite evidence or answer.
- Do not distill Kimi `reasoning_content` into SFT targets.
