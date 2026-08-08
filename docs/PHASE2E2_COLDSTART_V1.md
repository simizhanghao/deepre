# Phase 2E2 — Coldstart v1 construction

> **Status: Teacher I/O contract rewritten (JSON rationale + code `<think>` wrap).**  
> Train from Base with identical LoRA hparams after `coldstart_v1.jsonl` passes audit (Phase 2E3).

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

Fallback **B**: `json_object` if schema unsupported.  
Ablation **C**: thinking enabled, `temperature=1.0`, `max_tokens=16384` (content JSON only enters SFT).

Acceptance (semantic only):

- parseable JSON
- answer consistent
- grounded (no novel proper nouns)
- both gold evidences lexically used (multi-hop)
- length ~20–150 words
- `quality_score >= 4`
- no meta/protocol phrases

Over-generate + filter: ~550 requests → keep best ~400.

## Code

| Path | Role |
|------|------|
| `src/sft/teacher_reasoning.py` | mining + JSON prompt + semantic validation |
| `scripts/generate_teacher_reasoning.py` | API caller (modes A/B/C) → `reasoning_cache.jsonl` |
| `src/sft/coldstart_v1_builder.py` | mixture assembly (`think` = bare rationale) |
| `scripts/build_sft_coldstart_v1.py` | write `data/sft/coldstart_v1.jsonl` + audit |

Endpoint defaults (override with env):

```bash
export KIMI_BASE_URL=http://10.16.137.2:8000/v1
export KIMI_API_KEY=EMPTY          # do not commit real keys
export KIMI_MODEL=Kimi-K2.6-CT-FP8KV
```

## Commands

```bash
# 1) A/B/C smoke (5 each) — pick stable mode
python scripts/generate_teacher_reasoning.py \
  --mode abc --max-samples 5 --run-tag smoke_abc --concurrency 4

# 2) Mode A smoke 20
python scripts/generate_teacher_reasoning.py \
  --mode A --max-samples 20 --run-tag smoke20_A --concurrency 8

# 3) Over-generate ~550 → later filter to ~400
python scripts/generate_teacher_reasoning.py \
  --mode A --n-persistent 440 --n-other 110 \
  --run-tag teacher550 --concurrency 16

# 4) Build v1
python scripts/build_sft_coldstart_v1.py \
  --teacher-cache results/teacher_reasoning_n*_*/reasoning_cache.jsonl \
  --n-teacher-reasoning 400 \
  --output-jsonl data/sft/coldstart_v1.jsonl \
  --audit-dir results/phase2e2_coldstart_v1
```

## Smoke pass bar (not 20/20)

- parse success ≥ 95%
- semantic accept ≥ 70%
- answer consistency ≥ 95%
- grounding ≥ 90%
- mean accepted rationale ≈ 30–100 words
- human check 10–20: bridge across 2 evidences? no new facts? better than `template_v0`?

## Smoke log

- Candidate mine OK: persistent available **1586**, other hard **512**.
- 2026-08-07: smoke20 failed (timeout / 500 / connection refused).
- 2026-08-08 `smoke20_c16`: API OK but **0/20** — XML `<think>` gate + thinking-mode budget mismatch.
- 2026-08-08: **I/O rewrite** — Teacher JSON rationale; code wraps `<think>`; modes A/B/C.
- 2026-08-08 `smoke_abc` (5 each):

  | Mode | parse | accept | ans | ground | avg words | finish |
  |------|------:|-------:|----:|-------:|----------:|--------|
  | **A** (default) | 1.00 | 0.80 | 0.80 | 1.00 | 44.8 | stop×5 |
  | B | 1.00 | 0.80 | 0.80 | 1.00 | 46.0 | stop×5 |
  | C | 1.00 | 0.80 | 0.80 | 1.00 | 50.0 | stop×5 |

  - Artifacts: `results/teacher_reasoning_n5_*_smoke_abc_mode{A,B,C}/`
  - Compare: `results/teacher_mode_compare_*_smoke_abc.json`
  - All `finish_reason=stop`; no `reasoning_content` on this endpoint (thinking flag likely ignored).
  - Single reject: pathological long gold string (bio) vs short comparison answer — filter is working as designed.
  - **Select mode A** → next smoke20, then over-generate ~550.

## Non-goals

- Do not continue-train from SFT-v0 adapter.
- Do not change LoRA rank/lr/epochs for v1.
- Do not let Teacher rewrite evidence or answer.
- Do not distill Kimi `reasoning_content` into SFT targets.
