# Phase 2E2 — Coldstart v1 construction

> **Status: scaffolding ready; Kimi teacher smoke blocked on API availability.**  
> Train from Base with identical LoRA hparams after `coldstart_v1.jsonl` passes audit (Phase 2E3).

## Design

| Mix | Target n | Signal |
|-----|--------:|--------|
| internal calibration | 950 | Base Direct✓ |
| search-required | 850 | Direct✗ ∧ Base-Oracle✓ + BM25 obs |
| BM25 hard-neg evidence | 1150 | Candidate Top-5 context → gold `<evidence>` |
| evidence + reasoning | 1200 | ~400 Kimi hard `<think>` + template fill |
| search_format protocol | 400 | gold-covered BM25 |
| **total** | **≈4550** | |

Teacher (P2): **Kimi-K2.6-CT-FP8KV** via OpenAI-compatible endpoint.  
Generates **only** `<think>...</think>`; gold evidence/answer locked in builder.

## Code

| Path | Role |
|------|------|
| `src/sft/teacher_reasoning.py` | mining + prompt + validation |
| `scripts/generate_teacher_reasoning.py` | API caller → `reasoning_cache.jsonl` |
| `src/sft/coldstart_v1_builder.py` | mixture assembly |
| `scripts/build_sft_coldstart_v1.py` | write `data/sft/coldstart_v1.jsonl` + audit |

Endpoint defaults (override with env):

```bash
export KIMI_BASE_URL=http://10.16.137.2:8000/v1
export KIMI_API_KEY=EMPTY          # do not commit real keys
export KIMI_MODEL=Kimi-K2.6-CT-FP8KV
```

## Commands

```bash
# 1) Teacher smoke (20) — wait until endpoint is up
python scripts/generate_teacher_reasoning.py \
  --max-samples 20 --run-tag smoke20 \
  --n-persistent 16 --n-other 4 --timeout 300

# 2) After smoke OK → full ~400
python scripts/generate_teacher_reasoning.py \
  --run-tag teacher400 --n-persistent 320 --n-other 80

# 3) Build v1
python scripts/build_sft_coldstart_v1.py \
  --teacher-cache results/teacher_reasoning_n400_*/reasoning_cache.jsonl \
  --n-teacher-reasoning 400 \
  --output-jsonl data/sft/coldstart_v1.jsonl \
  --audit-dir results/phase2e2_coldstart_v1
```

## Smoke log

- Candidate mine OK: persistent available **1586**, other hard **512**.
- 2026-08-07: several smoke20 runs failed with timeout / 500 / connection refused under concurrency (service unstable).
- 2026-08-08 `smoke20_c16`: API reachable; **0/20 accepted**.
  - Dir: `results/teacher_reasoning_n20_20260808_113323_smoke20_c16/`
  - Dominant reject: `need exactly one non-empty <think>...</think>`
  - Kimi returns long meta-planning text without wrapping a single `<think>` block (connectivity OK, format supervision not OK yet).
- Next: fix teacher prompt / post-parse for Kimi output shape, re-smoke, then scale to ~400.

## Non-goals

- Do not continue-train from SFT-v0 adapter.
- Do not change LoRA rank/lr/epochs for v1.
- Do not let Teacher rewrite evidence or answer.
