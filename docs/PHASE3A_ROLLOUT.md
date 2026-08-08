# Phase 3A — SFT-v1 Search Agent Rollout (no training)

> **Status: closed (smoke passed).** Agent loop healthy on n=8 and n=32. Next: Phase 3B GRPO.

## Goal

```text
Question
  → SFT-v1
  → <think>? / <internal>|<search>
  → [if search] Candidate-BM25 tool
  → <observation>  (env, loss_mask=False)
  → continue → <evidence>/<think>/<answer>
  → optional 2nd search (max_search_turns=2)
  → TraceRecord
```

## Smoke results

### n=8 (`phase3a_n8`)

| Metric | Value |
|--------|------:|
| finish_rate | **1.0** |
| parse_ok / obs mask | **1.0** |
| mean_search_count | 1.0 |
| max_search_turn_hit | 0 |
| search / internal | 100% / 0% |
| mean EM / Evid F1 | 0.75 / 0.725 |
| Dir | `results/agent_rollout_n8_20260808_163047_phase3a_n8/` |

### n=32 (`phase3a_n32`) — gate

| Metric | Value | Gate |
|--------|------:|------|
| finish_rate | **0.9688** | ≥0.8 ✅ |
| parse_ok_rate | **1.0** | ✅ |
| observation_mask_ok_rate | **1.0** | ✅ |
| mean_search_count | 0.9375 | stable ✅ |
| max_search_turn_hit_rate | 0 | ✅ |
| duplicate_query | 0 | ✅ |
| search / internal | 93.75% / 6.25% | over-search (expected) |
| mean EM / Evid F1 | 0.469 / 0.588 | secondary |
| Dir | `results/agent_rollout_n32_20260808_163356_phase3a_n32/` |

1/32 unfinished (`rollout_unfinished_no_answer`) — acceptable noise.

## Code

| Path | Role |
|------|------|
| `src/tools/candidate_bm25.py` | BM25 over sample contexts with **model query** |
| `src/agents/react_loop.py` | generate→parse→tool→observe→continue |
| `scripts/run_agent_rollout_smoke.py` | CLI smoke |

## Metric meanings (health)

- **finish_rate**: reached `<answer>`
- **parse_ok / obs mask**: protocol parseable; observation `loss_mask=False`
- **search_count / duplicate / max_turn_hit**: search discipline
- **search/internal rates**: first-action routing bias (not Agent EM)
- **EM / Evid F1**: secondary quality peek only

## After 3A → 3B

See `docs/PHASE3B_GRPO.md`. Answer-only GRPO first; no evidence/cost rewards yet.
