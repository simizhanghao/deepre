# Phase 3C-GEN — Held-out Agent generalization gate

> Status: **IN PROGRESS** (2026-08-09)  
> Purpose: smoke128 train-window `answer≈0.61` is **not** val EM. Measure transfer on frozen val-200.

## Protocol (locked)

| Knob | Value |
|------|--------|
| Eval | `data/eval/hotpotqa_200.jsonl` |
| Loop | `scripts/run_agent_rollout_smoke.py` (Agentic, **not** single-turn baseline) |
| Retriever | in-process Candidate-BM25 (`contexts`) |
| max_search_turns | 2 |
| top_k | 5 |
| temperature | 0.0 |
| Models | SFT-v1 merged · 3B `global_step_100` · 3C `global_step_400` |

## Merge

```bash
bash scripts/export_verl_fsdp_to_hf.sh \
  outputs/rl/grpo_sftv1_smoke/global_step_100 \
  outputs/rl/hf_merged/grpo_sftv1_smoke_step100

bash scripts/export_verl_fsdp_to_hf.sh \
  outputs/rl/grpo_sftv1_evidence_3c/global_step_400 \
  outputs/rl/hf_merged/grpo_sftv1_evidence_3c_step400
```

## Eval

```bash
bash scripts/run_phase3c_gen.sh
# or per-model commands in that script
```

## Gates

See [ROADMAP.md](ROADMAP.md) §3C-GEN. Fill table after runs:

| Model | Answer EM | Evid F1 | search_rate | search_count | P0/P1/P2 | finish |
|-------|----------:|--------:|------------:|-------------:|----------|-------:|
| SFT-v1 | | | | | | |
| 3B@100 | | | | | | |
| 3C@400 | | | | | | |

Verdict: _pending_
