# Phase 3A — SFT-v1 Search Agent Rollout (no training)

> **Status: scaffolding.** Prove the tool loop before any GRPO.

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

Not a routing classifier. Not GRPO yet.

## Constraints (v1 smoke)

| Knob | Value |
|------|------:|
| Policy | `outputs/sft_qwen25_3b_coldstart_v1_merged` |
| Retriever | Candidate-BM25 (per-sample contexts) |
| top_k | 5 |
| max_search_turns | 2 |
| n | 8 → 32 |
| Train | **no** |

## Code

| Path | Role |
|------|------|
| `src/tools/candidate_bm25.py` | BM25 over sample contexts with **model query** |
| `src/agents/react_loop.py` | generate→parse→tool→observe→continue |
| `scripts/run_agent_rollout_smoke.py` | CLI smoke → `results/agent_rollout_*/` |

## Hard gates (health, not EM)

```text
finish_rate              (reach <answer>)
parse_rate / format_valid
search_count / duplicate_query_count
max_search_turn_hit_rate
internal_rate / search_rate
answer EM/F1, evidence F1 (secondary)
observation_tokens, latency
```

Must keep: **observation steps `loss_mask=False`**.

## Commands

```bash
cd /data1/hcc/deepresearch
conda activate deepresearch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# n=8 smoke
CUDA_VISIBLE_DEVICES=4 python scripts/run_agent_rollout_smoke.py \
  --model-path outputs/sft_qwen25_3b_coldstart_v1_merged \
  --eval-file data/eval/hotpotqa_200.jsonl \
  --max-samples 8 --top-k 5 --max-search-turns 2 \
  --run-tag phase3a_n8

# if healthy → n=32
CUDA_VISIBLE_DEVICES=4 python scripts/run_agent_rollout_smoke.py \
  --model-path outputs/sft_qwen25_3b_coldstart_v1_merged \
  --eval-file data/eval/hotpotqa_200.jsonl \
  --max-samples 32 --top-k 5 --max-search-turns 2 \
  --run-tag phase3a_n32
```

## After 3A passes → 3B

Answer-only GRPO (`R_answer + λ_f R_format`), monitor `zero_std_group_rate`.  
Full-Corpus search stays parallel; do not block 3A.
