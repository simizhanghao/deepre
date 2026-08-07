# Phase 2E1 — Train-split capability labeling (Coldstart v1 signals)

> **Status: GPU labeling closed.** Local artifacts under `results/` (gitignored).  
> Purpose: build routing / hard-neg / reasoning-hard pools for `coldstart_v1` — **not** a new eval claim on frozen val-200.

## Inputs (reused)

| Asset | Path |
|-------|------|
| Train pool n8000 | `data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl` |
| Candidate-BM25 Top-5 cache | `results/retrieval_candidate_bm25_n8000_20260807_162150/retrieval_results.jsonl` |
| Eval shards (4×) | `results/phase2e1_shards_20260807_205153/eval/train_shard_{0..3}.jsonl` |
| Base model | `/data1/hcc/.hf_home/Qwen2.5-3B-Instruct` |
| SFT-v0 merged | `outputs/sft_qwen25_3b_coldstart_v0_merged` |

## GPU runs completed

| Run | Model | Method | Dir | n | Correct / EM |
|-----|-------|--------|-----|--:|-------------:|
| Direct labels | Base | direct | `results/phase2e1_direct_label_n8000_20260807_202826_phase2e1/` | 8000 | **1500 / 0.1875** |
| Base Oracle | Base | oracle | `results/phase2e1_base_oracle_n8000_20260807_205154/merged/` | 7993 | **5445 / 0.6812** |
| SFT-v0 Oracle | coldstart_v0_merged | oracle | `results/phase2e1_sftv0_oracle_n8000_20260807_211627/merged/` | 7993 | **5895 / 0.7375** |

Note: Oracle joins use 7993 rows (`split -n l/4` dropped 7 lines from the 8000-line pool). Safe to ignore for pool sizing.

## Derived pools (join on `sample_id`)

Joined n = **7993** (Direct ∩ Base-Oracle ∩ SFT-v0-Oracle).

| Pool | Rule | Count | v1 use |
|------|------|------:|--------|
| Internal | Direct✓ | **1498** | P0 `<internal>` positives |
| Search-required | Direct✗ ∧ Base-Oracle✓ | **4151** | P0 `<search>` positives |
| C-like | Direct✗ ∧ Base-Oracle✗ | **2344** | **not** forced search; P2 reasoning |
| Hard reasoning | SFT-v0 Oracle✗ | **2098** | P2 Teacher candidates (~300–600 sample) |
| └ of which also C-like | SFT-v0✗ ∧ C-like | 1586 | prefer for Teacher |
| Train Oracle Δ (Base→SFT-v0) | — | **+5.63 pp** | consistent with val Oracle +5.5 |

Also: Base✓→SFT✗ on Oracle = 412; Base✗→SFT✓ = 862 (net positive on train Oracle).

## What this enables next (Phase 2E2)

Build `data/sft/coldstart_v1.jsonl` as targeted refinement over v0:

1. **P0 Routing** — sample from Internal + Search-required (calibrated), do not dump all C-like into search.
2. **P1 BM25 hard-neg evidence** — use existing Candidate-BM25 cache as observation context; gold supporting facts as `<evidence>`.
3. **P2 Targeted reasoning** — Teacher (or stronger rationale) only on a hard subset of the 2098, not full pool.

Keep LoRA hparams identical to v0 when entering 2E3 SFT.

## Reproduce (high level)

```bash
# Direct (4-way shard) — already done
python scripts/label_direct_train.py --eval-file data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl \
  --max-samples 8000 --num-shards 4 --shard-id $i --run-dir $RUN_DIR

# Oracle shards via scripts/run_baseline.py --method oracle on train_shard_*.jsonl
# Merge metrics/trace under results/phase2e1_*_oracle_n8000_*/merged/
```

## Non-goals

- No Full-Corpus / dense retriever in this milestone.
- No LoRA retrain in 2E1.
- No frozen val-200 contamination (train pool already audited zero overlap).
