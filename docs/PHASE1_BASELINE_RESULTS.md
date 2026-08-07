# Phase 1 Baseline Results (Frozen)

> Status: **Phase 1 frozen** after n=200 HotpotQA distractor/validation.  
> Model: Qwen2.5-3B-Instruct. Retriever: BM25s Candidate (in-sample contexts only).  
> Raw run artifacts live under local `results/` (gitignored). This doc is the public summary.

## 1. Setup

| Item | Value |
|------|-------|
| Dataset | `hotpotqa/hotpot_qa`, config=`distractor`, split=`validation` |
| Eval subsets | `data/eval/hotpotqa_{8,50,200}.jsonl` (seed=42 nested prefixes) |
| Generator | `/data1/hcc/.hf_home/Qwen2.5-3B-Instruct` |
| Methods | `direct` / `oracle` / `candidate_bm25` via `scripts/run_baseline.py` |
| Candidate retrieval | `scripts/retrieve_candidate_bm25.py`, top_k=5, scope=`candidate` |
| Metrics | EM / token F1 / format_valid; title Recall@K for retrieval |
| Taxonomy | `scripts/compare_baselines.py` labels A–E |

**Not claimed here:** Full-Corpus Wikipedia BM25 (deferred).

## 2. Main table (n=200, freeze)

| Method | EM | token F1 | Prompt tokens | Obs tokens | Notes |
|--------|---:|---------:|--------------:|-----------:|-------|
| Direct | 0.180 | 0.245 | 53.9 | 0 | no retrieval |
| Candidate-BM25 | 0.435 | 0.553 | 906.1 | 842.2 | Top-5 in-sample |
| Oracle | 0.595 | 0.714 | 363.4 | 299.5 | gold supporting docs |

Deltas (EM):

| Contrast | Δ EM |
|----------|-----:|
| Oracle − Direct | +0.415 |
| BM25 − Direct | +0.255 |
| Oracle − BM25 | +0.160 |

Retrieval (Candidate-BM25, n=200):

| Metric | Value |
|--------|------:|
| mean title Recall@1 | 0.443 |
| mean title Recall@3 | 0.765 |
| mean title Recall@5 | 0.880 |
| title hit_all@5 | 0.760 |

Structure holds:

```text
Direct < Candidate-BM25 < Oracle
```

## 3. Stability vs n=50

| Method | n=50 EM | n=200 EM |
|--------|--------:|---------:|
| Direct | 0.12 | 0.18 |
| Candidate-BM25 | 0.36 | 0.435 |
| Oracle | 0.56 | 0.595 |

| Taxonomy | n=50 | n=200 |
|----------|-----:|------:|
| A retrieval helps | 28% | 28.5% |
| B retrieval miss/noise | 16% | 15.5% |
| C evidence/reasoning gap | 44% | **38%** |
| D search unnecessary | 4% | **10%** |
| E retrieval hurts | 8% | 7.5% |

Absolute EM moves a few points; **ordering and failure structure are stable**.

## 4. Failure taxonomy (n=200)

| Label | Count | Rate | Meaning |
|------:|------:|-----:|---------|
| A | 57 | 28.5% | Direct❌ Oracle✅ BM25✅ — retrieval helps |
| B | 31 | 15.5% | Direct❌ Oracle✅ BM25❌ — retrieval miss / noise |
| C | 76 | **38.0%** | Direct❌ Oracle❌ — evidence / multi-hop gap |
| D | 20 | 10.0% | Direct✅ Oracle✅ BM25✅ — search likely unnecessary |
| E | 15 | 7.5% | Direct✅ BM25❌ — retrieval hurts |
| O | 1 | 0.5% | other |

Derived:

- Direct❌ Oracle✅ (A+B) = **44%** → external knowledge value is real  
- Among those, B/(A+B) = **35%** → retrieval still loses vs oracle  
- C = **38%** → largest single bucket: evidence utilization / reasoning  
- D+E = **17.5%** → fixed always-search is not optimal (routing/cost motivation)

## 5. Local run pointers (not in git)

```text
results/retrieval_candidate_bm25_n200_20260807_154802/
results/baseline_direct_n200_20260807_154900_phase1_final_n200/
results/baseline_oracle_n200_20260807_154912_phase1_final_n200/
results/baseline_candidate_bm25_n200_20260807_154918_phase1_final_n200/
results/compare_baselines_n200_20260807_155225/
```

Reproduce:

```bash
# retrieval cache
python scripts/retrieve_candidate_bm25.py \
  --eval-file data/eval/hotpotqa_200.jsonl --max-samples 200 --top-k 5 --seed 42

# baselines (example)
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/run_baseline.py --method direct \
  --eval-file data/eval/hotpotqa_200.jsonl --max-samples 200 --seed 42

# taxonomy
python scripts/compare_baselines.py \
  --direct results/baseline_direct_n200_*/metrics.json \
  --oracle results/baseline_oracle_n200_*/metrics.json \
  --bm25 results/baseline_candidate_bm25_n200_*/metrics.json \
  --retrieval-summary results/retrieval_candidate_bm25_n200_*/summary.json
```

## 6. Phase 1 conclusions → Phase 2

1. **Search is justified** — Oracle and BM25 both beat Direct by large margins.  
2. **Evidence / reasoning SFT is the top priority** — C remains the largest failure class.  
3. **Retrieval / noise still matters** — BM25 trails Oracle by ~16 EM; Recall@5 is high but not perfect; prompts are ~3× Oracle size.  
4. **Routing / cost is motivated** — D+E ≈ 17.5%; blind search can waste tokens or hurt.  
5. **Stop scaling HotpotQA eval for now** — n=200 is enough to freeze; do not chase 500/1000 before SFT.

**Phase 1 status: complete (frozen).**  
Next: Phase 2A — SFT data contract for `internal` / `evidence` / `reasoning` / format.
