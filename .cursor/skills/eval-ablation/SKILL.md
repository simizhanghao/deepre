---
name: eval-ablation
description: >-
  Builds evaluation scripts, benchmark subsets, multi-dimensional metrics, and
  ablation tables comparing baseline/RAG/ReAct/SFT/RL. Use when running evals
  or preparing experiment comparison tables.
---

# Evaluation and Ablation Skill

## When to Use

- Running benchmark eval (HotpotQA, 2Wiki, MuSiQue, Bamboogle)
- Building ablation tables for README / interview
- Comparing SFT vs SFT+RL vs cost-control variants

## Minimum Experiment Groups

1. Baseline — direct answer, no search
2. One-shot RAG — single retrieve + answer
3. ReAct prompt — no training
4. SFT cold-start
5. SFT + RL (Search-R1 style)
6. SFT + RL + search cost control (R1-Searcher++ style)

## Metrics (Not EM/F1 Alone)

**QA:** EM, token F1, answer extract rate

**Agent behavior:** valid action rate, invalid action rate, avg search_count, repeated query rate, early final rate, avg response length

**Evidence:** evidence support rate, citation support rate, unsupported claim rate

**Cost:** avg search count, retrieved tokens, generated tokens, latency proxy

## Output Contract (Single Location)

All eval artifacts under `outputs/{run_name}/eval/`:

```
results.json
results.csv
sample_predictions.jsonl
metrics_summary.md
```

Ablation table → `docs/ablation_table.md` (update in place, do not create `ablation_v2.md`).

## Ablation Minimum Set

| Experiment | Proves |
|------------|--------|
| Base vs RAG | External retrieval needed |
| RAG vs ReAct | Multi-turn search needed |
| ReAct vs SFT | Cold start stabilizes format |
| SFT vs SFT+RL | RL improves strategy |
| RL w/o cost penalty | Cost control prevents search explosion |
| RL w/o evidence reward | Evidence reward reduces hallucination |

## Common Pitfalls

| Issue | Fix |
|-------|-----|
| Eval on train split | Use held-out splits only |
| Different sample sizes per method | Fix `--max-samples` or full split |
| Results in stdout only | Always write JSON/CSV |
| Re-run creates duplicate tables | Overwrite same output path |
| Claiming SOTA | Report relative gains vs baselines only |

## Done Criteria

- One markdown table pasteable into README
- All methods evaluated on same subset with same seed
- Sample predictions saved for badcase analysis (≤20 lines in `docs/badcases/`)
