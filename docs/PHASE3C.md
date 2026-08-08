# Phase 3C — Evidence-Aware GRPO

> Status: **formal short/long probe from SFT-v1** (not from 3B step100).  
> Formula: \(R_{3C}=R_{answer}+0.5\,R_{evidence\text{-}F1}+0.1\,R_{format}\)  
> Default run: **STEPS=500**, SAVE_FREQ=25, n=4, 4×A100.

## Why aggressive

3B closed the infra story and exposed the pathology:

```text
answer≈0.20  format≈0.99  search=0  zero_std≈0.77
```

3C asks whether dense Evidence F1 breaks the no-search / zero-std shortcut.

## Frozen (same as 3B except reward + OUT_DIR)

```text
Init / Ref   = SFT-v1 merged (fresh global_step=0)
group n      = 4
retrieval    = Candidate-BM25 + sample_id
max_search   = 2
λ_e          = 0.5   (env ECA_EVIDENCE_WEIGHT)
Cost/Dup     = OFF (weights reserved for 3D)
```

## Launch

```bash
# once: rebuild parquet with supporting_facts in ground_truth
python scripts/build_grpo_smoke_dataset.py

# offline gate (CPU)
python scripts/offline_reward_replay_3c.py

# 4-GPU train
STEPS=500 SAVE_FREQ=25 bash scripts/tmux_grpo_evidence.sh
tmux attach -t eca-grpo-3c
# TB: http://127.0.0.1:6007
```

Stop only on: NaN, finish collapse, KL explode, evidence reward stuck all-zero.

## Success (vs 3B@100)

| Metric | 3B | 3C target |
|--------|---:|----------:|
| zero_std | 0.77 | **&lt;0.60** (great if ~0.40) |
| Evidence F1 | ~0 | **↑** |
| Answer | ~0.20 | ≥0.22 or not down |
| Search | 0 | may ↑ (OK; Cost=3D) |
| Finish | 0.99 | &gt;0.95 |

## Ablation tree

```text
SFT-v1
 ├── 3B Answer+Format
 └── 3C Answer+Evidence+Format   ← this run
 └── 3D +Cost+Duplicate          ← later, also from SFT-v1
```
