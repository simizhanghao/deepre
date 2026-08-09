# Phase 3C — Evidence-Aware GRPO

> Status: **CLOSED @ step 400** (2026-08-09). Do **not** extend this answer+evidence run to 500.  
> Ckpt: `outputs/rl/grpo_sftv1_evidence_3c/global_step_400`  
> Audit: [step400](../results/phase3c_grpo_sftv1_evidence_step400_20260809/)  
> Board: [RESULTS_BOARD.md](RESULTS_BOARD.md)

## Verdict (one paragraph)

veRL+SGLang multi-turn GRPO stays stable (finish≈1, format≈1, abort=0, KL≈0.015).  
Dense Evidence F1 (**λ_e=0.5**) **breaks the 3B no-search shortcut**: search rises 0→~1,  
Evidence F1 →~0.62, Answer EM ~0.10→~0.61 (window means). Early `zero_std` is healthy (~0.2);  
late `zero_std` climbs (~0.58) with entropy collapsed and search locked at 1 — expected without Cost.  
Enough to **close 3C and hand off to 3D Cost** from SFT-v1; do not grind 500 on this reward.

## Formula & frozen knobs

\[
R_{3C}=R_{answer}+0.5\,R_{evidence\text{-}F1}+0.1\,R_{format}
\]

```text
Init / Ref   = SFT-v1 merged (NOT 3B step100)
group n      = 4
retrieval    = Candidate-BM25 + sample_id
max_search   = 2
λ_e          = 0.5   (ECA_EVIDENCE_WEIGHT)
Cost / Dup   = OFF (reserved for 3D)
lr           = 1e-6
kl_loss_coef = 0.001
train        = smoke128 parquet, batch=8
SAVE_FREQ    = 50
```

## Ops timeline

| When | Event |
|------|--------|
| start | fresh from SFT-v1, target 500, SAVE_FREQ was 25 then set to **50** |
| ~324 | `/data1` full → stop mid-save; incomplete 325 discarded |
| cleanup | dropped non-final 0/1/2 LoRA dirs + mid 3B/3C ckpts; kept merged SFT + 3B@100 + 3C@300 |
| resume | 300→… with `SAVE_FREQ=50` |
| **400** | user stop after `global_step_400` saved; train killed (not continued to 500) |

Kept on disk:

```text
SFT finals:  outputs/sft_qwen25_3b_coldstart_{v0,v1}_merged
3B final:    outputs/rl/grpo_sftv1_smoke/global_step_100
3C final:    outputs/rl/grpo_sftv1_evidence_3c/global_step_400
(+ optional intermediates 300/350 for debug)
```

## Hard audit @400 — window means

From `[phase3c]` console lines (see `results/.../phase3c_by_step.csv`):

| Window | answer | evidence | zero_std | search_rate | finish | kl |
|--------|-------:|---------:|---------:|------------:|-------:|---:|
| 1–50 | 0.098 | 0.272 | 0.193 | 0.408 | 0.970 | 0.003 |
| 51–100 | 0.203 | 0.498 | 0.220 | 0.950 | 0.971 | 0.016 |
| 101–200 | 0.482 | 0.558 | 0.351 | 1.000 | 0.998 | 0.013 |
| 201–300 | 0.572 | 0.590 | 0.479 | 1.000 | 0.999 | 0.016 |
| 301–350 | 0.602 | 0.595 | 0.430 | 0.999 | 0.999 | 0.020 |
| **351–399** | **0.614** | **0.617** | **0.582** | **0.999** | **0.999** | **0.015** |
| last 20 (380–399) | 0.636 | 0.625 | 0.594 | ~1.0 | ~1.0 | 0.014 |

### vs 3B@61–100 (same smoke / n=4 infra)

| Metric | 3B@61–100 | 3C@61–100 | 3C late (351–399) |
|--------|----------:|----------:|------------------:|
| answer | ~0.205 | ~0.22 | **0.614** |
| evidence | ~0 | **~0.51** | **0.617** |
| search | **0** | **~0.98** | **~1.0** |
| zero_std | **0.77** | **~0.24** | 0.58 |
| finish | ~0.99 | ~0.97 | ~1.0 |
| kl | ~0.078 | ~0.016 | ~0.015 |

## Curve reading (short)

- **entropy ↓→~0.01**: policy becomes deterministic (~step 100); not the same as loss.  
- **grad_norm spikes**: GRPO group advantage noise on small batch; OK while KL/finish healthy.  
- **actor/loss≈0 + pg_clipfrac=0**: late updates small; explore/lock issue, not crash.  
- **agent/***: behavior monitors — finish/format≈1; obs tokens mean↑ then lock (~630) = real search use.  
- **Side effect**: search→1 without Cost → Phase 3D.

## Success criteria (retrospective)

| Criterion | Target | Result |
|-----------|--------|--------|
| Break no-search | search > 0 | **PASS** (→1.0) |
| Evidence ↑ | vs ~0 on 3B | **PASS** (~0.62) |
| Answer ↑ / not collapse | ≥ mid 3B | **PASS** (~0.61 late) |
| Pipeline health | finish>0.95, no NaN | **PASS** |
| Early zero_std | <0.60 | **PASS** (~0.22 @51–100) |
| Late zero_std / Cost | watch | **FAIL for long grind** → 3D |

## Launch (reference; closed)

```bash
python scripts/offline_reward_replay_3c.py
STEPS=400 SAVE_FREQ=50 bash scripts/tmux_grpo_evidence.sh   # historical
# TB: http://127.0.0.1:6007  run=grpo_sftv1_evidence_3c
```

## Next

```text
SFT-v1
 ├── 3B Answer+Format          CLOSED @100 (no-search pathology)
 ├── 3C Answer+Evidence+Format CLOSED @400  ← here
 └── 3D +Cost (+Dup)          NEXT, from SFT-v1
```
