# Phase 3B2 — Search-R1-style GRPO baseline (answer-only)

> Status: **CLOSED @ step 100** — do **not** extend answer-only baseline.  
> Ckpt: `outputs/rl/grpo_sftv1_smoke/global_step_100`  
> Audits: [step50](../results/phase3b2_grpo_sftv1_smoke_step50_20260808/) · [step100](../results/phase3b2_grpo_sftv1_baseline_step100_20260809/)

## Verdict (one paragraph)

veRL+SGLang multi-turn GRPO **pipeline is stable** (abort=0, finish≈1, format≈1, no NaN).  
Native `critic/score` rises weakly across windows (0.23→0.27→0.29) so the policy moves,  
but with diagnostics on (steps 61–100): **`search_count = 0` every step**,  
`zero_std_group_rate ≈ 0.77` (42% of steps ≥0.8), response length collapses (~326→75),  
KL climbs (~0.004→0.07→0.10). Answer-only GRPO learned a **no-search shortcut** with  
sparse group signal — enough to **close 3B and motivate 3C Evidence**, not to grind 200/1000.

## Plan locked → executed

```text
1. Fix logging only (Ray TaskRunner metrics hook) — NO knob changes     ✓
2. Resume → 100 formal baseline                                        ✓
3. Hard audit @100 (windows + diagnostics)                             ✓
4. CLOSE 3B (no 200/500/1000 baseline grind)                           ← now
5. Phase 3C from SFT-v1 fresh init (NOT from step100):
     R_3B = Answer + 0.1 Format
     R_3C = Answer + λ_e Evidence + 0.1 Format
```

## Frozen knobs (do not change mid-run)

```text
Init       = SFT-v1 merged
GPU        = 4 × A100 (container devices 0–3 = host 4–7)
group n    = 4
retrieval  = Candidate-BM25 + sample_id
max search = 2
reward     = EM + 0.1 × format
Evidence / Cost / Duplicate = OFF
lr         = 1e-6
kl_loss_coef = 0.001
train      = 128 (smoke parquet), batch=8 → 16 steps/epoch
```

## Launch (host) — resume 50→100 after metrics hook

```bash
tmux kill-session -t eca-grpo 2>/dev/null || true
STEPS=100 SAVE_FREQ=5 bash scripts/tmux_grpo_smoke.sh
tmux attach -t eca-grpo   # Ctrl-b d to detach
# expect log lines: [phase3b] TaskRunner metrics: patched
# then each step:   [phase3b] step=N answer=... zero_std=... search=...
```

Must-have TB/console keys (51–100): `reward/answer_reward/{mean,std}`,
`reward/format_reward/mean`, `reward/total_reward/{mean,std}`,
`grpo/zero_std_group_rate`, `agent/finish_rate`, `agent/search_count/mean`,
`agent/duplicate_query_count/mean`, `agent/max_search_hit_rate`,
`agent/search_rate`, `agent/internal_rate`.

**Metrics hook (2026-08-09):** file-patch `TaskRunnerV1.run` → `apply()` inside Ray
actor (`scripts/patch_verl_phase3b_metrics.py`). Use `list(extra_fields)` not
`.tolist()` (TransferQueue LinkedList). Verified live e.g.
`[phase3b] step=61 answer=... | zero_std=... | search=...`.
Watch whether `search_count→0` explains `response_length` drop (shortcut vs concise).

### Ops notes (false “crashes” at step 16)

Two separate issues looked the same (GPU empty, Ray SIGTERM, no traceback):

1. **Host `nohup docker exec ... &`**: Cursor/agent shell end → Docker SIGTERM into
   container. Fix: launcher uses **`docker exec -d`**.
2. **`trainer.total_epochs=1`**: with 128 samples / batch 8 → 16 steps/epoch. veRL exits
   when `current_epoch >= total_epochs` even if `total_training_steps=50`. After step 16
   the driver returns cleanly → Ray shutdown. Fix: `run_grpo_smoke.sh` sets
   `TOTAL_EPOCHS=${TOTAL_EPOCHS:-$STEPS}` so epochs do not bind early.

---

## Hard audit @ step 100 (2026-08-09) — CLOSE 3B

### Completion

| Item | Value |
|------|-------|
| Latest ckpt | `outputs/rl/grpo_sftv1_smoke/global_step_100` |
| Saves | every 5 through 100 |
| Log | `logs/grpo_grpo_sftv1_smoke_to100_20260809_000013.log` (resume@60) |
| Phase3B lines | steps **61–100** (40 steps; earlier legs pre-hook) |
| NaN / abort | **none** / **0** |

### Window trends (native `critic/score`, full run)

| Window | score mean±std | kl_loss | entropy | resp_len |
|--------|----------------|---------|---------|----------|
| 1–20 | **0.234 ± 0.085** | 0.0035 | 0.470 | 326 |
| 21–50 | **0.275 ± 0.108** | 0.0087 | 0.553 | 207 |
| 51–100 | **0.291 ± 0.132** | 0.068 | 0.568 | 122 |
| late 80–100 | 0.308 ± 0.128 | **0.099** | 0.617 | **75** |
| **step 100** | 0.413 | 0.123 | 0.574 | 74 |

Score is a **weak upward** window trend with huge step noise (e.g. 50=0.475, 60=0.100, 90=0.131, 100=0.413).  
Do **not** treat step100 alone as “learned.”

### Diagnostics (phase3b, steps 61–100 only)

| Metric | 61–80 | 81–100 | 61–100 |
|--------|------:|-------:|-------:|
| answer_reward | 0.205 | 0.205 | **0.205** |
| format_reward | 0.983 | 0.997 | **0.990** |
| total_reward | 0.303 | 0.304 | 0.304 |
| zero_std_group_rate | 0.756 | 0.788 | **0.772** |
| finish_rate | 0.983 | 0.997 | **0.990** |
| search_count | **0.000** | **0.000** | **0.000** (0/40 steps nonzero) |
| max_search_hit | 0 | 0 | 0 |
| kl_loss | 0.057 | 0.099 | 0.078 |

### Gate checklist @100

| Gate | Verdict | Evidence |
|------|---------|----------|
| Finish stable | **PASS** | finish≈0.99; abort=0 |
| Format OK | **PASS** | format≈0.99 |
| EM / reward directional | **WEAK / noisy** | native score 0.23→0.29; phase answer flat ~0.20 on 61–100 |
| zero_std not stuck | **FAIL (chronic high)** | mean 0.77; 42% steps ≥0.8 |
| KL slow | **BORDERLINE** | late kl_loss→0.10–0.12 (still no NaN/explosion) |
| Search healthy | **FAIL (collapse)** | search_count=0 all diagnosed steps |
| No NaN | **PASS** | none |

### Four 3B questions — final

1. **Can agentic GRPO learn?** Pipeline yes; **search behavior no** — policy moved toward format-stable no-search answers.  
2. **How sparse?** **Severe**: zero_std≈0.77 under n=4 answer-only.  
3. **Over-search?** Opposite: **under-search / zero-search** shortcut.  
4. **Stable?** Train loop yes; **behaviorally degraded** (len↓, search→0, KL↑).

### Decision

**CLOSE Phase 3B.** Do not run 200/500/1000 answer-only.

Next: **Phase 3C Evidence Reward**, same frozen infra knobs where possible, init = **SFT-v1 merged** (not `global_step_100`).

```text
SFT-v1
 ├── 3B: R = Answer + 0.1 Format          ← done (baseline pathology documented)
 └── 3C: R = Answer + λ_e Evidence + 0.1 Format   ← next (50–100 step recipe probe)
```

Optional before 3C coding: short frozen-eval ckpt50 vs ckpt100 (Candidate EM / search_count) to quantify shortcut on held-out; not required to start 3C scaffolding.

---

## Hard audit @ step 50 (2026-08-08)

### Completion

| Item | Value |
|------|-------|
| Latest ckpt | `outputs/rl/grpo_sftv1_smoke/global_step_50` |
| Saves | 5,10,15,20,25,30,35,40,45,50 |
| Log (final leg) | `logs/grpo_sftv1_smoke_to50_resume15_20260808_231101.log` |
| TB | `outputs/rl/tensorboard/grpo_sftv1_smoke` (INACTIVE after finish) |
| NaN | **none** |
| `response/aborted_ratio` (late) | **0.0** |

### Proxy metrics (console / TB `actor/*`, `critic/score/*`)

Split Phase3B tags (`reward/answer_reward`, `grpo/zero_std_group_rate`, `agent/finish_rate`,
`agent/search_count/*`) **did not appear** in this run — Ray `TaskRunner` does not see the
in-process monkeypatch. Audit below uses veRL native scalars; treat EM/format/search as
**proxies**.

`R = EM + 0.1×format` ⇒ `critic/score` ∈ {0, 0.1, 1.0, 1.1}. When `score/min=0.1` on a
step, every sample in the batch had valid format; rough batch EM ≈ `score/mean − 0.1`.

| Window | score mean±std | kl_loss | entropy | resp_len |
|--------|----------------|---------|---------|----------|
| early 6–15 | **0.245 ± 0.057** | 0.0036 | 0.464 | 340 |
| mid 20–35 | **0.255 ± 0.110** | 0.0065 | 0.536 | 237 |
| late 40–50 | **0.286 ± 0.122** | 0.0119 | 0.587 | 187 |
| **step 50** | **0.475** (min=0.1, max=1.1) | 0.0143 | 0.624 | 151 |

Milestone snapshot:

| step | score | kl_loss | entropy | grad_norm | resp_len | clip |
|------|------:|--------:|--------:|----------:|---------:|-----:|
| 10 | 0.319 | 0.0065 | 0.631 | 0.90 | 171 | 0.031 |
| 20 | 0.119 | 0.0030 | 0.489 | 1.82 | 404 | 0.094 |
| 30 | 0.403 | 0.0042 | 0.362 | 1.17 | 366 | 0.094 |
| 40 | 0.159 | 0.0107 | 0.534 | 0.97 | 195 | 0.031 |
| 45 | 0.350 | 0.0100 | 0.615 | 1.02 | 135 | 0.031 |
| 50 | 0.475 | 0.0143 | 0.624 | 1.83 | 151 | 0.031 |

TB (actor filter) matches: entropy ↑ ~0.4→0.62, `kl_loss` ↑ ~0.002→0.014, `lr` flat 1e-6,
`actor/loss` noisy then drops near end.

### Gate checklist

| Gate | Verdict | Evidence |
|------|---------|----------|
| Finish stable | **PASS (proxy)** | `aborted_ratio=0`; `num_turns/mean=2.0` every logged step |
| Format OK | **PASS (improving)** | late steps often `score/min=0.1` (all-format batches); early often `min=0` |
| EM / reward directional | **WEAK PASS** | window mean 0.245 → 0.286; high batch noise; step50 strong (≈EM 0.375 if format=1) |
| zero_std not stuck | **UNKNOWN / soft OK** | no `zero_std_group_rate`; advantage spread stays ~2.3–2.5 (not collapsed) |
| KL slow | **PASS** | kl_loss 0.0036 → 0.012; rollout_corr/kl ~1e-4; no spike |
| Search not glued to max=2 | **UNKNOWN** | no search_count; `num_turns=2` always (likely 1 search + answer, not max-hit proof) |
| No NaN | **PASS** | none in parsed steps |

### Answers to the four 3B questions

1. **Can agentic GRPO learn?** Mild yes — reward window up, format more consistent late, entropy/KL move as expected under small KL coef. Not a clean EM climb; smoke-128 + n=4 is noisy.
2. **How sparse?** Unmeasured directly. Non-trivial advantage spread argues against total zero-std collapse; still need wired metrics or offline group-std before trusting.
3. **Over-search?** Unmeasured. Fixed `num_turns=2` suggests a stable short tool pattern, not runaway multi-search; confirm with `search_count` / max-hit on next leg.
4. **Stable?** Yes for train loop: no abort/NaN, KL controlled, resp_len/clip down (less ramble/truncation).

### Decision

**Conditional continue → 100** (same frozen knobs, same OUT_DIR).

Do **not** change lr / n / reward mid-run. Before or during 50→100:

- Prefer **file-patch** veRL `_compute_metrics` so `[phase3b]` / TB split tags stick on Ray workers.
- Watch: score window, kl_loss slope, clip_ratio, and (once wired) zero_std + search_count.
- Stop early only if zero_std≈1 chronic, finish/format collapse, KL spike, or NaN.

Ablation for 3C: both B and C restart from **SFT-v1**, not from this 3B ckpt.

---

## Metrics map (intended)

| Signal | TB key |
|--------|--------|
| Answer EM | `reward/answer_reward/mean` |
| Format | `reward/format_reward/mean` |
| Total R | `reward/total_reward/mean` / `critic/score/mean` |
| Zero-std groups | `grpo/zero_std_group_rate` |
| Finish | `agent/finish_rate` |
| Search | `agent/search_count/mean`, `agent/max_search_hit_rate` |
| KL / grad | `actor/kl_loss`, `rollout_corr/kl`, `actor/grad_norm` |

```bash
tensorboard --logdir /data1/hcc/deepresearch/outputs/rl/tensorboard/grpo_sftv1_smoke --port 6006 --bind_all --load_fast=false
```
