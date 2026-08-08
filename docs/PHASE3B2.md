# Phase 3B2 — Search-R1-style GRPO baseline (answer-only)

> Status: **formal 3B baseline — metrics fix then step50 → 100 (do not expand to 1000)**  
> Experiment dir: `outputs/rl/grpo_sftv1_smoke` (name historical; ckpt continuity).  
> Step50 audit: [`results/phase3b2_grpo_sftv1_smoke_step50_20260808/`](../results/phase3b2_grpo_sftv1_smoke_step50_20260808/)

## Plan locked (2026-08-08)

```text
1. Fix logging only (Ray TaskRunner metrics hook) — NO knob changes
2. Resume global_step_50 → 100  (= formal 3B baseline, not "smoke")
3. Hard audit @100: windows 1–20 / 21–50 / 51–100 + ckpt50 vs ckpt100 eval
4. If learnable + stable → CLOSE 3B (no 200/500/1000 baseline grind)
5. Phase 3C from SFT-v1 fresh init (NOT from step100 ckpt):
     R_3B = Answer + 0.1 Format
     R_3C = Answer + λ_e Evidence + 0.1 Format
```

Qualitative read of step50: pipeline stable, policy moving, not diverging;
weak positive score trend; **learnability not closed** until zero_std + search
metrics land inside Ray workers. `response_length 340→187` is ambiguous without
`search_count` (concise finish vs shortcut).

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
