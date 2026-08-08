# Phase 3B2 — Search-R1-style GRPO baseline (answer-only)

> Status: **running / next** — resume from `global_step_5` → 50, hard audit, then maybe 100.  
> Experiment dir kept as `outputs/rl/grpo_sftv1_smoke` for checkpoint continuity (name = historical).

## Frozen knobs (do not change mid-run)

```text
Init       = SFT-v1 merged
GPU        = 4 × A100 (container devices 0–3 = host 4–7)
group n    = 4
retrieval  = Candidate-BM25 + sample_id
max search = 2
reward     = EM + 0.1 × format
Evidence / Cost / Duplicate = OFF
```

## Launch (host)

```bash
STEPS=50 SAVE_FREQ=5 bash scripts/tmux_grpo_smoke.sh
tmux attach -t eca-grpo   # Ctrl-b d to detach
```

### Ops note (2026-08-08 crash at step16)

Root cause: host used `nohup docker exec ... &` (client-attached). When the Cursor/agent
shell ended, Docker sent **SIGTERM** into the container → Ray
`INTENDED_USER_EXIT` / `ray.shutdown()` with **no Python traceback**.  
Fix: launcher now uses **`docker exec -d`** so the train job is owned by the
container daemon and survives SSH/tmux/Cursor disconnects.

Resume after step 50 if healthy:

```bash
tmux kill-session -t eca-grpo 2>/dev/null || true
STEPS=100 SAVE_FREQ=10 bash scripts/tmux_grpo_smoke.sh
```

## Metrics (TensorBoard + `[phase3b]` console line)

| Signal | TB key |
|--------|--------|
| Answer EM | `reward/answer_reward/mean` |
| Format | `reward/format_reward/mean` |
| Total R | `reward/total_reward/mean` |
| Zero-std groups | `grpo/zero_std_group_rate` |
| Finish | `agent/finish_rate` |
| Search | `agent/search_count/mean`, `agent/max_search_hit_rate` |
| Dup query | `agent/duplicate_query_count/mean` |
| Routing | `agent/search_rate`, `agent/internal_rate` |
| Obs tokens | `agent/observation_tokens/mean` |
| KL / grad | `actor/kl_loss`, `rollout_corr/kl`, `actor/grad_norm` |

```bash
tensorboard --logdir /data1/hcc/deepresearch/outputs/rl/tensorboard/grpo_sftv1_smoke --port 6006 --bind_all
```

## Hard gates at step 50

Continue → 100 if: finish stable, format OK, EM/reward directional, zero_std not stuck ≥0.8–0.9, KL slow, search not glued to max=2, no NaN.

Stop / close 3B if: zero_std chronic, finish collapse, search max-hit + flat EM, KL spike, NaN.

## 3B answers four questions then → 3C

1. Can agentic GRPO learn? (answer_reward trend)  
2. How sparse? (`zero_std_group_rate`)  
3. Does GRPO over-search? (search_count / max-hit)  
4. Stable? (finish / format / KL)

Ablation for 3C: both B and C restart from **SFT-v1**, not from 3B ckpt.
