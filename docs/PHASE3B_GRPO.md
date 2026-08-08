# Phase 3B — Search-R1 answer-only GRPO baseline (plan)

> **Status: planning.** Prove our SFT-v1 + rollout + mask + GRPO can learn.  
> Do **not** add Evidence/Cost rewards yet.

## Objective

```text
R = R_answer + λ_f R_format
```

Close to Search-R1 backbone. Success = non-trivial learning signal + stable training, not HotpotQA SOTA.

## Proposed smoke shape

| Knob | Smoke target |
|------|-------------|
| Policy init | `outputs/sft_qwen25_3b_coldstart_v1_merged` |
| Prompts | 8–32 train questions (Hotpot distractor pool subset) |
| Group size `n` | 4 (or 5 if following Search-R1 default) |
| max_search_turns | 2 |
| Retriever | Candidate-BM25 first (controlled) |
| GPUs | 1–2× (3B, no critic) |
| Batch | **far below** upstream 512 — must downscale |

## Must monitor

```text
reward mean / std
group std
zero_std_group_rate   ← critical
finish_rate
search_count / duplicate_search
KL / policy loss / entropy
```

If most groups are all-zero EM → `std≈0` → no learning. That is the main engineering risk.

## Asset map

| Asset | Path | Note |
|-------|------|------|
| Upstream GRPO | `external/Search-R1/train_grpo.sh` → `verl.trainer.main_ppo` | Vendored verl inside Search-R1 |
| Generation+mask | `external/Search-R1/search_r1/llm_agent/generation.py` | `info_mask` / `state_masking` |
| Our 3A loop | `src/agents/react_loop.py` | Trace-level mask only; not yet wired to veRL |
| Rewards (empty) | `src/rewards/` | Need `answer` + weak `format` |
| Train configs | none yet | Need `configs/grpo/*_smoke.yaml` |

## Implementation options (to decide)

**A. Adapt Search-R1/veRL in-place**  
- Pros: proven GRPO + masking path  
- Cons: expects HTTP `/retrieve`; default batch huge; tag dialect `<information>` vs our `<observation>`

**B. Keep our react_loop, wrap a minimal GRPO trainer**  
- Pros: ECA TraceRecord / Candidate-BM25 native  
- Cons: more code; easy to get token-level obs mask wrong

Recommendation for first smoke: **A with Candidate retrieve adapter** (HTTP shim over `retrieve_candidate_bm25`), tiny batch — fastest path to a learning curve.

## Open questions (need user input)

1. **Train backend:** Prefer wrapping `external/Search-R1` veRL, or a thinner custom GRPO on our `react_loop`?
2. **GPUs for first smoke:** How many free cards (suggest 4–5 or 6–7), and ok to use 1 GPU micro-smoke first?
3. **Retriever during RL:** Stay on Candidate-BM25 until 3B learns, or stand up Search-R1 full-corpus server in parallel now?
4. **Reward detail:** Answer = EM only, or EM+token-F1 blend? Format penalty weight `λ_f` — start at 0.1?
5. **Data split:** Smoke prompts from train pool (not val-200)? Confirm val-200 stays frozen eval-only.
6. **Docker vs conda:** Run GRPO inside `lf-sft:ready` / new env, or host `deepresearch` + Search-R1 deps?

## Non-goals for 3B

- Evidence reward, cost penalty, routing reward  
- SFT-v2  
- Claiming Agent EM from unfinished protocol routing numbers  

## Exit criteria for 3B smoke

```text
finish_rate stays high under sampling
zero_std_group_rate not stuck near 1.0
reward mean moves or at least group std > 0 on a non-trivial fraction
no observation tokens in policy loss (spot-check)
```
