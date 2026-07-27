---
name: rl-reward-grpo
description: >-
  Implements GRPO/PPO rollout, modular reward functions, KL control, retrieved
  token masking, and search cost penalties for Search-R1-style RL. Use when
  writing reward code, rollout logic, or debugging RL training.
---

# RL Reward and GRPO Skill

## When to Use

- Implementing `reward/*.py`
- Connecting veRL / DeepResearch-R1 GRPO training
- Debugging reward hacking or KL collapse

## Reward Components (Modular, Logged Separately)

```python
total = (
    1.0 * answer_reward          # EM or token F1
  + 0.1 * format_reward          # valid XML tags
  + 0.1 * valid_search_reward    # at least one useful search when needed
  - 0.03 * search_count          # cost penalty
  - 0.05 * repeated_query_count
  - 0.2 * invalid_action_count
  + 0.3 * evidence_support_score
  - 0.3 * early_final_penalty    # final without search/evidence
)
```

## Safety Rules (Anti Reward-Hacking)

1. **Never** let `format_reward` dominate `answer_reward`
2. **Never** optimize search_count alone — always pair cost penalty with answer reward
3. Log every component per trajectory; inspect top/bottom 5 before training
4. Offline reward script must run on saved trajectories without GPU

## GRPO Defaults

| Param | Value |
|-------|-------|
| algorithm | GRPO (PPO as ablation only) |
| rollout per prompt | 4–8 |
| max turns | 3–5 |
| max response length | 2k–4k |
| KL coef | start small, monitor curve |

## Retrieved Token Masking (Search-R1 Core)

- Rollout inserts observation tokens into context
- Policy gradient optimizes **model-generated tokens only**
- Observation / retrieved tokens excluded from loss and advantage

## veRL / Multi-Turn Known Issues

| Issue | Mitigation |
|-------|------------|
| SGLang rollout hangs (long-tail) | Start with vLLM; limit max_turns; smoke 2-turn first |
| Tokenization delta mismatch | Set `tokenization_sanity_check_mode=ignore_strippable` if needed |
| OOM after SGLang init | Smoke on 1 GPU before multi-GPU |
| Chat template reasoning mismatch | Use same template for SFT and RL |
| MCP transport errors | Use local BM25 tool, not MCP, for v1 |

## Workflow

1. Implement reward functions + unit tests
2. Run offline reward on 5 handcrafted trajectories; verify ranking matches intuition
3. Smoke GRPO: ≤8 prompts, ≤2 turns, 1 GPU
4. Only then scale to full training with user approval

## Done Criteria

- Reward unit tests pass
- 5 handcrafted examples ranked correctly with written explanation
- Component-wise reward logged to `outputs/{run_name}/reward_debug.jsonl`
- No RL job started without successful smoke
