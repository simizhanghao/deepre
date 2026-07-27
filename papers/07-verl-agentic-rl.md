# 资料 7：veRL Agentic RL 文档

> 非 PDF，框架文档链接汇总。先跑通 example，不必一开始读全部源码。

## 核心文档

| 主题 | URL |
|------|-----|
| Agentic RL 入门 | https://verl.readthedocs.io/en/latest/start/agentic_rl.html |
| Multi-turn Rollout (SGLang) | https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html |
| Search-R1 官方仓库 | https://github.com/PeterGriffinJin/Search-R1 |
| veRL 主仓库 | https://github.com/volcengine/verl |

## 必须理解的概念

### Rollout
- 用当前 policy model 对 prompt 采样生成 trajectory
- Agentic 场景下 rollout = 多轮 generation + tool call + observation 注入

### Multi-turn Tool Call
- 每轮 assistant 生成 → 解析 action → 环境返回 observation → 拼回 context → 下一轮
- veRL 支持 server-based async rollout

### Reward Function
- Rule-based outcome reward（本项目首选）
- 通过 training config 传入 custom reward
- 每个 component 应单独 log

### GRPO / PPO 配置
- `actor_rollout_ref.actor` — learning rate, KL, batch size
- `actor_rollout_ref.rollout` — max turns, response length, engine (vLLM/SGLang)
- `trainer` — n_gpus, project name, experiment name

## 本项目注意事项

| 已知问题 | 应对 |
|----------|------|
| SGLang multi-turn 挂死 | smoke 先用 vLLM；限制 max_turns=2 |
| Retrieved token masking | observation token 不参与 policy loss |
| Tokenization delta mismatch | `tokenization_sanity_check_mode=ignore_strippable` |
| OOM | smoke 1 GPU；LoRA 而非 full finetune |

## 读完后回答

1. rollout 和 training step 的关系是什么？
2. multi-turn 下 observation 从哪注入 context？
3. reward 在哪个阶段计算、怎么回传？

## 代码对应

- `training/rl/grpo_search.py`
- `configs/grpo_search_r1.yaml`
