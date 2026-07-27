# Search-R1 源码通读笔记（searchR1_read）

> 面向 **Evidence-Cost-Aware Deep Research Agent** 项目的 Search-R1 源码精读。
> 目标：在动手改造前，彻底读通 Search-R1 从「一条数据」到「一次梯度更新」的完整闭环，
> 标出关键机制、易错坑点，以及本项目的改造落点。
>
> 阅读顺序（也是数据流方向）：
> **数据格式 → 多轮 rollout（generation / tensor_helper）→ 检索服务 → 单轮 demo（infer）→ reward（A）→ GRPO 优势与训练循环（D）**
>
> 所有行号引用基于 `external/Search-R1/` 下的源码。

---

## 0. 全局心智图

```
train_grpo.sh                       # 启动脚本，配置所有超参
  └─ verl/trainer/main_ppo.py       # 入口：构建 RewardManager + RayPPOTrainer
       └─ ray_trainer.py::fit()     # 训练主循环（每步一次梯度更新）
            ├─ generation.py        # 多轮 think→search→information→answer rollout
            │    └─ retrieval_server.py (HTTP)  # 外部检索服务
            ├─ RewardManager (main_ppo.py)      # 打分：抽 <answer> 做 EM
            │    └─ qa_em.py                     # EM 归一化 + 判定
            ├─ compute_advantage (ray_trainer)  # GRPO 组内标准化优势
            │    └─ core_algos.py                # GRPO/PPO 数学核心
            └─ update_actor                      # clipped policy loss 更新
                 └─ _create_loss_mask            # info_mask → loss_mask
```

**一句话**：Search-R1 = veRL(GRPO) + 多轮搜索 rollout + retrieved token masking + 纯 EM outcome reward。

---

## 1. 数据格式（`scripts/data_process/nq_search.py`）

训练数据是 parquet，每条样本核心字段：

| 字段 | 含义 | 备注 |
|------|------|------|
| `prompt` | 对话模板，含 `<think>/<search>/<information>/<answer>` 使用说明，并带一个示例 `<answer> Beijing </answer>` | ⭐ 这个示例后面 reward 抽取会依赖它 |
| `data_source` | 数据集名（nq / hotpotqa / ...） | reward 按它选打分函数 |
| `ability` | `"fact-reasoning"` | 任务能力标签，仅作分类/记录，训练不直接用 |
| `reward_model.ground_truth` | `{"target": [答案列表]}` | EM 比对目标，可有多个可接受答案 |
| `extra_info` | `{"split": split, "index": idx}` | ⭐ `index` 后面当 GRPO **分组 id（uid）**；`split` 标记 train/test |

**关键联系**：`extra_info.index` 不是随便的序号——它在 `fit()` 里被复制成 `uid`，作为 GRPO「同一题的多次采样归到一组」的分组依据（见第 6 节）。

---

## 2. 多轮 rollout（`search_r1/llm_agent/generation.py`）

核心类 `LLMGenerationManager`，`run_llm_loop()` 实现「生成→搜索→观测→再生成」的循环。

**流程**：
1. 模型生成文本，遇到 `</search>` 停下（stop token）。
2. 用正则从**当前轮输出**里抽出 `<search>query</search>` 的 query（训练侧用 `re.search` 取第一个）。
3. 调用检索服务（HTTP），拿到 topk 文档。
4. 把结果包成 `<information>...</information>` 追加回上下文。
5. 继续下一轮，直到出 `<answer>` 或达到 `max_turns`。

**关键机制 — retrieved token masking（双份 mask 的由来）**：
- `attention_mask`：标记哪些 token 是有效的（非 padding），用于注意力计算。
- `info_mask`：额外标记「哪些 token 是模型自己生成的」vs「哪些是检索回来的 `<information>` 内容」。
- 检索内容**可以被模型读到**（在 attention 里可见），但**不应被当作模型要学习的输出**（不进 loss、不进 KL）。
- 所以需要两份 mask：attention 用全部有效 token，loss/KL 用去掉检索内容的 `info_mask`。

**`active_mask`**：批量生成时，标记哪些序列还在「活跃」（还没结束）。已经产出 `<answer>` 或超轮数的序列停止继续生成，避免浪费算力。

**坑点**：
- `requests.post` 无 timeout/retry → 检索服务卡住会 hang 住整个 rollout。
- `execute_predictions` 里用 `assert` 校验检索结果条数 → 不匹配直接崩，无优雅降级。
- `_passages2string` 只保留 title/text，**丢掉 doc_id** → 后续做引用（citation）溯源困难（本项目改造要补回 doc_id）。
- `max_obs_length` 默认较小（500）→ topk 较大时观测会被截断，丢证据。

---

## 3. 张量工具（`search_r1/llm_agent/tensor_helper.py`）

为变长序列的批量生成服务，提供：padding、attention_mask 生成、position_ids 生成、拼接。

**关键点**：
- **左填充 vs 右填充**：prompt 左填充（内容靠右），response 右填充（内容靠左）。这样拼接后有效内容连续，且新生成 token 总在右端追加。
- **position_ids**：随 padding 调整，保证位置编码正确（padding 不占真实位置）。
- `info_mask` 的对齐/拼接也在这里维护，和 `generation.py` 配合完成 retrieved token masking 的张量层落地。

---

## 4. 检索服务（`search_r1/search/retrieval_server.py`）

FastAPI 应用，暴露 `POST /retrieve` 端点（默认 `http://127.0.0.1:8000/retrieve`）。

- 支持 **BM25**（稀疏，CPU）或 **Dense（e5/BGE + FAISS）**（稠密，GPU）。
- 训练侧和推理侧通过 HTTP **解耦**：检索是独立进程/服务，rollout 只发请求。
- 返回 topk 文档，格式化为 `{title, text}` 列表。

**坑点**：
- 无条件 `import faiss` → 即使只用 BM25 也依赖 faiss。
- Dense 检索 `model.cuda()` 硬编码 → 一定占 GPU，需用 `CUDA_VISIBLE_DEVICES` 和训练分卡，否则抢显存。
- **对 2×A100 的启示**：若用 Dense 检索会吃掉一部分 GPU；若用 BM25（CPU）则两张卡都能留给训练。这是能否跑起 3B GRPO 的关键取舍。

---

## 5. 单轮推理 demo（`infer.py`）

`generation.py` 的极简版：用 HuggingFace `model.generate` + 自定义 `StoppingCriteria` 演示「生成→搜索→再生成」。

- `StopOnSequence`：遇到 `</search>` 停下。因 tokenizer 对 `</search>` 有多种编码（带/不带空格、换行），需注册多个变体。
- `get_query`：用 `re.findall(...)[-1]` 取**整个累积 prompt 里最后一个** query（训练侧则用 `re.search` 取当前轮第一个，二者语义不同）。
- 调同一个 `/retrieve` 端点，结果同样包成 `<information>` 追加。

**坑点**：硬编码 Qwen2.5 EOS、硬编码 topk、**无 max_turns 限制**（只靠 EOS，可能死循环）、每轮重编码整个 prompt（低效）。demo 只为理解逻辑，不用于训练。

---

## 6. Reward 计算（A）：`main_ppo.py` + `qa_em.py`

### 6.1 打分函数选择（main_ppo.py:25-29）

```25:29:external/Search-R1/verl/trainer/main_ppo.py
def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError
```

7 个 QA 数据集全部用 `compute_score_em`。**坑**：新数据源不注册会 `raise NotImplementedError`。

### 6.2 RewardManager（main_ppo.py:32-97）

- `reward_tensor = torch.zeros_like(responses)`：形状与 responses 相同（每 token 一个位置），初始全 0。
- 逐条用 attention_mask 切出有效 prompt/response，解码成 `sequences_str`。
- 调打分函数得 `score`（0 或 1）。
- ⭐⭐ `reward_tensor[i, valid_response_length - 1] = score`：**只在最后一个有效 token 写分**。

```80:80:external/Search-R1/verl/trainer/main_ppo.py
            reward_tensor[i, valid_response_length - 1] = score
```

这就是 **outcome reward + 稀疏奖励**：整条轨迹只在末尾给一个信号，其余全 0。

### 6.3 EM 打分（qa_em.py）

- `normalize_answer`（19-33）：SQuAD 标准归一化 = 小写 + 去标点 + 去冠词(a/an/the) + 规范空格。EM 不是死字符串相等，是归一化后相等。
- `em_check`（36-46）：归一化后与任一 gold 完全相等即 1。
- `subem_check`（49-59）：改为**子串包含**（更宽松）。默认训练用严格 EM，不是 subem。
- `extract_solution`（62-82）：抽 `<answer>...</answer>`。

```77:82:external/Search-R1/verl/utils/reward_score/qa_em.py
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()
```

⭐⭐ **头号坑**：`<answer>` 必须 **≥2 个**才取最后一个，否则返回 None（判 0）。原因：**prompt 模板里自带一个示例 `<answer> Beijing </answer>`**，算第 1 个；模型真正生成的是第 2 个。**改 prompt 模板（去掉示例/换 chat template）会导致全盘 0 分、训练崩溃。**

- `compute_score_em`（85-110）：三档 —— 抽不到 answer → 0；抽到且 EM 对 → 1；抽到但错 → `format_score`（默认 0）。`format_score` 是预留的「格式奖励」接口，默认关闭。

### 6.4 Reward 全貌 & 本项目改造点

Search-R1 reward = **纯 EM，0/1 二值，只在末 token（稀疏 outcome）**。没有 evidence / citation / cost / duplicate / process / format(默认) 奖励。

| 本项目要加 | 改哪里 | 怎么改 |
|-----------|--------|--------|
| evidence reward | `compute_score` | 检查 `<evidence>` 是否覆盖 gold supporting facts |
| citation reward | 同上 | evidence 的 quote 是否真来自 observation（需保留 doc_id） |
| format reward | 开 `format_score` 或独立算 | 标签完整性 |
| cost penalty | RewardManager 内 | 从 `batch.meta_info` 读 `valid_search_stats`，按 search_count 扣分 |
| duplicate penalty | 解析轨迹 query | 重复 query 扣分 |
| process reward（非稀疏） | `reward_tensor` 写入位置 | 中间 token 也写值 |

---

## 7. GRPO 优势与训练循环（D）：`core_algos.py` + `ray_trainer.py`

### 7.1 GRPO 一句话

> **不训练 critic。对同一题采样一组回答，用「组内均值」当基线，比平均好→正优势、差→负优势，再按组内标准差归一化。**
> 省掉与 actor 同量级的 critic → 显存/算力省近一半 → 这是 2×A100 能跑 3B 的关键。

### 7.2 GRPO 优势（core_algos.py:111-155）

```130:132:external/Search-R1/verl/trainer/ppo/core_algos.py
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
```
把稀疏 token reward 沿序列求和 → 每条一个标量分。

```151:153:external/Search-R1/verl/trainer/ppo/core_algos.py
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
```
**优势 = (自己分 − 组内均值) / (组内标准差 + ε)**，再广播到该条所有 response token（⭐ 同一条回答每个 token 优势相同），用 eos_mask 清零 padding。返回 `scores, scores`（advantages == returns，因为没有 critic 就没有独立 value target）。

⭐⭐ **zero-std 陷阱**：组内**全对**或**全错** → std=0 → 优势全 0 → 这组无梯度、白采样。
- 缓解：数据难度适中；**dense reward（evidence+cost）把 0/1 变连续值 → 组内几乎不会全同 → std 不为 0**。这是本项目 dense reward 的额外卖点。

### 7.3 优势分发器（ray_trainer.py:123-154）

```140:149:external/Search-R1/verl/trainer/ppo/ray_trainer.py
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        ...
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
```
`index = uid`，而 `fit()` 里 `uid = index.copy()`（第 744 行）——即数据里的 `extra_info.index`。**同一题多次采样共享 index → 分到一组**；`_balance_batch` 打乱顺序也不影响，因为分组靠 id。

### 7.4 KL 惩罚（ray_trainer.py:91-120）

```96:96:external/Search-R1/verl/trainer/ppo/ray_trainer.py
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
```
⭐ KL 优先用 **info_mask** → 检索 token 不参与 KL。
`token_level_rewards = token_level_scores − beta * kld`。
两种施加方式：`use_kl_loss=False` → KL 进 reward；`=True` → KL 作独立 loss 项（GRPO 常用后者）。

### 7.5 Policy Loss（core_algos.py:163-194）

```189:193:external/Search-R1/verl/trainer/ppo/core_algos.py
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
```
⭐ **GRPO 与 PPO 共用同一 clipped loss**，GRPO 只是把优势换成组内标准化优势，loss 形式不变。`ratio = exp(log_prob − old_log_prob)`，clip 防一步更新过大。

### 7.6 retrieved token masking 的最后一块（ray_trainer.py:854-867）

```856:860:external/Search-R1/verl/trainer/ppo/ray_trainer.py
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask
```
`loss_mask = info_mask`，只在 `do_search and state_masking` 时启用，传给 `update_actor`。**检索 `<information>` 不进 policy loss**。

**完整 masking 链路**：
```
generation.py 产生 info_mask
  → tensor_helper.py padding 对齐
    → ray_trainer: KL 用 info_mask(96) + loss 用 info_mask(859)
      → 检索 token 既不算 KL 也不算 policy loss
```

### 7.7 fit() 主循环（ray_trainer.py:654-838）

```
for batch in dataloader:
  batch.repeat(n_agent)                 # 每题复制多份，为组内采样准备
  run_llm_loop(...)                     # ★ 多轮搜索 rollout（generation.py）
  compute_log_prob                      # old_log_probs（PPO ratio 用）
  uid = index.copy()                    # ★ 分组 id
  _balance_batch                        # 负载均衡（打乱顺序，靠 uid 分组不怕）
  ref_log_prob                          # 参考模型概率（KL 用）
  [use_critic] values                   # ★ GRPO 下跳过
  reward_fn(batch)                      # ★ RewardManager 打 EM 分（A）
  apply_kl_penalty                      # KL 进 reward 或留作 loss
  compute_advantage                     # ★ GRPO 组内标准化（D）
  [use_critic] update_critic            # ★ GRPO 下跳过
  update_actor + _create_loss_mask      # ★ clipped loss 更新，info_mask 屏蔽检索
  定期 validate / save_checkpoint
```
⭐ GRPO 省显存一眼可见：`compute values` 与 `update_critic` **在 GRPO 下全部跳过**。

`use_critic` 判定（567-574）：`adv_estimator=='grpo'` → `use_critic=False`。

---

## 8. 面试 / 易错点速查

1. **GRPO vs PPO**：GRPO 用组内均值当基线去掉 critic，省显存；loss 形式与 PPO 相同（clipped ratio）。
2. **优势公式**：(分 − 组均值)/组标准差，同一条回答所有 token 优势相同。
3. **zero-std 陷阱**：组内全对/全错 → 无梯度；dense reward 可缓解。
4. **分组靠 `uid=index`**：同题多采样共享 index；balance 打乱不影响。
5. **info_mask 双重作用**：KL(96) 与 policy loss(859) 都排除检索 token。
6. **GRPO 中 advantages == returns**（返回 `scores, scores`）。
7. **EM 需归一化**（SQuAD 标准）；**`<answer>` 必须 ≥2 个**（依赖模板示例），改模板易致全 0。
8. **稀疏 outcome reward**：只在末 token 给 0/1，是最简 RLVR。
9. **检索内容不参与打分**：sequences_str 含 `<information>` 但只抽 `<answer>` → 证据未被利用（本项目改造点）。
10. **检索方式决定分卡**：BM25(CPU) 省 GPU，Dense(GPU) 抢显存 —— 影响 2×A100 能否跑 3B。

---

## 9. 一句话总结

> Search-R1 = **veRL(GRPO) + 多轮搜索 rollout + retrieved token masking + 纯 EM outcome reward**。
> `fit()` 每步：多轮搜索生成 → 算 old/ref log_prob → RewardManager 打 EM 分 → GRPO 把同题一组分数标准化成优势（省掉 critic）→ PPO clipped loss 更新 actor，全程用 info_mask 把检索 token 排除在 KL 与 loss 之外。
> 命门是 **zero-std**（组内全同则无梯度），而本项目的 **dense evidence/cost reward 恰好能缓解它** —— 这是把 reward(A) 与 GRPO(D) 连起来的最大收获。
