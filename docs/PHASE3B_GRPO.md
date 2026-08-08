# Phase 3B — Answer-only GRPO (Search-R1 baseline learnability)

> **Status: 3B2 step 50 done — conditional continue → 100.** Hard audit in [`PHASE3B2.md`](PHASE3B2.md).  
> Frozen policy init: `outputs/sft_qwen25_3b_coldstart_v1_merged`.  
> Ckpt: `outputs/rl/grpo_sftv1_smoke/global_step_50`.

## Decisions (locked 2026-08-08)

| Item | Choice |
|------|--------|
| Framework | **Official veRL** (multi-turn Agent Loop / Search-R1-like recipe) |
| Not used for train | Old vendored trainer in `external/Search-R1` (keep as algorithm reference only) |
| Rollout engine | **SGLang** + FSDP |
| Docker | **New** `eca-verl` from `verlai/verl` — do **not** reuse `lf-sft:ready` |
| GPUs | **4× A100** (`device=4,5,6,7`), `n_gpus_per_node=4`, `tensor_parallel_size=1` |
| Init / Ref | **Merged** SFT-v1 (both actor init and frozen reference) — no RL-LoRA in 3B |
| Group size | `rollout.n = 4` |
| Retriever | **Candidate-BM25** via custom `BaseTool` + HTTP (must pass `sample_id`) |
| Full-Corpus | Parallel prep only; does not block 3B |
| Reward | `R = EM + 0.1 × Format`；F1 **metric only** |
| Data | Hotpot **train** subset (512/1024 later); **val-200 frozen eval-only** |
| KL | `use_kl_in_reward=false`；actor 侧极小 KL（≈0.001），ref=冻结 SFT-v1 |
| Sampling (RL) | **Not greedy** — e.g. temperature 0.8–1.0, top_p≈0.95（避免 4 条同轨迹） |

## Stage split

```text
3B0  Docker + scaffolding + gates                 ✓
3B1  2～5 step micro-smoke                         ✓ (2026-08-08)
3B2  5→50 answer-only GRPO baseline + TB           ✓ (audit 2026-08-08; next 50→100)
3C   + Evidence Reward
3D   + Cost / duplicate
3E   Routing / cost tradeoff
4    Formal EM/F1/Pareto / ablation
```

## 3B success criteria (not val EM)

```text
finish_rate stays high under sampling
reward_mean moves OR group std > 0 on a non-trivial fraction
zero_std_group_rate not stuck ≈ 1.0 forever
KL grows slowly / controllably
search_count does not explode
observation tokens NEVER appear in response_mask==1 decode
```

Even `val Candidate 0.485 → 0.49` is irrelevant for 3B0/3B1.

---

## Critical notes (read before coding)

### 1. Do not mix SFT docker with veRL
`lf-sft:ready` has LlamaFactory torch/transformers stack. veRL needs Ray + SGLang/vLLM + FSDP. Mixing = days of ABI pain.

### 2. Pin image after first green smoke
`verlai/verl:sgl055.latest` (or newer `sgl059.latest`) is a floating tag. After 3B0 works, record:

```text
image digest
verl commit (inside container)
torch / CUDA / SGLang / transformers versions
```

All formal runs freeze that digest.

### 3. Observation mask is a hard stop
veRL Agent Loop: `response_mask=1` LLM tokens, `=0` tool/observation tokens.  
**First trajectory dump must show:** decode(`response_mask==1`) contains **no** observation text.  
If it does → **stop RL immediately** (model would learn to emit retrieval text).

### 4. Candidate-BM25 tool must carry `sample_id`
Global Search-R1 HTTP `{"queries":[...]}` is wrong for us. Tool call must be:

```json
{"sample_id": "...", "query": "...", "topk": 5}
```

Each trajectory’s tool instance is bound to its sample’s candidate docs.

### 5. Old `SearchTool` class may be gone
Do not `from verl.tools.search_tool import SearchTool` from outdated tutorials.  
Implement a custom tool subclassing current veRL `BaseTool`.

### 6. Upstream batch defaults are huge
Search-R1-era scripts use `train_batch_size=512` etc. Smoke must downscale, e.g.:

```text
~8 prompts / step × n=4 → 32 trajectories
2～5 optimizer steps first
```

### 7. zero_std is expected early — diagnose before declaring failure
All-0 or all-1 groups → σ=0 → no GRPO signal. Check diversity / temperature / reward spread.  
If answers all wrong but evidence would differ → that motivates Phase 3C Evidence Reward.

### 8. KL only once
No KL-in-reward **and** heavy KL-in-loss double penalty.

### 9. Tags
Prefer aligning tool observation injection with veRL’s expected tool-response path; keep ECA TraceRecord export for analysis. Internal dialect `<observation>` vs Search-R1 `<information>` must be handled in the tool/agent adapter — one mapping, documented.

---

## Dashboard (every few steps)

Priority five: `reward_mean`, `zero_std_group_rate`, `finish_rate`, `search_count`, `KL`.  
Also: reward std, format_valid, duplicate_query, internal/search, entropy, train EM/F1 (log only), observation_tokens, latency.

---

## 3B0 checklist

1. Pull/pin veRL SGLang Docker; create `eca-verl` on GPU 4–7  
2. Mount deepresearch + HF cache; record versions  
3. Actor/ref = SFT-v1 merged  
4. Candidate-BM25 HTTP/`BaseTool` with `sample_id`  
5. Train subset → veRL agent multi-turn data format  
6. `n=4`, EM + 0.1 format  
7. Verify `response_mask` strips observations  
8. 4-GPU, 2–5 optimizer steps  
9. Dump metrics + 5 full trajectories  

See commands in `docs/PHASE3B_SETUP.md`.
