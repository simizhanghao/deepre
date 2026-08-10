# Next Steps — Text ECA (v2)

> [ROADMAP.md](ROADMAP.md) · [RESULTS_BOARD.md](RESULTS_BOARD.md)

## Done (Routing Sampler Alignment Audit — locked findings)

Under `results/16_audit_routing_exploration/worker_mismatch/`:

| Step | Gate / result |
|------|----------------|
| Path C full Eca | 80/80 search; multi-turn length=2048 deferred |
| Path B-current | 80/80 search; first-gen len≈26; stop OK |
| HF Root Score | `p̃_internal≈0.65` → not π≈0 |
| sampler-align | **top_p falsified**; HF@.95 NoSearch internal≈**28%** |
| greedy-tim | HF greedy 20/20 search; tok0 agree PathB **100%** |
| Path B forensic | sampling_params OK (`T=0.9,top_p=0.95,top_k=-1`); **SGLang logp(tok0)≈−0.003 (p≈0.997)** vs HF ≈−0.39 (p≈0.68) |

**Hard verdict:** `SGLANG_ROUTE_TOKEN_LOGIT_TIM`  
Mode/argmax 对齐，但 **route tok0 概率质量在 SGLang 上塌缩到 search**；HF 仍有 ~30%+ internal。  
不是 nucleus、不是缺 T、不是 Branching 场景。

## NOW

1. TIM δ_t formalize on tok0 (HF vs SGLang logprobs) + optional VeXact/debug env  
2. Do **not** Mixed-action / Branching / top_p=1-as-fix / REINFORCE until TIM root-cause fixed or calibrated  
3. Trajectory budget (Path C 2048) still deferred

## Later

Full-Corpus · Phase4 · multimodal

## Naming

`hotpotqa_200` = **dev-200**. Train: `data/rl/train_smoke_128`.
