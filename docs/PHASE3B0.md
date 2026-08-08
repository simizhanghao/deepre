# Phase 3B0 / 3B1 — Wire Search Agent into veRL + GRPO micro-smoke

> Status (2026-08-08): **3B0 scaffolding + 3B1 micro-smoke PASS**  
> (`STEPS=5`, ckpt `outputs/rl/grpo_sftv1_smoke/global_step_5`, reward mean≈0.19–0.23).

## Scope (only these four)

1. Candidate-BM25 HTTP + veRL `BaseTool` (`sample_id` bound at `create`)
2. `train_grpo_smoke_128` parquet (policy never sees gold/contexts)
3. Minimal GRPO config / launcher (`n=4`, EM+0.1 format, 4 GPU, 2–5 steps later)
4. `response_mask` audit hard gate

Evidence / Cost / Duplicate / Full-Corpus / SFT retune: **OFF**.

## Layout

| Path | Role |
|------|------|
| `src/rl/retrieval_server.py` | HTTP `/retrieve` `{sample_id,query,topk}` |
| `src/rl/candidate_bm25_tool.py` | veRL `BaseTool` name=`search` |
| `src/rl/eca_search_agent_loop.py` | XML `<search>/<observation>` AgentLoop |
| `src/rl/rewards_3b.py` | `R = EM + 0.1 * Format` |
| `configs/rl/candidate_bm25_tool.yaml` | tool registry |
| `configs/rl/eca_agent_loop.yaml` | agent loop registry |
| `configs/rl/grpo_sftv1_smoke.yaml` | intended knobs (reference) |
| `scripts/build_grpo_smoke_dataset.py` | build parquet + contexts index |
| `scripts/start_candidate_retrieval_server.py` | start HTTP server |
| `scripts/audit_response_mask.py` | mask leak gate |
| `scripts/run_grpo_smoke.sh` | **3B1** launcher (after gates) |

## Why custom AgentLoop (not hermes ToolAgentLoop)

SFT-v1 speaks:

```text
<search>...</search>
<observation>...</observation>
<evidence>/<think>/<answer>
```

veRL's default `tool_agent` + hermes injects OpenAI `<tool_call>` schemas — distribution shift vs cold-start.  
`eca_search_agent` keeps the SFT dialect and still calls the same `CandidateBM25Tool` (sample_id via `create_kwargs`).

## Commands (3B0 only)

### 1) Build smoke data

```bash
cd /data1/hcc/deepresearch
conda activate deepresearch
python scripts/build_grpo_smoke_dataset.py --n-train 128 --n-val 16
```

### 2) Start retrieval server (host, network=host visible to eca-verl)

```bash
python scripts/start_candidate_retrieval_server.py \
  --index data/rl/grpo_smoke_128/contexts_index.jsonl \
  --port 8001
# other terminal:
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8001/retrieve \
  -H 'content-type: application/json' \
  -d '{"sample_id":"<id from train_ids.txt>","query":"test","topk":3}'
```

### 3) Synthetic mask audit

```bash
python scripts/audit_response_mask.py
# must print PASS
```

### 4) 3B1 micro-smoke (inside eca-verl) — already passed once

```bash
docker exec -it eca-verl bash
export PYTHONPATH=/workspace/deepresearch:/workspace/verl
bash /workspace/deepresearch/scripts/run_grpo_smoke.sh   # default STEPS=5
```

### 5) Longer run in tmux + auto-resume (host)

```bash
# Host: pane0=retriever :8001, pane1=docker GRPO
STEPS=50 SAVE_FREQ=5 bash scripts/tmux_grpo_smoke.sh
tmux attach -t eca-grpo   # detach: Ctrl-b d

# Continue after kill / crash (same OUT_DIR, raise STEPS > last ckpt):
# last ckpt is in outputs/rl/grpo_sftv1_smoke/latest_checkpointed_iteration.txt
STEPS=100 SAVE_FREQ=10 bash scripts/tmux_grpo_smoke.sh
```

`trainer.resume_mode=auto` (default): loads `OUT_DIR/global_step_*` via `latest_checkpointed_iteration.txt`.

### 6) TensorBoard curves

Launcher writes under:

```text
outputs/rl/tensorboard/grpo_sftv1_smoke/
```

Host:

```bash
# conda env with tensorboard, or: pip install tensorboard
tensorboard --logdir /data1/hcc/deepresearch/outputs/rl/tensorboard/grpo_sftv1_smoke \
  --port 6006 --bind_all
# browser → http://<host>:6006
# watch: actor/loss, critic/score/mean, critic/rewards/mean, response_length/mean
```

Note: the first 5-step smoke used `logger=["console"]` only — no TB events for that run.
Re-runs with the updated launcher will create TB logs.

## Hard stop

If `decode(response_mask==1)` contains observation body / `<observation>` → **STOP**, do not train.

## Known env mismatch (sgl055.latest)

Docker image ships **SGLang 0.5.5**; latest veRL git expects newer pause/continue request types
(`ContinueGenerationReqInput`). Smoke launcher auto-runs:

```bash
python scripts/patch_verl_sgl055_compat.py
```

Longer-term options: pull `verlai/verl:sgl059.latest`, or pin a verl tag matched to the image.

## Smoke metrics (3B1, 5 steps)

| step | actor/loss | score/mean | score/max | notes |
|------|------------|------------|-----------|-------|
| 1 | ~0.118 | ~0.225 | 1.1 | first GRPO update OK |
| 5 | ~0.389 | ~0.191 | 1.1 | ckpt saved; ~4m43s total |
