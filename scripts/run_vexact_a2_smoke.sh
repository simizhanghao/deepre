#!/usr/bin/env bash
# A2: minimal EcaSearchAgentLoop -> VeXact integration smoke (lr=0, no formal training).
set -euo pipefail

repo=/data1/hcc/deepresearch
vexact_repo=/data1/hcc/eca-verl-vexact
config=
seed=
output_dir=
max_samples=
debug=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --max-samples) max_samples=$2; shift 2 ;;
    --debug) debug=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$config" && -n "$seed" && -n "$output_dir" && -n "$max_samples" ]] || {
  echo "required: --config --seed --output-dir --max-samples --debug" >&2
  exit 2
}
[[ "$debug" -eq 1 && "$max_samples" -eq 2 ]] || {
  echo "A2 smoke is locked to --debug --max-samples 2" >&2
  exit 2
}

config_abs=$repo/${config#./}
output_abs=$repo/${output_dir#./}
train_file=$output_abs/train_parity_smoke2.parquet
dump_file=$output_abs/a2_first_generate.jsonl
parity_file=$output_abs/a2_trajectories.jsonl
run_log=$output_abs/run.log

test -f "$config_abs"
test -f "$train_file"
test -d "$repo/outputs/rl/03_hf_evidence_step400"
test -x "$vexact_repo/.venv/bin/python"
curl -sf http://127.0.0.1:8001/health >/dev/null || {
  echo "Candidate-BM25 server is not healthy on :8001" >&2
  exit 3
}
mkdir -p "$output_abs"
rm -f "$dump_file" "$parity_file"

export PYTHONPATH=$repo${PYTHONPATH:+:$PYTHONPATH}
export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export ECA_ROLLOUT_BACKEND=vexact
export ECA_ROUTING_MISMATCH_AUDIT=1
export ECA_AUDIT_FIRST_GENERATE_ONLY=1
export ECA_AUDIT_PATH=A2
export ECA_AUDIT_STOP_MODE=none
export ECA_ROUTING_MISMATCH_DUMP=$dump_file
export ECA_PARITY_DUMP=$parity_file
export ECA_BOUNDARY_TABLE=$repo/outputs/rl/04_table_search_boundary/boundary_latest.json
export ECA_BOUNDARY_STRICT=1
export ECA_EVIDENCE_WEIGHT=0.5
export ECA_SEARCH_COST_WEIGHT=0.30
export INFER_FA_IMPL=triton-invariant
export VEOMNI_ATTN_IMPLEMENTATION=triton-invariant
export MODELING_BACKEND=veomni
export VEOMNI_USE_LIGER_KERNEL=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$repo"
set +e
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=4,5,6,7 \
  "$vexact_repo/.venv/bin/python" -m verl.trainer.main_ppo \
  model_engine=veomni \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$train_file" \
  data.val_files="$train_file" \
  data.train_batch_size="$max_samples" \
  data.max_prompt_length=1024 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.seed="$seed" \
  actor_rollout_ref.model.path="$repo/outputs/rl/03_hf_evidence_step400" \
  actor_rollout_ref.model.external_lib=vexact.integrations.verl.fsdp_enable_invariant \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=triton-invariant \
  actor_rollout_ref.actor.optim.lr=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.veomni.fsdp_size=4 \
  actor_rollout_ref.actor.veomni.attn_implementation=triton-invariant \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vexact \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.seed="$seed" \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=0.9 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks=64 \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=triton-invariant \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=384 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$repo/configs/rl/candidate_bm25_tool.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=eca_search_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$repo/configs/rl/eca_agent_loop.yaml" \
  reward.custom_reward_function.path="$repo/src/rl/rewards_boundary.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=False \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.logger='["console"]' \
  trainer.project_name=eca_rollout_alignment \
  trainer.experiment_name=vexact_a2_agent_loop_smoke \
  trainer.default_local_dir="$output_abs/ckpt_scratch" \
  2>&1 | tee "$run_log"
run_rc=${PIPESTATUS[0]}
set -e

echo "a2_run_rc=$run_rc"
[[ "$run_rc" -eq 0 ]]
test -s "$dump_file"

"$vexact_repo/.venv/bin/python" - "$dump_file" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
assert len(rows) == 4, f"expected 4 first-generate rows, got {len(rows)}"
assert all(row["backend"] == "vexact_eca_search_agent_loop" for row in rows)
assert all(row["first_generate_len"] > 0 for row in rows)
print(json.dumps({
    "gate": "A2_AGENT_LOOP_SMOKE_PASS",
    "n_rows": len(rows),
    "backends": sorted({row["backend"] for row in rows}),
    "routes": {route: sum(row["route_first"] == route for row in rows)
               for route in sorted({row["route_first"] for row in rows})},
}, indent=2))
PY
