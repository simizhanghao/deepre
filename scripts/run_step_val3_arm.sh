#!/usr/bin/env bash
set -euo pipefail

repo=/data1/hcc/deepresearch
vexact=/data1/hcc/eca-verl-vexact
arm=${1:-}
gpus=${STEP_VAL3_GPUS:-0,1,2,3}
data_dir=$repo/data/cur/step_val3_fresh128
rollouts=1
case "$arm" in
  root)
    data=$data_dir/root_baselines.parquet; rows=256; batch=64; micro=4
    agent=eca_search_agent; agent_config=$repo/configs/rl/eca_agent_loop.yaml ;;
  step_allsearch)
    data=$data_dir/step_allsearch.parquet; rows=128; batch=64; micro=4
    agent=eca_step_adaptive_agent; agent_config=$repo/configs/rl/eca_step_adaptive_agent_loop.yaml ;;
  step_gate)
    data=$data_dir/step_gate.parquet; rows=128; batch=64; micro=4
    agent=eca_step_adaptive_agent; agent_config=$repo/configs/rl/eca_step_adaptive_agent_loop.yaml ;;
  step_gate_smoke)
    data=$data_dir/step_gate_smoke2.parquet; rows=2; batch=2; micro=1
    rollouts=2
    agent=eca_step_adaptive_agent; agent_config=$repo/configs/rl/eca_step_adaptive_agent_loop.yaml ;;
  *) echo "usage: $0 root|step_allsearch|step_gate|step_gate_smoke" >&2; exit 2 ;;
esac
steps=$((rows / batch))
root=$repo/results/25_step_adaptive/val3/$arm
mkdir -p "$root"
rm -f -- "$root/outcomes.jsonl" "$root/step_records.jsonl" "$root/run.log" "$root/run_status.json"

curl -fsS http://127.0.0.1:8006/health >/dev/null || { echo STEP_VAL3_RETRIEVER_UNHEALTHY; exit 3; }
if [[ "$arm" == step_gate* ]]; then
  curl -fsS http://127.0.0.1:8007/health >/dev/null || { echo STEP_GATE_SERVER_UNHEALTHY; exit 4; }
fi

export PYTHONPATH=$repo${PYTHONPATH:+:$PYTHONPATH}
export ECA_REPO_ROOT=$repo VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export ECA_ROLLOUT_BACKEND=vexact ECA_FORWARD_ONLY_CAPTURE=1
export ECA_STEP_GATE_URL=http://127.0.0.1:8007 ECA_STEP_GATE_TIMEOUT_S=180
export INFER_FA_IMPL=triton-invariant VEOMNI_ATTN_IMPLEMENTATION=triton-invariant MODELING_BACKEND=veomni
export VEOMNI_USE_LIGER_KERNEL=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset ECA_ATTRIBUTION_CAPTURE_DIR ECA_BOUNDARY_TABLE ECA_ROOT_PIVOT ECA_PARITY_DUMP
if [[ "$agent" == eca_search_agent ]]; then
  export ECA_CUR_CAPTURE_JSONL=$root/outcomes.jsonl
  unset ECA_STEP_CAPTURE_JSONL
else
  export ECA_STEP_CAPTURE_JSONL=$root/step_records.jsonl
  unset ECA_CUR_CAPTURE_JSONL
fi

started=$(date +%s)
set +e
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpus" \
  "$vexact/.venv/bin/python" "$repo/scripts/launch_grpo_main.py" \
  model_engine=veomni algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files="$data" data.val_files="$data" data.train_batch_size="$batch" \
  data.max_prompt_length=1024 data.max_response_length=2048 data.filter_overlong_prompts=True \
  data.truncation=error data.return_raw_chat=True data.seed=2026081203 \
  actor_rollout_ref.model.path="$repo/outputs/rl/03_hf_evidence_step400" \
  actor_rollout_ref.model.external_lib=vexact.integrations.verl.fsdp_enable_invariant \
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch actor_rollout_ref.model.enable_gradient_checkpointing=False \
  +actor_rollout_ref.model.override_config.attn_implementation=triton-invariant \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size="$batch" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$micro" actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.veomni.fsdp_size=4 actor_rollout_ref.actor.veomni.attn_implementation=triton-invariant \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$micro" \
  actor_rollout_ref.rollout.name=vexact actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.seed=2026081203 actor_rollout_ref.rollout.n="$rollouts" \
  actor_rollout_ref.rollout.temperature=0 actor_rollout_ref.rollout.top_p=1 \
  actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.calculate_log_probs=True actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$micro" \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks=1024 \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=triton-invariant \
  actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.multi_turn.max_assistant_turns=9 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=8 actor_rollout_ref.rollout.multi_turn.max_tool_response_length=384 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$repo/configs/rl/step_val3_candidate_bm25_tool.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop="$agent" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$agent_config" \
  reward.custom_reward_function.path="$repo/src/rl/rewards_cur.py" reward.custom_reward_function.name=compute_score \
  trainer.nnodes=1 trainer.n_gpus_per_node=4 trainer.total_epochs=100 trainer.total_training_steps="$steps" \
  trainer.val_before_train=False trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
  'trainer.logger=[console]' trainer.project_name=eca_step_val3 trainer.experiment_name="step_val3_${arm}" \
  trainer.default_local_dir="$root/no_checkpoint" 2>&1 | tee "$root/run.log"
run_rc=${PIPESTATUS[0]}
set -e
elapsed=$(($(date +%s) - started))
env -u LD_LIBRARY_PATH "$vexact/.venv/bin/python" - "$root/run_status.json" "$arm" "$rows" "$steps" "$elapsed" "$run_rc" <<'PY'
import json,sys
path,arm,rows,steps,elapsed,rc=sys.argv[1:]
open(path,'w').write(json.dumps({'arm':arm,'rows':int(rows),'steps':int(steps),'wall_seconds':int(elapsed),'run_rc':int(rc),'original_test_read':False},indent=2)+'\n')
PY
echo "step_val3_${arm}_rc=$run_rc elapsed=$elapsed"
[[ "$run_rc" -eq 0 ]]
grep -q "FORWARD_ONLY: actor backward/optimizer/scheduler are disabled" "$root/run.log"
expected=$((rows * rollouts))
if [[ "$agent" == eca_search_agent ]]; then
  test "$(wc -l < "$root/outcomes.jsonl")" -eq "$expected"
else
  test "$(wc -l < "$root/step_records.jsonl")" -eq "$expected"
fi
echo "STEP_VAL3_${arm^^}_PASS"
