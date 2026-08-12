#!/usr/bin/env bash
# Phase25 S0 contract audit; Train-only and forward-only.
set -euo pipefail

repo=/data1/hcc/deepresearch
vexact_repo=/data1/hcc/eca-verl-vexact
stage=${STEP_STAGE:-smoke2}
case "$stage" in
  smoke2)
    root=$repo/results/25_step_adaptive/s0/smoke2
    data=$repo/data/step_adaptive/s0_train32/smoke2.parquet
    batch=2
    ngpus=2
    gpus=${STEP_GPUS:-0,1}
    max_num_seqs=8
    max_batched_tokens=8192
    max_cache_blocks=512
    ;;
  train32)
    root=$repo/results/25_step_adaptive/s0/train32
    data=$repo/data/step_adaptive/s0_train32/train32.parquet
    batch=32
    ngpus=4
    gpus=${STEP_GPUS:-0,1,2,3}
    max_num_seqs=16
    max_batched_tokens=16384
    max_cache_blocks=1024
    ;;
  replay8a|replay8b)
    suffix=${stage#replay8}
    root=$repo/results/25_step_adaptive/s0/replay8/run_$suffix
    data=$repo/data/step_adaptive/s0_train32/replay8.parquet
    batch=8
    ngpus=4
    gpus=${STEP_GPUS:-0,1,2,3}
    max_num_seqs=8
    max_batched_tokens=16384
    max_cache_blocks=1024
    ;;
  *) echo "unknown STEP_STAGE=$stage" >&2; exit 2 ;;
esac
mkdir -p "$root"
rm -f -- "$root/step_records.jsonl" "$root/run.log"

curl -fsS http://127.0.0.1:8003/health >/dev/null || {
  echo "CUR-1 Train retriever is not healthy on :8003" >&2
  exit 3
}
export PYTHONPATH=$repo${PYTHONPATH:+:$PYTHONPATH}
export ECA_REPO_ROOT=$repo VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export ECA_ROLLOUT_BACKEND=vexact ECA_FORWARD_ONLY_CAPTURE=1
export ECA_STEP_CAPTURE_JSONL=$root/step_records.jsonl
export INFER_FA_IMPL=triton-invariant VEOMNI_ATTN_IMPLEMENTATION=triton-invariant MODELING_BACKEND=veomni
export VEOMNI_USE_LIGER_KERNEL=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset ECA_CUR_CAPTURE_JSONL ECA_PARITY_DUMP ECA_ROOT_PIVOT ECA_BOUNDARY_TABLE

set +e
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpus" \
  "$vexact_repo/.venv/bin/python" "$repo/scripts/launch_grpo_main.py" \
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
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.veomni.fsdp_size="$ngpus" actor_rollout_ref.actor.veomni.attn_implementation=triton-invariant \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vexact actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.seed=2026081203 actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.temperature=0 actor_rollout_ref.rollout.top_p=1 \
  actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 actor_rollout_ref.rollout.max_num_seqs="$max_num_seqs" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$max_batched_tokens" actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.calculate_log_probs=True actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks="$max_cache_blocks" \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=triton-invariant \
  actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.multi_turn.max_assistant_turns=9 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=8 actor_rollout_ref.rollout.multi_turn.max_tool_response_length=384 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$repo/configs/rl/cur1_candidate_bm25_tool.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=eca_step_adaptive_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$repo/configs/rl/eca_step_adaptive_agent_loop.yaml" \
  reward.custom_reward_function.path="$repo/src/rl/rewards_cur.py" reward.custom_reward_function.name=compute_score \
  trainer.nnodes=1 trainer.n_gpus_per_node="$ngpus" trainer.total_epochs=1 trainer.total_training_steps=1 \
  trainer.val_before_train=False trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
  'trainer.logger=[console]' trainer.project_name=eca_step_adaptive trainer.experiment_name="step_s0_${stage}" \
  trainer.default_local_dir="$root/no_checkpoint" 2>&1 | tee "$root/run.log"
run_rc=${PIPESTATUS[0]}
set -e
echo "step_s0_${stage}_run_rc=$run_rc"
[[ "$run_rc" -eq 0 ]]
grep -q "FORWARD_ONLY: actor backward/optimizer/scheduler are disabled" "$root/run.log"
test "$(wc -l < "$root/step_records.jsonl")" -eq "$batch"
echo "STEP_S0_${stage^^}_EXECUTION_PASS"
