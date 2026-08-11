#!/usr/bin/env bash
# Evidence@400 fixed-policy attribution capture: rollout/reward/logprob/dump only.
set -euo pipefail

repo=/data1/hcc/deepresearch
vexact_repo=/data1/hcc/eca-verl-vexact
batches=
validate_only=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --batches) batches=$2; shift 2 ;;
    --validate-only) validate_only=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$batches" == 2 || "$batches" == 10 ]] || {
  echo "required: --batches 2|10" >&2; exit 2
}

stage=smoke
[[ "$batches" == 10 ]] && stage=full
root=$repo/results/19_optimizer_attribution/$stage
raw=$root/raw
root_logits=$root/root_logits
log=$root/fixed_policy_capture.log
model=$repo/outputs/rl/03_hf_evidence_step400
train=$repo/results/18_boundary_exact_rollout/data/train.parquet
val=$repo/results/18_boundary_exact_rollout/data/val.parquet
boundary=$repo/outputs/rl/04_table_search_boundary/boundary_latest.json

for path in "$model/config.json" "$train" "$val" "$boundary"; do
  test -e "$path" || { echo "missing frozen input: $path" >&2; exit 3; }
done
curl -fsS http://127.0.0.1:8001/health >/dev/null || {
  echo "Candidate-BM25 server is not healthy on :8001" >&2; exit 3
}

if [[ "$validate_only" -eq 0 && -d "$root" ]]; then
  archive=$repo/results/19_optimizer_attribution/attempts/${stage}_$(date +%Y%m%d_%H%M%S)
  mkdir -p "$(dirname "$archive")"
  mv -- "$root" "$archive"
  echo "ARCHIVED_PREVIOUS_CAPTURE=$archive"
fi
mkdir -p "$raw" "$root_logits"

export PYTHONPATH=$repo${PYTHONPATH:+:$PYTHONPATH}
export ECA_REPO_ROOT=$repo
export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export ECA_ROLLOUT_BACKEND=vexact
export ECA_FORWARD_ONLY_CAPTURE=1
export ECA_ATTRIBUTION_CAPTURE_DIR=$raw
export ECA_TRAIN_METRICS_JSONL=$root/train_metrics.jsonl
export ECA_BOUNDARY_TABLE=$boundary
export ECA_BOUNDARY_STRICT=1
export ECA_BOUNDARY_DEFAULT=Undetermined
export ECA_EVIDENCE_WEIGHT=0.5
export ECA_SEARCH_COST_WEIGHT=0.30
export ECA_AUDIT_STOP_MODE=sequence
export ECA_MAX_ASSISTANT_TURN_TOKENS=256
export ECA_FINAL_ANSWER_RESERVE=256
export INFER_FA_IMPL=triton-invariant
export VEOMNI_ATTN_IMPLEMENTATION=triton-invariant
export MODELING_BACKEND=veomni
export VEOMNI_USE_LIGER_KERNEL=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset ECA_SCHEDULE_HORIZON ECA_PARITY_DUMP

runner=("$vexact_repo/.venv/bin/python" "$repo/scripts/launch_grpo_main.py")
if [[ "$validate_only" -eq 1 ]]; then
  runner=("$vexact_repo/.venv/bin/python" "$repo/scripts/validate_verl_config.py")
fi

set +e
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "${runner[@]}" \
  model_engine=veomni \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  data.train_files="$train" data.val_files="$val" \
  data.train_batch_size=16 data.max_prompt_length=1024 data.max_response_length=2048 \
  data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True data.seed=42 \
  actor_rollout_ref.model.path="$model" \
  actor_rollout_ref.model.external_lib=vexact.integrations.verl.fsdp_enable_invariant \
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  +actor_rollout_ref.model.override_config.attn_implementation=triton-invariant \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.veomni.fsdp_size=8 \
  actor_rollout_ref.actor.veomni.attn_implementation=triton-invariant \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vexact actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.seed=42 actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=0.9 actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks=512 \
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
  trainer.nnodes=1 trainer.n_gpus_per_node=8 trainer.total_epochs=50 \
  trainer.total_training_steps="$batches" trainer.val_before_train=False \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
  'trainer.logger=[console]' trainer.project_name=eca_optimizer_attribution \
  trainer.experiment_name="fixed_policy_${stage}" trainer.default_local_dir="$root/no_checkpoint" \
  2>&1 | tee "$log"
run_rc=${PIPESTATUS[0]}
set -e
echo "fixed_policy_${stage}_rc=$run_rc"
[[ "$run_rc" -eq 0 ]]
[[ "$validate_only" -eq 0 ]] || exit 0

grep -q "FORWARD_ONLY: actor backward/optimizer/scheduler are disabled" "$log"
test "$(find "$raw" -maxdepth 1 -name 'step_*.npz' | wc -l)" -eq "$batches"
unique_samples=$(env -u LD_LIBRARY_PATH "$vexact_repo/.venv/bin/python" - "$raw/prompt_manifest.jsonl" <<'PY'
import json,sys
print(len({json.loads(x)["sample_id"] for x in open(sys.argv[1]) if x.strip()}))
PY
)

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
  "$vexact_repo/.venv/bin/python" "$repo/scripts/capture_vexact_exact2.py" \
  --config "$repo/configs/rl/grpo_smoke128.yaml" --seed 42 \
  --output-dir "$root_logits" --max-samples "$unique_samples" --route-probe-only \
  --model-path "$model" --capture-manifest "$raw/prompt_manifest.jsonl" \
  --attn-impl triton-invariant --n-rollouts 0 --temperature 0.9 --top-p 0.95 \
  2>&1 | tee "$root/root_logits.log"

min_mixed=0
[[ "$batches" == 10 ]] && min_mixed=15
env -u LD_LIBRARY_PATH "$vexact_repo/.venv/bin/python" \
  "$repo/scripts/audit_optimizer_attribution.py" \
  --capture-dir "$raw" --root-logits-dir "$root_logits" \
  --model-config "$model/config.json" --expected-trajectories "$((batches * 64))" \
  --n-rollouts 4 --temperature 0.9 --top-p 0.95 \
  --min-mixed-nosearch-groups "$min_mixed" \
  2>&1 | tee "$root/attribution_audit.log"

echo "FIXED_POLICY_${stage^^}_CAPTURE_PASS=$root"
