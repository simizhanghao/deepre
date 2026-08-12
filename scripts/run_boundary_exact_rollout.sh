#!/usr/bin/env bash
# Exact-Rollout Boundary Learning: causal Stage-II segments ending at 10/25/50.
set -euo pipefail

repo=/data1/hcc/deepresearch
vexact_repo=/data1/hcc/eca-verl-vexact
target_step=
validate_only=0
profile=grpo
n_gpus=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-step) target_step=$2; shift 2 ;;
    --profile) profile=$2; shift 2 ;;
    --validate-only) validate_only=1; shift ;;
    --n-gpus) n_gpus=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$target_step" == 10 || "$target_step" == 25 || "$target_step" == 50 ]] || {
  echo "required: --target-step 10|25|50" >&2
  exit 2
}
[[ "$profile" == grpo || "$profile" == rfpp_baseline || "$profile" == grpo_no_std ]] || {
  echo "--profile must be grpo|rfpp_baseline|grpo_no_std" >&2
  exit 2
}
[[ "$n_gpus" == 4 || "$n_gpus" == 8 ]] || {
  echo "--n-gpus must be 4|8" >&2
  exit 2
}
visible_gpus=0,1,2,3
[[ "$n_gpus" == 8 ]] && visible_gpus=0,1,2,3,4,5,6,7

run_root=$repo/results/18_boundary_exact_rollout
ckpt_root=$repo/outputs/rl/07_ckpt_boundary_exact
adv_estimator=grpo
norm_adv_by_std=true
experiment_name=boundary_exact_vexact
eval_script=$repo/scripts/run_boundary_checkpoint_eval.sh
if [[ "$profile" == rfpp_baseline ]]; then
  run_root=$repo/results/20_rfpp_baseline
  ckpt_root=$repo/outputs/rl/08_ckpt_rfpp_baseline
  adv_estimator=reinforce_plus_plus_baseline
  experiment_name=rfpp_baseline_exact_vexact
  eval_script=$repo/scripts/run_rfpp_checkpoint_eval.sh
elif [[ "$profile" == grpo_no_std ]]; then
  run_root=$repo/results/20_grpo_no_std
  ckpt_root=$repo/outputs/rl/09_ckpt_grpo_no_std
  adv_estimator=grpo
  norm_adv_by_std=false
  experiment_name=grpo_no_std_exact_vexact
  eval_script=$repo/scripts/run_grpo_nostd_checkpoint_eval.sh
fi
model_path=$repo/outputs/rl/03_hf_evidence_step400
boundary_table=$repo/outputs/rl/04_table_search_boundary/boundary_latest.json
train_source=$repo/data/rl/train_smoke_128/train.parquet
val_source=$repo/data/rl/train_smoke_128/val.parquet
train_file=$run_root/data/train.parquet
val_file=$run_root/data/val.parquet
metrics_file=$run_root/train_metrics.jsonl
trajectory_file=$run_root/trajectories.jsonl
segment_log=$run_root/logs/train_to_step${target_step}.log
tracker=$ckpt_root/latest_checkpointed_iteration.txt

mkdir -p "$run_root/logs" "$run_root/checkpoints" "$run_root/data" "$ckpt_root"
for path in "$model_path/config.json" "$boundary_table" "$train_source" "$val_source" \
  "$repo/src/rl/rewards_boundary.py"; do
  test -e "$path" || { echo "missing frozen input: $path" >&2; exit 3; }
done
test -x "$vexact_repo/.venv/bin/python"
curl -fsS http://127.0.0.1:8001/health >/dev/null || {
  echo "Candidate-BM25 server is not healthy on :8001" >&2
  exit 3
}

if [[ "$profile" == rfpp_baseline ]]; then
  parity=$run_root/estimator_parity.json
  env -u LD_LIBRARY_PATH "$vexact_repo/.venv/bin/python" \
    "$repo/scripts/verify_rfpp_estimator_parity.py" \
    --capture-npz "$repo/results/19_optimizer_attribution/full/raw/attribution_capture.npz" \
    --output "$parity"
  grep -q '"gate": "RFPP_ESTIMATOR_PARITY_PASS"' "$parity"
fi

for split in train val; do
  src=$train_source; dst=$train_file
  if [[ "$split" == val ]]; then
    src=$val_source; dst=$val_file
  fi
  "$vexact_repo/.venv/bin/python" "$repo/scripts/sanitize_parquet_for_vexact.py" \
    --source "$src" --output "$dst" \
    --manifest "$run_root/data/${split}_sanitize_manifest.json"
done

if [[ "$target_step" == 10 ]]; then
  if [[ -s "$tracker" && "$(<"$tracker")" != 0 ]]; then
    echo "step-10 must start from Evidence@400, but tracker already exists: $(<"$tracker")" >&2
    exit 4
  fi
  resume_mode=disable
else
  expected_previous=10
  [[ "$target_step" == 50 ]] && expected_previous=25
  [[ -s "$tracker" && "$(<"$tracker")" == "$expected_previous" ]] || {
    echo "target $target_step requires checkpoint tracker=$expected_previous" >&2
    exit 4
  }
  resume_mode=auto
fi

# A step-10 retry is a clean Evidence@400 run. Preserve diagnostics from any
# failed pre-checkpoint attempt instead of mixing duplicate steps into JSONL.
if [[ "$target_step" == 10 && "$validate_only" -eq 0 ]]; then
  if [[ -s "$metrics_file" || -s "$trajectory_file" || -s "$segment_log" ]]; then
    attempt_dir=$run_root/attempts/pre_8gpu_$(date +%Y%m%d_%H%M%S)
    mkdir -p "$attempt_dir"
    for prior in "$metrics_file" "$trajectory_file" "$segment_log"; do
      [[ ! -e "$prior" ]] || mv -- "$prior" "$attempt_dir/"
    done
    echo "ARCHIVED_PREVIOUS_ATTEMPT=$attempt_dir"
  fi
fi

"$vexact_repo/.venv/bin/python" "$repo/scripts/audit_boundary_table.py" \
  --table "$boundary_table" --train-parquet "$train_file" \
  --require-full-coverage --out "$run_root/boundary_audit_pretrain.json"

"$vexact_repo/.venv/bin/python" - "$run_root/frozen_contract.json" "$adv_estimator" "$norm_adv_by_std" \
  "$model_path/config.json" "$boundary_table" "$repo/src/rl/rewards_boundary.py" \
  "$train_source" "$val_source" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1])
rows={}
for raw in sys.argv[4:]:
 p=Path(raw).resolve(); rows[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
contract={
 "model":"Evidence@400", "rollout":"VeXact", "algorithm":sys.argv[2],
 "norm_adv_by_std_in_grpo":sys.argv[3].lower()=="true",
 "n":4, "temperature":0.9, "top_p":0.95, "steps":50,
 "evidence_weight":0.5, "search_cost_weight":0.30,
 "files_sha256":rows,
}
if sys.argv[2] == "reinforce_plus_plus_baseline":
 contract["loss_agg_mode"]="token-mean"
if out.exists():
 assert json.loads(out.read_text())==contract,"frozen Boundary@50 contract changed"
else:
 out.write_text(json.dumps(contract,indent=2)+"\n")
print("BOUNDARY_EXACT_FROZEN_CONTRACT_PASS")
PY

export PYTHONPATH=$repo${PYTHONPATH:+:$PYTHONPATH}
export ECA_REPO_ROOT=$repo
export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export ECA_ROLLOUT_BACKEND=vexact
export ECA_SCHEDULE_HORIZON=50
if [[ "$profile" == rfpp_baseline ]]; then
  export ECA_EXPECT_ADV_ESTIMATOR=reinforce_plus_plus_baseline
  export ECA_EXPECT_LOSS_AGG_MODE=token-mean
elif [[ "$profile" == grpo_no_std ]]; then
  export ECA_EXPECT_ADV_ESTIMATOR=grpo
  export ECA_EXPECT_LOSS_AGG_MODE=token-mean
  export ECA_EXPECT_NORM_ADV_BY_STD=false
fi
export ECA_TRAIN_METRICS_JSONL=$metrics_file
export ECA_PARITY_DUMP=$trajectory_file
export ECA_BOUNDARY_TABLE=$boundary_table
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

runner=("$vexact_repo/.venv/bin/python" "$repo/scripts/launch_grpo_main.py")
if [[ "$validate_only" -eq 1 ]]; then
  runner=("$vexact_repo/.venv/bin/python" "$repo/scripts/validate_verl_config.py")
fi

set +e
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$visible_gpus" \
  "${runner[@]}" \
  model_engine=veomni \
  algorithm.adv_estimator="$adv_estimator" \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo="$norm_adv_by_std" \
  data.train_files="$train_file" \
  data.val_files="$val_file" \
  data.train_batch_size=16 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.seed=42 \
  actor_rollout_ref.model.path="$model_path" \
  actor_rollout_ref.model.external_lib=vexact.integrations.verl.fsdp_enable_invariant \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=triton-invariant \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.veomni.fsdp_size="$n_gpus" \
  actor_rollout_ref.actor.veomni.attn_implementation=triton-invariant \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]' \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.name=vexact \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=0.9 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.72 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks=4096 \
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
  trainer.n_gpus_per_node="$n_gpus" \
  trainer.total_epochs=50 \
  trainer.total_training_steps="$target_step" \
  trainer.val_before_train=False \
  trainer.save_freq=5 \
  trainer.test_freq=-1 \
  trainer.resume_mode="$resume_mode" \
  trainer.max_actor_ckpt_to_keep=2 \
  'trainer.logger=[console,tensorboard]' \
  trainer.project_name=eca_boundary_exact \
  trainer.experiment_name="$experiment_name" \
  trainer.default_local_dir="$ckpt_root" \
  2>&1 | tee "$segment_log"
run_rc=${PIPESTATUS[0]}
set -e
echo "boundary_train_to_${target_step}_rc=$run_rc"
[[ "$run_rc" -eq 0 ]]
[[ "$validate_only" -eq 0 ]] || exit 0

[[ -s "$tracker" && "$(<"$tracker")" == "$target_step" ]]
hf_src=$ckpt_root/global_step_${target_step}/actor/huggingface
hf_dst=$run_root/checkpoints/step${target_step}_hf
test -f "$hf_src/config.json"
rm -rf -- "$hf_dst.tmp"
cp -a "$hf_src" "$hf_dst.tmp"
mv "$hf_dst.tmp" "$hf_dst"
echo "BOUNDARY_CHECKPOINT_STEP_${target_step}_READY=$hf_dst"

bash "$eval_script" "$target_step"

# After the newer resumable checkpoint and its lightweight HF artifact have
# both passed evaluation, retire only the explicitly superseded full state.
if [[ "$target_step" == 25 ]]; then
  rm -rf -- "$ckpt_root/global_step_10"
elif [[ "$target_step" == 50 ]]; then
  rm -rf -- "$ckpt_root/global_step_25"
fi
echo "BOUNDARY_SEGMENT_${target_step}_COMPLETE"
