#!/usr/bin/env bash
# Phase 3D2 Capability-Aware GRPO — fresh from SFT-v1 (NOT from 3C@400).
# R = EM + λ_e(1−p_int)EvidF1 + λ_f Format − λ_s p_int 1[N_s>0]
#
# Requires ECA_PINT_TABLE (JSON). Build with scripts/build_capability_pint_table.py
# Defaults: λ_e=0.5, λ_s=0.30 (provisional), STEPS=50 (one refresh window).
set -euo pipefail

REPO=${REPO:-/workspace/deepresearch}
VERL_ROOT=${VERL_ROOT:-/workspace/verl}
export PYTHONPATH="${REPO}:${VERL_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ECA_EVIDENCE_WEIGHT=${ECA_EVIDENCE_WEIGHT:-0.5}
export ECA_SEARCH_COST_WEIGHT=${ECA_SEARCH_COST_WEIGHT:-0.30}
# Train: missing sample_id is FATAL. Default only for selftest with STRICT=0.
export ECA_PINT_STRICT=${ECA_PINT_STRICT:-1}
export ECA_PINT_DEFAULT=${ECA_PINT_DEFAULT:-0.0}

# Host/container path to frozen p_int table (required).
ECA_PINT_TABLE=${ECA_PINT_TABLE:-$REPO/outputs/rl/capability/p_int_latest.json}
export ECA_PINT_TABLE

TRAIN_FILE=${TRAIN_FILE:-$REPO/data/rl/grpo_smoke_128/train.parquet}
VAL_FILE=${VAL_FILE:-$REPO/data/rl/grpo_smoke_128/val.parquet}
MODEL_PATH=${MODEL_PATH:-$REPO/outputs/sft_qwen25_3b_coldstart_v1_merged}
TOOL_CFG=$REPO/configs/rl/candidate_bm25_tool.yaml
AGENT_CFG=$REPO/configs/rl/eca_agent_loop.yaml
REWARD_PATH=$REPO/src/rl/rewards_3d2.py
OUT_DIR=${OUT_DIR:-$REPO/outputs/rl/grpo_sftv1_cap_3d2}
STEPS=${STEPS:-50}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-$STEPS}
BATCH=${BATCH:-16}
N=${N:-4}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}
MICRO_BATCH=${MICRO_BATCH:-2}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
SAVE_FREQ=${SAVE_FREQ:-50}
RESUME_MODE=${RESUME_MODE:-disable}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_sftv1_cap_3d2}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$REPO/outputs/rl/tensorboard/${EXPERIMENT_NAME}}
mkdir -p "$TENSORBOARD_DIR" "$OUT_DIR"

test -f "$TRAIN_FILE"
test -d "$MODEL_PATH"
test -f "$ECA_PINT_TABLE" || {
  echo "Missing ECA_PINT_TABLE=$ECA_PINT_TABLE"
  echo "Build first: python scripts/build_capability_pint_table.py --model-path ..."
  exit 1
}
# Hard gate: coverage must be 100% vs train parquet
python "$REPO/scripts/audit_pint_table.py" \
  --table "$ECA_PINT_TABLE" \
  --train-parquet "$TRAIN_FILE" \
  --require-full-coverage \
  --out "$OUT_DIR/pint_audit_pretrain.json"
curl -sf http://127.0.0.1:8001/health >/dev/null || {
  echo "Candidate-BM25 server not up on :8001"
  exit 1
}
python -c "import transfer_queue" 2>/dev/null || pip3 install -q TransferQueue

echo "[3D2] STEPS=$STEPS BATCH=$BATCH N=$N GPU_MEM_UTIL=$GPU_MEM_UTIL MICRO_BATCH=$MICRO_BATCH"
echo "[3D2] λ_e=$ECA_EVIDENCE_WEIGHT λ_s=$ECA_SEARCH_COST_WEIGHT STRICT=$ECA_PINT_STRICT"
echo "[3D2] p_int=$ECA_PINT_TABLE"
echo "[3D2] OUT=$OUT_DIR resume=$RESUME_MODE model=$MODEL_PATH"

cd "$VERL_ROOT"
python "$REPO/scripts/launch_grpo_main.py" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$BATCH" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="$N" \
  actor_rollout_ref.rollout.temperature=0.9 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CFG" \
  actor_rollout_ref.rollout.agent.default_agent_loop=eca_search_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_CFG" \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=compute_score \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$STEPS" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq=-1 \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=eca_phase3d \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUT_DIR"
