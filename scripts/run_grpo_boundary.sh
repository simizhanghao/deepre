#!/usr/bin/env bash
# Phase 3D2b Search-Boundary-Aware GRPO — Stage-II from 3C@400 HF (NOT fresh SFT).
# Boundary table required (ECA_BOUNDARY_TABLE).
#
# Defaults: λ_e=0.5, α=0.30 (NoSearch only), STEPS=50.
set -euo pipefail

REPO=${REPO:-/workspace/deepresearch}
VERL_ROOT=${VERL_ROOT:-/workspace/verl}
export PYTHONPATH="${REPO}:${VERL_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ECA_EVIDENCE_WEIGHT=${ECA_EVIDENCE_WEIGHT:-0.5}
export ECA_SEARCH_COST_WEIGHT=${ECA_SEARCH_COST_WEIGHT:-0.30}
export ECA_BOUNDARY_STRICT=${ECA_BOUNDARY_STRICT:-1}
export ECA_BOUNDARY_DEFAULT=${ECA_BOUNDARY_DEFAULT:-Undetermined}

ECA_BOUNDARY_TABLE=${ECA_BOUNDARY_TABLE:-$REPO/outputs/rl/boundary/boundary_latest.json}
export ECA_BOUNDARY_TABLE

TRAIN_FILE=${TRAIN_FILE:-$REPO/data/rl/grpo_smoke_128/train.parquet}
VAL_FILE=${VAL_FILE:-$REPO/data/rl/grpo_smoke_128/val.parquet}
# Stage II: init from 3C@400 HF merge
MODEL_PATH=${MODEL_PATH:-$REPO/outputs/rl/hf_merged/grpo_sftv1_evidence_3c_step400}
TOOL_CFG=$REPO/configs/rl/candidate_bm25_tool.yaml
AGENT_CFG=$REPO/configs/rl/eca_agent_loop.yaml
REWARD_PATH=$REPO/src/rl/rewards_3d2b.py
OUT_DIR=${OUT_DIR:-$REPO/outputs/rl/grpo_3c400_boundary_3d2b}
STEPS=${STEPS:-50}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-$STEPS}
BATCH=${BATCH:-16}
N=${N:-4}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}
MICRO_BATCH=${MICRO_BATCH:-2}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
SAVE_FREQ=${SAVE_FREQ:-50}
RESUME_MODE=${RESUME_MODE:-disable}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_3c400_boundary_3d2b}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$REPO/outputs/rl/tensorboard/${EXPERIMENT_NAME}}
mkdir -p "$TENSORBOARD_DIR" "$OUT_DIR"

test -f "$TRAIN_FILE"
test -d "$MODEL_PATH"
test -f "$ECA_BOUNDARY_TABLE" || {
  echo "Missing ECA_BOUNDARY_TABLE=$ECA_BOUNDARY_TABLE"
  echo "Build: python scripts/build_search_boundary_table.py --model-path ..."
  exit 1
}
python "$REPO/scripts/audit_boundary_table.py" \
  --table "$ECA_BOUNDARY_TABLE" \
  --train-parquet "$TRAIN_FILE" \
  --require-full-coverage \
  --out "$OUT_DIR/boundary_audit_pretrain.json"
curl -sf http://127.0.0.1:8001/health >/dev/null || {
  echo "Candidate-BM25 server not up on :8001"
  exit 1
}
python -c "import transfer_queue" 2>/dev/null || pip3 install -q TransferQueue

echo "[3D2b] STEPS=$STEPS BATCH=$BATCH N=$N GPU_MEM_UTIL=$GPU_MEM_UTIL MICRO_BATCH=$MICRO_BATCH"
echo "[3D2b] λ_e=$ECA_EVIDENCE_WEIGHT α=$ECA_SEARCH_COST_WEIGHT STRICT=$ECA_BOUNDARY_STRICT"
echo "[3D2b] boundary=$ECA_BOUNDARY_TABLE"
echo "[3D2b] OUT=$OUT_DIR resume=$RESUME_MODE model=$MODEL_PATH"

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
