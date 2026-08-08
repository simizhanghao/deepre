#!/usr/bin/env bash
# Phase 3C Evidence GRPO — from SFT-v1 (NOT from 3B step100).
# Default: λ_e=0.5, STEPS=500, SAVE_FREQ=25, same infra knobs as 3B.
set -euo pipefail

REPO=${REPO:-/workspace/deepresearch}
VERL_ROOT=${VERL_ROOT:-/workspace/verl}
export PYTHONPATH="${REPO}:${VERL_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ECA_EVIDENCE_WEIGHT=${ECA_EVIDENCE_WEIGHT:-0.5}

TRAIN_FILE=${TRAIN_FILE:-$REPO/data/rl/grpo_smoke_128/train.parquet}
VAL_FILE=${VAL_FILE:-$REPO/data/rl/grpo_smoke_128/val.parquet}
MODEL_PATH=${MODEL_PATH:-$REPO/outputs/sft_qwen25_3b_coldstart_v1_merged}
TOOL_CFG=$REPO/configs/rl/candidate_bm25_tool.yaml
AGENT_CFG=$REPO/configs/rl/eca_agent_loop.yaml
REWARD_PATH=$REPO/src/rl/rewards_3c.py
OUT_DIR=${OUT_DIR:-$REPO/outputs/rl/grpo_sftv1_evidence_3c}
STEPS=${STEPS:-500}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-$STEPS}
BATCH=${BATCH:-8}
N=${N:-4}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
SAVE_FREQ=${SAVE_FREQ:-25}
RESUME_MODE=${RESUME_MODE:-auto}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_sftv1_evidence_3c}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$REPO/outputs/rl/tensorboard/${EXPERIMENT_NAME}}
mkdir -p "$TENSORBOARD_DIR" "$OUT_DIR"

test -f "$TRAIN_FILE"
test -d "$MODEL_PATH"
curl -sf http://127.0.0.1:8001/health >/dev/null || {
  echo "Candidate-BM25 server not up on :8001"
  exit 1
}
python -c "import transfer_queue" 2>/dev/null || pip3 install -q TransferQueue

# Hard gate: ground_truth must carry supporting_facts for Evidence reward.
python - <<PY
import datasets
ds = datasets.Dataset.from_parquet("$TRAIN_FILE")
gt = ds[0]["reward_model"]["ground_truth"]
assert isinstance(gt, dict) and gt.get("supporting_facts"), (
    "train.parquet missing reward_model.ground_truth.supporting_facts — rebuild with build_grpo_smoke_dataset.py"
)
print(f"[ok] SF in GT n_sf={len(gt['supporting_facts'])} λ_e={__import__('os').environ.get('ECA_EVIDENCE_WEIGHT')}")
PY

if [[ -f "$OUT_DIR/latest_checkpointed_iteration.txt" ]]; then
  LAST=$(tr -d '[:space:]' <"$OUT_DIR/latest_checkpointed_iteration.txt" || true)
  echo "[resume] OUT_DIR=$OUT_DIR last=${LAST:-none} target_steps=$STEPS"
else
  echo "[fresh] 3C from SFT-v1 → STEPS=$STEPS λ_e=$ECA_EVIDENCE_WEIGHT OUT_DIR=$OUT_DIR"
fi

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
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="$N" \
  actor_rollout_ref.rollout.temperature=0.9 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
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
  trainer.project_name=eca_phase3c \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUT_DIR"
