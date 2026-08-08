#!/usr/bin/env bash
# Phase 3B1 launcher (after 3B0 artifacts exist). Do NOT run until mask audit passes.
# Inside eca-verl container, GPUs 4-7 already visible as cuda:0-3.
#
# Resume (default auto): if OUT_DIR/latest_checkpointed_iteration.txt exists,
# veRL loads that ckpt and continues. Raise STEPS above the last saved step.
# Example continue from step 5 → 50:
#   STEPS=50 OUT_DIR=.../grpo_sftv1_smoke bash scripts/run_grpo_smoke.sh
set -euo pipefail

REPO=${REPO:-/workspace/deepresearch}
VERL_ROOT=${VERL_ROOT:-/workspace/verl}
export PYTHONPATH="${REPO}:${VERL_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

TRAIN_FILE=${TRAIN_FILE:-$REPO/data/rl/grpo_smoke_128/train.parquet}
VAL_FILE=${VAL_FILE:-$REPO/data/rl/grpo_smoke_128/val.parquet}
MODEL_PATH=${MODEL_PATH:-$REPO/outputs/sft_qwen25_3b_coldstart_v1_merged}
TOOL_CFG=$REPO/configs/rl/candidate_bm25_tool.yaml
AGENT_CFG=$REPO/configs/rl/eca_agent_loop.yaml
REWARD_PATH=$REPO/src/rl/rewards_3b.py
OUT_DIR=${OUT_DIR:-$REPO/outputs/rl/grpo_sftv1_smoke}
STEPS=${STEPS:-5}
BATCH=${BATCH:-8}
N=${N:-4}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}
# Skip val_before_train for first smoke (val path dies if rollout actors crash mid-batch)
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
SAVE_FREQ=${SAVE_FREQ:-5}
RESUME_MODE=${RESUME_MODE:-auto}   # auto | disable | resume_path
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_sftv1_smoke}
# TensorBoard (host-readable via mount)
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$REPO/outputs/rl/tensorboard/${EXPERIMENT_NAME}}
mkdir -p "$TENSORBOARD_DIR" "$OUT_DIR"

# Preflight
test -f "$TRAIN_FILE"
test -d "$MODEL_PATH"
curl -sf http://127.0.0.1:8001/health >/dev/null || {
  echo "Candidate-BM25 server not up on :8001"
  echo "Host: python scripts/start_candidate_retrieval_server.py"
  exit 1
}
# veRL V1 trainer requires TransferQueue (pip package name TransferQueue, import transfer_queue)
python -c "import transfer_queue" 2>/dev/null || pip3 install -q TransferQueue

if [[ -f "$OUT_DIR/latest_checkpointed_iteration.txt" ]]; then
  LAST=$(tr -d '[:space:]' <"$OUT_DIR/latest_checkpointed_iteration.txt" || true)
  echo "[resume] OUT_DIR=$OUT_DIR last_ckpt_step=${LAST:-none} resume_mode=$RESUME_MODE target_steps=$STEPS"
  echo "[resume] TB_DIR=$TENSORBOARD_DIR"
else
  echo "[fresh] no ckpt in $OUT_DIR — training from scratch (resume_mode=$RESUME_MODE)"
  echo "[fresh] TB_DIR=$TENSORBOARD_DIR"
fi

# Same-process launch: sgl055 file patch + Phase3B2 metrics monkeypatch + main_ppo
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
  trainer.total_epochs=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq=-1 \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=eca_phase3b \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUT_DIR"
