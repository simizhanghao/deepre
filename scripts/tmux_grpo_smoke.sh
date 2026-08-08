#!/usr/bin/env bash
# Host-side Phase 3B2 launcher (detach-safe).
#
# Critical: must use `docker exec -d` so training is owned by the container
# daemon. Plain `nohup docker exec ... &` dies when the host client/session
# ends → container process gets SIGTERM → Ray INTENDED_USER_EXIT (seen at step16).
#
# Usage:
#   STEPS=50 SAVE_FREQ=5 bash scripts/tmux_grpo_smoke.sh
#   tmux attach -t eca-grpo
set -euo pipefail

REPO=${REPO:-/data1/hcc/deepresearch}
SESSION=${SESSION:-eca-grpo}
CONTAINER=${CONTAINER:-eca-verl}
STEPS=${STEPS:-50}
SAVE_FREQ=${SAVE_FREQ:-5}
OUT_DIR=${OUT_DIR:-/workspace/deepresearch/outputs/rl/grpo_sftv1_smoke}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_sftv1_smoke}
RESUME_MODE=${RESUME_MODE:-auto}
TB_PORT=${TB_PORT:-6006}
HOST_TB_DIR=${HOST_TB_DIR:-$REPO/outputs/rl/tensorboard/$EXPERIMENT_NAME}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_NAME=${LOG_NAME:-grpo_${EXPERIMENT_NAME}_to${STEPS}_${STAMP}.log}
LOG=${LOG:-$REPO/logs/$LOG_NAME}
CONTAINER_LOG=/workspace/deepresearch/logs/$LOG_NAME

command -v tmux >/dev/null || { echo "tmux not found"; exit 1; }
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
docker start "$CONTAINER" >/dev/null

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  echo "Kill first if restarting monitor only: tmux kill-session -t $SESSION"
  echo "(Training itself is independent of tmux when launched with docker exec -d.)"
  exit 1
fi

docker exec "$CONTAINER" bash -lc \
  "mkdir -p /workspace/deepresearch/outputs/rl/tensorboard/${EXPERIMENT_NAME} \
            /workspace/deepresearch/outputs/rl/grpo_sftv1_smoke \
            /workspace/deepresearch/logs && chmod -R a+rwX /workspace/deepresearch/outputs/rl /workspace/deepresearch/logs" || true
mkdir -p "$(dirname "$LOG")" "$HOST_TB_DIR" 2>/dev/null || true

# Retriever on host
if ! curl -sf http://127.0.0.1:8001/health >/dev/null; then
  echo "Starting Candidate-BM25 on :8001 ..."
  nohup bash -lc "cd $REPO; (conda activate deepresearch 2>/dev/null || true); python scripts/start_candidate_retrieval_server.py --index data/rl/grpo_smoke_128/contexts_index.jsonl --port 8001" \
    >"$REPO/logs/retriever_8001.log" 2>&1 &
  sleep 3
fi
curl -sf http://127.0.0.1:8001/health >/dev/null || { echo "retriever failed"; exit 1; }
echo RETRIEVER_OK

# Optional: stop leftover Ray from a previous crashed job (keeps GPUs clean)
if [[ "${RAY_STOP_FIRST:-1}" == "1" ]]; then
  docker exec "$CONTAINER" bash -lc \
    'ray stop --force >/dev/null 2>&1 || true; pkill -9 -f "sglang.launch_server|sglang::" >/dev/null 2>&1 || true; true'
  sleep 2
fi

# Detached train: survives SSH/Cursor/tmux client death
: >"$LOG"
docker exec -d "$CONTAINER" bash -lc "\
  export PYTHONPATH=/workspace/deepresearch:/workspace/verl; \
  export TENSORBOARD_DIR=/workspace/deepresearch/outputs/rl/tensorboard/${EXPERIMENT_NAME}; \
  cd /workspace/deepresearch; \
  STEPS=${STEPS} SAVE_FREQ=${SAVE_FREQ} OUT_DIR=${OUT_DIR} \
  EXPERIMENT_NAME=${EXPERIMENT_NAME} RESUME_MODE=${RESUME_MODE} \
  bash /workspace/deepresearch/scripts/run_grpo_smoke.sh \
    >${CONTAINER_LOG} 2>&1"

sleep 2
if ! docker exec "$CONTAINER" bash -lc "pgrep -f launch_grpo_main.py >/dev/null || pgrep -f run_grpo_smoke.sh >/dev/null"; then
  echo "WARN: train process not visible yet; check $LOG"
  docker exec "$CONTAINER" bash -lc "tail -20 ${CONTAINER_LOG} 2>/dev/null || true"
fi
echo "TRAIN_LOG=$LOG (docker exec -d)"

# Monitor tmux only
TB_ROOT=$(dirname "$HOST_TB_DIR")
tmux new-session -d -s "$SESSION" -n grpo -c "$REPO"
tmux send-keys -t "$SESSION:grpo.0" \
  "echo RETRIEVER; curl -s http://127.0.0.1:8001/health; while true; do sleep 3600; done" C-m
tmux split-window -h -t "$SESSION:grpo" -c "$REPO"
tmux send-keys -t "$SESSION:grpo.1" "tail -F $LOG" C-m
tmux split-window -v -t "$SESSION:grpo.1" -c "$REPO"
tmux send-keys -t "$SESSION:grpo.2" \
  "(conda activate deepresearch 2>/dev/null || true); pgrep -f 'tensorboard.*${TB_PORT}' >/dev/null || tensorboard --logdir $TB_ROOT --port $TB_PORT --bind_all --load_fast=false; echo TB http://127.0.0.1:$TB_PORT; tail -F $REPO/logs/tensorboard_6006.log 2>/dev/null || sleep infinity" C-m
tmux select-pane -t "$SESSION:grpo.1"

echo "Started Phase 3B2 (detach-safe): resume → STEPS=$STEPS"
echo "  attach:  tmux attach -t $SESSION"
echo "  log:     $LOG"
echo "  TB:      http://127.0.0.1:$TB_PORT"
echo "  ckpt:    $REPO/outputs/rl/grpo_sftv1_smoke (latest=$(cat $REPO/outputs/rl/grpo_sftv1_smoke/latest_checkpointed_iteration.txt 2>/dev/null || echo none))"
echo "  note:    training survives tmux/SSH kill; only docker stop / ray stop ends it"
