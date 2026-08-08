#!/usr/bin/env bash
# Host-side Phase 3B2 launcher:
#   - starts GRPO via `docker exec` (nohup; survives SSH/tmux detach)
#   - opens tmux for retriever keepalive + log tail + TensorBoard
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
LOG=${LOG:-$REPO/logs/grpo_${EXPERIMENT_NAME}_step_to${STEPS}.log}

command -v tmux >/dev/null || { echo "tmux not found"; exit 1; }
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
docker start "$CONTAINER" >/dev/null

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  echo "Kill first if restarting: tmux kill-session -t $SESSION"
  exit 1
fi

# outputs/rl often root-owned — create via container
docker exec "$CONTAINER" bash -lc \
  "mkdir -p /workspace/deepresearch/outputs/rl/tensorboard/${EXPERIMENT_NAME} \
            /workspace/deepresearch/outputs/rl/grpo_sftv1_smoke \
            /workspace/deepresearch/logs && chmod -R a+rwX /workspace/deepresearch/outputs/rl /workspace/deepresearch/logs" || true
mkdir -p "$(dirname "$LOG")" "$HOST_TB_DIR" 2>/dev/null || true

# Ensure retriever
if ! curl -sf http://127.0.0.1:8001/health >/dev/null; then
  echo "Starting Candidate-BM25 on :8001 ..."
  nohup bash -lc "cd $REPO; (conda activate deepresearch 2>/dev/null || true); python scripts/start_candidate_retrieval_server.py --index data/rl/grpo_smoke_128/contexts_index.jsonl --port 8001" \
    >"$REPO/logs/retriever_8001.log" 2>&1 &
  sleep 3
fi
curl -sf http://127.0.0.1:8001/health >/dev/null || { echo "retriever failed"; exit 1; }
echo RETRIEVER_OK

# Train in docker (host docker CLI — not inside tmux pane)
: >"$LOG"
nohup docker exec "$CONTAINER" bash -lc \
  "export PYTHONPATH=/workspace/deepresearch:/workspace/verl; \
   STEPS=$STEPS SAVE_FREQ=$SAVE_FREQ OUT_DIR=$OUT_DIR \
   EXPERIMENT_NAME=$EXPERIMENT_NAME RESUME_MODE=$RESUME_MODE \
   bash /workspace/deepresearch/scripts/run_grpo_smoke.sh" \
  >>"$LOG" 2>&1 &
echo "TRAIN_LOG=$LOG (docker exec background)"

# Monitor tmux
tmux new-session -d -s "$SESSION" -n grpo -c "$REPO"
tmux send-keys -t "$SESSION:grpo.0" \
  "echo RETRIEVER; curl -s http://127.0.0.1:8001/health; while true; do sleep 3600; done" C-m
tmux split-window -h -t "$SESSION:grpo" -c "$REPO"
tmux send-keys -t "$SESSION:grpo.1" "tail -F $LOG" C-m
tmux split-window -v -t "$SESSION:grpo.1" -c "$REPO"
tmux send-keys -t "$SESSION:grpo.2" \
  "(conda activate deepresearch 2>/dev/null || true); tensorboard --logdir $HOST_TB_DIR --port $TB_PORT --bind_all || python -m tensorboard.main --logdir $HOST_TB_DIR --port $TB_PORT --bind_all" C-m
tmux select-pane -t "$SESSION:grpo.1"

echo "Started Phase 3B2: resume → STEPS=$STEPS"
echo "  attach:  tmux attach -t $SESSION"
echo "  detach:  Ctrl-b d"
echo "  log:     $LOG"
echo "  TB:      http://127.0.0.1:$TB_PORT  dir=$HOST_TB_DIR"
echo "  ckpt:    $REPO/outputs/rl/grpo_sftv1_smoke"
echo "  watch:   [phase3b] answer_reward | zero_std | finish_rate | search_count | KL"
