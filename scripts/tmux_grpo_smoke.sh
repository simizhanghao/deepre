#!/usr/bin/env bash
# Host-side tmux launcher for Phase 3B GRPO smoke / continue.
# Pane 0: Candidate-BM25 retriever on :8001
# Pane 1: docker exec → run_grpo_smoke.sh (detach-safe)
#
# Usage (host):
#   bash scripts/tmux_grpo_smoke.sh                  # fresh/continue default OUT_DIR, STEPS=50
#   STEPS=100 SAVE_FREQ=10 bash scripts/tmux_grpo_smoke.sh
#   tmux attach -t eca-grpo
#   # detach: Ctrl-b d
set -euo pipefail

REPO=${REPO:-/data1/hcc/deepresearch}
SESSION=${SESSION:-eca-grpo}
CONTAINER=${CONTAINER:-eca-verl}
STEPS=${STEPS:-50}
SAVE_FREQ=${SAVE_FREQ:-5}
OUT_DIR=${OUT_DIR:-/workspace/deepresearch/outputs/rl/grpo_sftv1_smoke}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_sftv1_smoke}
RESUME_MODE=${RESUME_MODE:-auto}

command -v tmux >/dev/null || { echo "tmux not found"; exit 1; }
docker start "$CONTAINER" >/dev/null

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  echo "Kill first if restarting: tmux kill-session -t $SESSION"
  exit 1
fi

# Window layout: retriever | trainer
tmux new-session -d -s "$SESSION" -n grpo -c "$REPO"
tmux send-keys -t "$SESSION:grpo.0" \
  "cd $REPO && conda activate deepresearch 2>/dev/null; python scripts/start_candidate_retrieval_server.py --index data/rl/grpo_smoke_128/contexts_index.jsonl --port 8001" C-m

tmux split-window -h -t "$SESSION:grpo" -c "$REPO"
tmux send-keys -t "$SESSION:grpo.1" \
  "sleep 3; curl -sf http://127.0.0.1:8001/health && echo RETRIEVER_OK; docker exec -it $CONTAINER bash -lc 'export PYTHONPATH=/workspace/deepresearch:/workspace/verl; STEPS=$STEPS SAVE_FREQ=$SAVE_FREQ OUT_DIR=$OUT_DIR EXPERIMENT_NAME=$EXPERIMENT_NAME RESUME_MODE=$RESUME_MODE bash /workspace/deepresearch/scripts/run_grpo_smoke.sh'" C-m

echo "Started tmux session: $SESSION"
echo "  attach:  tmux attach -t $SESSION"
echo "  detach:  Ctrl-b d"
echo "  TB dir:  $REPO/outputs/rl/tensorboard/$EXPERIMENT_NAME"
echo "  ckpt:    $REPO/outputs/rl/grpo_sftv1_smoke (host path)"
echo "  resume:  RESUME_MODE=$RESUME_MODE STEPS=$STEPS (must be > last ckpt step)"
echo "  TB view: tensorboard --logdir $REPO/outputs/rl/tensorboard/$EXPERIMENT_NAME --port 6006 --bind_all"
