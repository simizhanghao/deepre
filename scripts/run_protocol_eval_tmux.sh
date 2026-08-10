#!/usr/bin/env bash
# Launch Phase 2D3-C protocol evals inside tmux (survives laptop disconnect).
# Usage (on server, from repo root):
#   bash scripts/run_protocol_eval_tmux.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SESSION="${SESSION:-protocol_sft_v1}"
MERGED="${MERGED:-outputs/00_sft_v1_merged}"
TAG="${TAG:-sft_v1}"
LOGDIR="${LOGDIR:-logs/07_protocol_${TAG}}"
mkdir -p "$LOGDIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  echo "Or kill it first: tmux kill-session -t $SESSION"
  exit 1
fi

# One window, three panes; jobs keep running after detach.
tmux new-session -d -s "$SESSION" -n protocol -c "$REPO_ROOT"

run_one() {
  local gpu="$1"
  local mode="$2"
  local pane="$3"
  local cmd="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate deepresearch && cd \"$REPO_ROOT\" && CUDA_VISIBLE_DEVICES=${gpu} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_protocol_eval.py --mode ${mode} --model-path ${MERGED} --max-samples 200 --run-tag ${TAG} 2>&1 | tee ${LOGDIR}/${mode}.log; echo DONE_${mode}; exec bash"
  if [[ "$pane" == "0" ]]; then
    tmux send-keys -t "${SESSION}:0.0" "$cmd" Enter
  else
    tmux split-window -t "${SESSION}:0" -h -c "$REPO_ROOT"
    tmux send-keys -t "${SESSION}:0.${pane}" "$cmd" Enter
  fi
}

# pane 0
tmux send-keys -t "${SESSION}:0.0" \
  "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate deepresearch && cd \"$REPO_ROOT\" && CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_protocol_eval.py --mode evidence_oracle --model-path ${MERGED} --max-samples 200 --run-tag ${TAG} 2>&1 | tee ${LOGDIR}/evidence_oracle.log; echo DONE_evidence_oracle; exec bash" \
  Enter

tmux split-window -t "${SESSION}:0" -h -c "$REPO_ROOT"
tmux send-keys -t "${SESSION}:0.1" \
  "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate deepresearch && cd \"$REPO_ROOT\" && CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_protocol_eval.py --mode evidence_candidate --model-path ${MERGED} --max-samples 200 --run-tag ${TAG} 2>&1 | tee ${LOGDIR}/evidence_candidate.log; echo DONE_evidence_candidate; exec bash" \
  Enter

tmux split-window -t "${SESSION}:0" -v -c "$REPO_ROOT"
tmux send-keys -t "${SESSION}:0.2" \
  "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate deepresearch && cd \"$REPO_ROOT\" && CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_protocol_eval.py --mode routing --model-path ${MERGED} --max-samples 200 --run-tag ${TAG} 2>&1 | tee ${LOGDIR}/routing.log; echo DONE_routing; exec bash" \
  Enter

tmux select-layout -t "${SESSION}:0" tiled

echo "Started tmux session: $SESSION"
echo "  attach:  tmux attach -t $SESSION"
echo "  detach:  Ctrl-b 然后按 d"
echo "  logs:    $LOGDIR/"
echo "You can close the laptop after detach; jobs keep running on the server."
