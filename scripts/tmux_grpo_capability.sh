#!/usr/bin/env bash
# Host-side Phase 3D2 Capability-Aware launcher (detach-safe).
# Does NOT build p_int — set ECA_PINT_TABLE or rely on p_int_latest.json.
#
# Example (after building table):
#   ECA_PINT_TABLE=/workspace/deepresearch/outputs/rl/capability/p_int_latest.json \
#   STEPS=50 SAVE_FREQ=50 bash scripts/tmux_grpo_capability.sh
set -euo pipefail

# tmux often drops the docker supplementary group.
if ! docker info >/dev/null 2>&1; then
  if sg docker -c 'docker info' >/dev/null 2>&1; then
    exec sg docker -c "cd \"${PWD}\" && bash \"${BASH_SOURCE[0]}\" ${*@Q}"
  fi
  echo "ERROR: cannot talk to docker (not in docker group / daemon down)" >&2
  exit 1
fi

REPO=${REPO:-/data1/hcc/deepresearch}
SESSION=${SESSION:-eca-grpo-3d2}
CONTAINER=${CONTAINER:-eca-verl}
STEPS=${STEPS:-50}
SAVE_FREQ=${SAVE_FREQ:-50}
OUT_DIR=${OUT_DIR:-/workspace/deepresearch/outputs/rl/grpo_sftv1_cap_3d2}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_sftv1_cap_3d2}
RESUME_MODE=${RESUME_MODE:-disable}
ECA_EVIDENCE_WEIGHT=${ECA_EVIDENCE_WEIGHT:-0.5}
ECA_SEARCH_COST_WEIGHT=${ECA_SEARCH_COST_WEIGHT:-0.30}
ECA_PINT_STRICT=${ECA_PINT_STRICT:-1}
ECA_PINT_DEFAULT=${ECA_PINT_DEFAULT:-0.0}
ECA_PINT_TABLE=${ECA_PINT_TABLE:-/workspace/deepresearch/outputs/rl/capability/p_int_latest.json}
BATCH=${BATCH:-16}
N=${N:-4}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}
MICRO_BATCH=${MICRO_BATCH:-2}
TB_PORT=${TB_PORT:-6009}
HOST_TB_DIR=${HOST_TB_DIR:-$REPO/outputs/rl/tensorboard/$EXPERIMENT_NAME}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_NAME=${LOG_NAME:-grpo_${EXPERIMENT_NAME}_to${STEPS}_${STAMP}.log}
LOG=${LOG:-$REPO/logs/$LOG_NAME}
CONTAINER_LOG=/workspace/deepresearch/logs/$LOG_NAME

# host path check + coverage hard gate
HOST_PINT=${ECA_PINT_TABLE/\/workspace\/deepresearch/$REPO}
if [[ ! -f "$HOST_PINT" && ! -L "$HOST_PINT" ]]; then
  echo "Missing p_int table: $HOST_PINT"
  echo "Build: CUDA_VISIBLE_DEVICES=4 python scripts/build_capability_pint_table.py --model-path outputs/sft_qwen25_3b_coldstart_v1_merged"
  exit 1
fi
python "$REPO/scripts/audit_pint_table.py" \
  --table "$HOST_PINT" \
  --train-parquet "$REPO/data/rl/grpo_smoke_128/train.parquet" \
  --require-full-coverage \
  --out "$REPO/outputs/rl/grpo_sftv1_cap_3d2/pint_audit_pretrain_host.json"

command -v tmux >/dev/null || { echo "tmux not found"; exit 1; }
docker start "$CONTAINER" >/dev/null

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  exit 1
fi

docker exec "$CONTAINER" bash -lc \
  "mkdir -p /workspace/deepresearch/outputs/rl/tensorboard/${EXPERIMENT_NAME} \
            /workspace/deepresearch/outputs/rl/grpo_sftv1_cap_3d2 \
            /workspace/deepresearch/outputs/rl/capability \
            /workspace/deepresearch/logs && chmod -R a+rwX /workspace/deepresearch/outputs/rl /workspace/deepresearch/logs" || true
mkdir -p "$(dirname "$LOG")" "$HOST_TB_DIR" 2>/dev/null || true

if ! curl -sf http://127.0.0.1:8001/health >/dev/null; then
  echo "Starting Candidate-BM25 on :8001 ..."
  nohup bash -lc "cd $REPO; (conda activate deepresearch 2>/dev/null || true); python scripts/start_candidate_retrieval_server.py --index data/rl/grpo_smoke_128/contexts_index.jsonl --port 8001" \
    >"$REPO/logs/retriever_8001.log" 2>&1 &
  sleep 3
fi
curl -sf http://127.0.0.1:8001/health >/dev/null || { echo "retriever failed"; exit 1; }
echo RETRIEVER_OK

if [[ "${RAY_STOP_FIRST:-1}" == "1" ]]; then
  docker exec "$CONTAINER" bash -lc \
    'ray stop --force >/dev/null 2>&1 || true; pkill -9 -f sglang.launch_server >/dev/null 2>&1 || true; exit 0' \
    || true
  sleep 2
fi

: >"$LOG"
ln -sfn "$LOG" "$REPO/logs/grpo_3d2_latest.log"
docker exec -d "$CONTAINER" bash -lc "\
  export PYTHONPATH=/workspace/deepresearch:/workspace/verl; \
  export TENSORBOARD_DIR=/workspace/deepresearch/outputs/rl/tensorboard/${EXPERIMENT_NAME}; \
  export ECA_EVIDENCE_WEIGHT=${ECA_EVIDENCE_WEIGHT}; \
  export ECA_SEARCH_COST_WEIGHT=${ECA_SEARCH_COST_WEIGHT}; \
  export ECA_PINT_TABLE=${ECA_PINT_TABLE}; \
  export ECA_PINT_DEFAULT=${ECA_PINT_DEFAULT}; \
  export ECA_PINT_STRICT=${ECA_PINT_STRICT}; \
  cd /workspace/deepresearch; \
  STEPS=${STEPS} SAVE_FREQ=${SAVE_FREQ} OUT_DIR=${OUT_DIR} \
  EXPERIMENT_NAME=${EXPERIMENT_NAME} RESUME_MODE=${RESUME_MODE} \
  TOTAL_EPOCHS=${TOTAL_EPOCHS:-${STEPS}} \
  BATCH=${BATCH} N=${N} GPU_MEM_UTIL=${GPU_MEM_UTIL} MICRO_BATCH=${MICRO_BATCH} \
  ECA_EVIDENCE_WEIGHT=${ECA_EVIDENCE_WEIGHT} \
  ECA_SEARCH_COST_WEIGHT=${ECA_SEARCH_COST_WEIGHT} \
  ECA_PINT_TABLE=${ECA_PINT_TABLE} \
  ECA_PINT_DEFAULT=${ECA_PINT_DEFAULT} \
  ECA_PINT_STRICT=${ECA_PINT_STRICT} \
  bash /workspace/deepresearch/scripts/run_grpo_capability.sh \
    >${CONTAINER_LOG} 2>&1"

sleep 3
if ! docker exec "$CONTAINER" bash -lc "pgrep -f launch_grpo_main.py >/dev/null || pgrep -f run_grpo_capability.sh >/dev/null"; then
  echo "WARN: train process not visible yet; check $LOG"
  docker exec "$CONTAINER" bash -lc "tail -30 ${CONTAINER_LOG} 2>/dev/null || true"
fi

TB_ROOT=$(dirname "$HOST_TB_DIR")
tmux new-session -d -s "$SESSION" -n grpo -c "$REPO"
tmux send-keys -t "$SESSION:grpo.0" \
  "echo RETRIEVER; curl -s http://127.0.0.1:8001/health; while true; do sleep 3600; done" C-m
tmux split-window -h -t "$SESSION:grpo" -c "$REPO"
tmux send-keys -t "$SESSION:grpo.1" "tail -F $LOG" C-m
tmux split-window -v -t "$SESSION:grpo.1" -c "$REPO"
tmux send-keys -t "$SESSION:grpo.2" \
  "(conda activate deepresearch 2>/dev/null || true); pgrep -f 'tensorboard.*${TB_PORT}' >/dev/null || tensorboard --logdir $TB_ROOT --port $TB_PORT --bind_all --load_fast=false; echo TB http://127.0.0.1:$TB_PORT; sleep infinity" C-m
tmux select-pane -t "$SESSION:grpo.1"

echo "Started Phase 3D2 Capability-Aware (NOT executed by docs-only workflow unless you run this)."
echo "  attach:  tmux attach -t $SESSION"
echo "  log:     $LOG"
echo "  TB:      http://127.0.0.1:$TB_PORT"
echo "  p_int:   $ECA_PINT_TABLE"
echo "  stop if: search extinction + answer collapse / KL explode / NaN"
