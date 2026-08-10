#!/usr/bin/env bash
# Wait for boundary bootstrap (or skip if done), then launch 4-GPU Stage-II GRPO.
# Safe to nohup while build_search_boundary_table.py is already running.
#
# Usage:
#   BOOTSTRAP_PID=125475 nohup bash scripts/run_boundary_bootstrap_then_train.sh \
#     > logs/06_grpo_boundary_chain.log 2>&1 &
set -euo pipefail

REPO=${REPO:-/data1/hcc/deepresearch}
cd "$REPO"

BOOTSTRAP_PID=${BOOTSTRAP_PID:-}
BOOTSTRAP_LOG=${BOOTSTRAP_LOG:-$REPO/logs/04_boundary_bootstrap.log}
BOUNDARY_LATEST=${BOUNDARY_LATEST:-$REPO/outputs/rl/04_table_search_boundary/boundary_latest.json}
CHAIN_LOG_TAG="[3D2b-chain $(date +%H:%M:%S)]"

log() { echo "$CHAIN_LOG_TAG $*"; }

# --- 1) wait for bootstrap process if provided / discovered ---
# Prefer real python argv (avoid matching the wrapping bash cmdline that embeds the same string).
if [[ -z "$BOOTSTRAP_PID" ]]; then
  BOOTSTRAP_PID=$(
    ps -eo pid=,args= | awk '
      $2 ~ /(^|\/)python([0-9.]+)?$/ && /scripts\/build_search_boundary_table\.py/ {
        print $1; exit
      }'
  )
fi

if [[ -n "${BOOTSTRAP_PID}" ]]; then
  if kill -0 "$BOOTSTRAP_PID" 2>/dev/null; then
    log "waiting for bootstrap pid=$BOOTSTRAP_PID ..."
    # wait without failing if process already exited
    while kill -0 "$BOOTSTRAP_PID" 2>/dev/null; do
      sleep 30
    done
    log "bootstrap pid=$BOOTSTRAP_PID exited"
  else
    log "bootstrap pid=$BOOTSTRAP_PID not running (already done?)"
  fi
else
  log "no bootstrap pid found; will require existing boundary table"
fi

# --- 2) success gate ---
sleep 2
if [[ ! -e "$BOUNDARY_LATEST" ]]; then
  log "ERROR: missing $BOUNDARY_LATEST after bootstrap"
  if [[ -f "$BOOTSTRAP_LOG" ]]; then
    tail -40 "$BOOTSTRAP_LOG" || true
  fi
  exit 1
fi

if [[ -f "$BOOTSTRAP_LOG" ]]; then
  if grep -q 'Traceback' "$BOOTSTRAP_LOG" && ! grep -q '\[boundary\] wrote' "$BOOTSTRAP_LOG"; then
    log "ERROR: bootstrap log has Traceback and no write"
    tail -40 "$BOOTSTRAP_LOG" || true
    exit 1
  fi
fi

PY=${PYTHON:-}
if [[ -z "$PY" ]]; then
  if [[ -x /home/hanchengcheng/miniconda3/envs/deepresearch/bin/python ]]; then
    PY=/home/hanchengcheng/miniconda3/envs/deepresearch/bin/python
  else
    PY=python3
  fi
fi
"$PY" "$REPO/scripts/audit_boundary_table.py" \
  --table "$BOUNDARY_LATEST" \
  --train-parquet "$REPO/data/rl/train_smoke_128/train.parquet" \
  --require-full-coverage \
  --out "$REPO/outputs/rl/04_table_search_boundary/boundary_audit_pre_chain.json"
log "boundary audit PASS → $(readlink -f "$BOUNDARY_LATEST" 2>/dev/null || realpath "$BOUNDARY_LATEST")"

# Free GPU4 used by bootstrap HF load before 4-GPU train (container maps host 4–7 → 0–3)
sleep 5

# --- 3) launch Stage-II train (4 GPUs inside eca-verl) ---
export STEPS=${STEPS:-50}
export SAVE_FREQ=${SAVE_FREQ:-50}
export SESSION=${SESSION:-eca-grpo-3d2b}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_boundary}
export OUT_DIR=${OUT_DIR:-/workspace/deepresearch/outputs/rl/06_ckpt_grpo_boundary}
export MODEL_PATH=${MODEL_PATH:-/workspace/deepresearch/outputs/rl/03_hf_evidence_step400}
export ECA_BOUNDARY_TABLE=${ECA_BOUNDARY_TABLE:-/workspace/deepresearch/outputs/rl/04_table_search_boundary/boundary_latest.json}
export ECA_BOUNDARY_STRICT=${ECA_BOUNDARY_STRICT:-1}
export ECA_EVIDENCE_WEIGHT=${ECA_EVIDENCE_WEIGHT:-0.5}
export ECA_SEARCH_COST_WEIGHT=${ECA_SEARCH_COST_WEIGHT:-0.30}
export BATCH=${BATCH:-16}
export N=${N:-4}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}
export MICRO_BATCH=${MICRO_BATCH:-2}
export RAY_STOP_FIRST=${RAY_STOP_FIRST:-1}

# Drop stale session so auto-launch does not abort
if tmux has-session -t "$SESSION" 2>/dev/null; then
  log "tmux session $SESSION exists — killing for clean auto-start"
  tmux kill-session -t "$SESSION" || true
  sleep 1
fi

log "launching tmux_grpo_boundary.sh (4-GPU Stage-II, STEPS=$STEPS) ..."
bash "$REPO/scripts/tmux_grpo_boundary.sh"
log "DONE launch. attach: tmux attach -t $SESSION"
log "train log symlink: $REPO/logs/06_grpo_boundary_latest.log"
