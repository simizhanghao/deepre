#!/usr/bin/env bash
# Watch a 3D1b GRPO log; kill train on search-collapse / always-search / KL / max step.
# Usage: bash scripts/watch_3d1b_early_stop.sh <logfile> [max_step=60]
set -euo pipefail
LOG=${1:?logfile}
MAX_STEP=${2:-60}
CONTAINER=${CONTAINER:-eca-verl}
COLLAPSE_STREAK=${COLLAPSE_STREAK:-10}
SEARCH_LO=${SEARCH_LO:-0.05}
SEARCH_HI=${SEARCH_HI:-0.95}
ALWAYS_CHECK_STEP=${ALWAYS_CHECK_STEP:-50}
KL_KILL=${KL_KILL:-0.25}

kill_train() {
  local why="$1"
  echo "[watch] EARLY_STOP reason=$why"
  docker exec "$CONTAINER" bash -lc \
    'pkill -9 -f launch_grpo_main >/dev/null 2>&1; pkill -9 -f run_grpo_cost >/dev/null 2>&1; ray stop --force >/dev/null 2>&1; exit 0' \
    || true
}

echo "[watch] log=$LOG max_step=$MAX_STEP"

streak=0
last_step=0
while true; do
  if ! docker exec "$CONTAINER" bash -lc 'pgrep -f launch_grpo_main >/dev/null' 2>/dev/null; then
    # maybe still in run_grpo_cost startup
    if ! docker exec "$CONTAINER" bash -lc 'pgrep -f run_grpo_cost >/dev/null' 2>/dev/null; then
      if [[ "$last_step" -gt 0 ]]; then
        echo "[watch] train exited naturally step~$last_step"
        exit 0
      fi
    fi
  fi

  if [[ ! -f "$LOG" ]]; then
    sleep 5
    continue
  fi

  line=$(grep -oE '\[phase3c\] step=[0-9]+ answer=[0-9.]+ \| evidence=[0-9.]+ \| format=[0-9.]+ \| total=[0-9.]+ \| total_std=[0-9.]+ \| zero_std=[0-9.]+ \| finish=[0-9.]+ \| search=[0-9.]+ \| search_rate=[0-9.]+ \| ev_f1=[0-9.]+ \| ev_nz=[0-9.]+ \| kl=[0-9.eE+-]+' "$LOG" 2>/dev/null | tail -1 || true)
  if [[ -z "$line" ]]; then
    sleep 15
    continue
  fi
  step=$(echo "$line" | sed -n 's/.*step=\([0-9]*\).*/\1/p')
  sr=$(echo "$line" | sed -n 's/.*search_rate=\([0-9.]*\).*/\1/p')
  kl=$(echo "$line" | sed -n 's/.*kl=\([0-9.eE+-]*\).*/\1/p')
  last_step=$step
  echo "[watch] step=$step search_rate=$sr kl=$kl streak=$streak"

  if awk -v sr="$sr" -v lo="$SEARCH_LO" 'BEGIN{exit !(sr+0 < lo+0)}'; then
    streak=$((streak + 1))
  else
    streak=0
  fi

  if [[ "$streak" -ge "$COLLAPSE_STREAK" ]]; then
    kill_train "search_collapse step=$step sr=$sr"
    echo "search_collapse" >"${LOG}.stop_reason"
    exit 0
  fi

  if [[ "$step" -ge "$ALWAYS_CHECK_STEP" ]] && awk -v sr="$sr" -v hi="$SEARCH_HI" 'BEGIN{exit !(sr+0 > hi+0)}'; then
    # require recent mean also high: check last 5 lines all > hi
    n_hi=$(grep -oE 'search_rate=[0-9.]+' "$LOG" | tail -5 | awk -F= -v hi="$SEARCH_HI" '$2+0>hi+0{c++} END{print c+0}')
    if [[ "$n_hi" -ge 5 ]]; then
      kill_train "always_search step=$step sr=$sr"
      echo "always_search" >"${LOG}.stop_reason"
      exit 0
    fi
  fi

  if [[ "$step" -ge 15 ]] && awk -v kl="$kl" -v th="$KL_KILL" 'BEGIN{exit !(kl+0 >= th+0)}'; then
    kill_train "kl_spike step=$step kl=$kl"
    echo "kl_spike" >"${LOG}.stop_reason"
    exit 0
  fi

  if [[ "$step" -ge "$MAX_STEP" ]]; then
    kill_train "max_step=$MAX_STEP"
    echo "max_step" >"${LOG}.stop_reason"
    exit 0
  fi

  sleep 20
done
