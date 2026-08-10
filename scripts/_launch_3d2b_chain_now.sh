#!/usr/bin/env bash
set -euo pipefail
REPO=/data1/hcc/deepresearch
cd "$REPO"
mkdir -p logs
PY_PID=$(ps -eo pid=,args= | awk '
  $2 ~ /(^|\/)python([0-9.]+)?$/ && /scripts\/build_search_boundary_table\.py/ {
    print $1; exit
  }')
if [[ -z "${PY_PID}" ]]; then
  echo "no bootstrap python; will proceed if boundary_latest exists"
fi
echo "BOOTSTRAP_PID=${PY_PID:-none}" | tee logs/phase3d2b_chain.pidinfo
# ensure only one real chain (must be bash running that script as argv0/1)
EXISTING=$(
  ps -eo pid=,args= | awk '
    $2 ~ /(^|\/)bash$/ && /run_phase3d2b_bootstrap_then_train\.sh/ { print $1; exit }
  '
)
if [[ -n "${EXISTING}" ]]; then
  echo "chain already running pid=$EXISTING"
  ps -p "$EXISTING" -o pid,etime,args
  echo "$EXISTING" >"$REPO/logs/phase3d2b_chain.pid"
  exit 0
fi
nohup env BOOTSTRAP_PID="${PY_PID}" STEPS=50 \
  bash "$REPO/scripts/run_phase3d2b_bootstrap_then_train.sh" \
  >"$REPO/logs/phase3d2b_chain.log" 2>&1 &
CHAIN_PID=$!
echo "$CHAIN_PID" | tee "$REPO/logs/phase3d2b_chain.pid"
sleep 3
echo "=== chain log ==="
head -30 "$REPO/logs/phase3d2b_chain.log" || true
if ! ps -p "$CHAIN_PID" >/dev/null 2>&1; then
  echo "ERROR: chain died"
  cat "$REPO/logs/phase3d2b_chain.log" || true
  exit 1
fi
ps -p "$CHAIN_PID" -o pid,etime,args
