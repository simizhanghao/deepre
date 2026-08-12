#!/usr/bin/env bash
# Wait for an already-running CUR smoke and unlock full capture only on PASS.
set -euo pipefail
repo=/data1/hcc/deepresearch
while tmux has-session -t cur0_smoke 2>/dev/null; do sleep 10; done
summary=$repo/results/22_cur/cur0_capture/smoke/summary.json
test -s "$summary"
gate=$(/data1/hcc/eca-verl-vexact/.venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["gate"])' "$summary")
[[ "$gate" == CUR_CAPTURE_PASS ]] || {
  echo "FULL_LOCKED: smoke gate=$gate"; exit 1;
}
echo "SMOKE_PASS: launching detached full CUR-0 capture"
CUR_GPUS=${CUR_GPUS:-0,1,2,3} bash "$repo/scripts/run_cur0_forced_capture.sh" full
