#!/usr/bin/env bash
# Smoke-gated train+validation acquisition. Deliberately never opens test.
set -euo pipefail
repo=/data1/hcc/deepresearch
export CUR_GPUS=${CUR_GPUS:-0,1,2,3}

bash "$repo/scripts/run_cur1_capture.sh" smoke
gate=$(/data1/hcc/eca-verl-vexact/.venv/bin/python -c \
  'import json; print(json.load(open("/data1/hcc/deepresearch/results/23_cur1/capture/smoke/summary.json"))["gate"])')
[[ "$gate" == CUR_CAPTURE_PASS ]] || { echo "TRAINVAL_LOCKED: smoke=$gate"; exit 1; }
bash "$repo/scripts/run_cur1_capture.sh" train
bash "$repo/scripts/run_cur1_capture.sh" validation
echo "CUR1_TRAINVAL_ACQUISITION_COMPLETE_TEST_REMAINS_SEALED"
