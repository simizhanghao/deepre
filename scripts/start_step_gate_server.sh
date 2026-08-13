#!/usr/bin/env bash
set -euo pipefail
repo=/data1/hcc/deepresearch
vexact=/data1/hcc/eca-verl-vexact
cd "$repo"
extra=()
root_scores=${STEP_GATE_ROOT_SCORES:-$repo/results/25_step_adaptive/val3/root_static/root_scores.json}
if [[ -s "$root_scores" ]]; then
  extra=(--root-score-map "$root_scores")
fi
exec env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="${STEP_GATE_GPU:-4}" \
  "$vexact/.venv/bin/python" scripts/serve_step_preference_gate.py \
  --model outputs/rl/03_hf_evidence_step400 \
  --gate-dir results/25_step_adaptive/step_gate/models \
  --root-models results/23_cur1/offline/b0_b6_v1/models/b3 \
  --port 8007 --max-batch 16 --batch-wait-ms 10 "${extra[@]}"
