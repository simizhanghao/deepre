#!/usr/bin/env bash
# One-shot Phase25 S1: base states -> frozen pairs -> causal/Oracle decision.
set -euo pipefail

repo=/data1/hcc/deepresearch
py=/data1/hcc/eca-verl-vexact/.venv/bin/python
root=$repo/results/25_step_adaptive/s1
mkdir -p "$root"

cd "$repo"
env -u LD_LIBRARY_PATH "$py" scripts/build_step_s1_base.py | tee "$root/base_freeze.log"
bash scripts/run_step_s1_capture.sh base | tee "$root/base_launcher.log"
env -u LD_LIBRARY_PATH "$py" scripts/audit_step_s0.py \
  "$root/base/step_records.jsonl" --expected 640 --output "$root/base/summary.json" \
  | tee "$root/base_audit.log"

env -u LD_LIBRARY_PATH "$py" scripts/build_step_s1_branches.py | tee "$root/branch_freeze.log"
bash scripts/run_step_s1_capture.sh branches | tee "$root/branches_launcher.log"
env -u LD_LIBRARY_PATH "$py" scripts/analyze_step_s1.py \
  --records "$root/branches/step_records.jsonl" \
  --base-records "$root/base/step_records.jsonl" \
  --selections data/step_adaptive/s1_train640/checkpoint_selections.jsonl \
  --data data/step_adaptive/s1_train640/base640.parquet \
  --model outputs/rl/03_hf_evidence_step400 \
  --pairs-output "$root/analysis/pairs.jsonl" \
  --summary-output "$root/analysis/summary.json" \
  | tee "$root/analysis.log"

echo STEP_S1_PIPELINE_COMPLETE
