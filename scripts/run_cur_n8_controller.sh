#!/usr/bin/env bash
set -euo pipefail
repo=/data1/hcc/deepresearch
py=/data1/hcc/eca-verl-vexact/.venv/bin/python

CUR_GPUS=${CUR_GPUS:-0,1,2,3} bash "$repo/scripts/run_cur0_forced_capture.sh" n8
env -u LD_LIBRARY_PATH "$py" "$repo/scripts/repair_cur0_outcomes.py" \
  --input "$repo/results/22_cur/cur0_capture/n8/outcomes.jsonl" \
  --output "$repo/results/22_cur/cur0_capture/n8/outcomes_repaired.jsonl" \
  --summary "$repo/results/22_cur/cur0_capture/n8/outcome_repair_summary.json"
env -u LD_LIBRARY_PATH "$py" "$repo/scripts/merge_cur_outcomes.py" \
  --base "$repo/results/22_cur/cur0_capture/full/outcomes_repaired.jsonl" \
  --supplement "$repo/results/22_cur/cur0_capture/n8/outcomes_repaired.jsonl" \
  --output "$repo/results/22_cur/cur0_capture/merged_n8/outcomes.jsonl"
env -u LD_LIBRARY_PATH "$py" "$repo/scripts/analyze_cur0_outcomes.py" \
  --input "$repo/results/22_cur/cur0_capture/merged_n8/outcomes.jsonl" \
  --output-dir "$repo/results/22_cur/gate0a_n8"
env -u LD_LIBRARY_PATH "$py" "$repo/scripts/analyze_cur0_margin.py" \
  --outcomes "$repo/results/22_cur/gate0a_n8/question_outcomes.jsonl" \
  --capture-dir "$repo/results/22_cur/gate0b/root_capture" \
  --output "$repo/results/22_cur/gate0b/gate0b_summary_n8.json"
env -u LD_LIBRARY_PATH "$py" "$repo/scripts/probe_cur_hidden_states.py" \
  --hidden "$repo/results/22_cur/gate0c/hidden_states.npz" \
  --outcomes "$repo/results/22_cur/gate0a_n8/question_outcomes.jsonl" \
  --output "$repo/results/22_cur/gate0c/gate0c_summary_n8.json"
echo CUR_N8_PIPELINE_COMPLETE
