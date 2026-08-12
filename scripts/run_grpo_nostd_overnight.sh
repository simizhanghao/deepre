#!/usr/bin/env bash
# Locked segmented controller: current @10 -> gated @25 -> gated @50 -> dev-200.
set -euo pipefail

repo=/data1/hcc/deepresearch
root=$repo/results/20_grpo_no_std
log=$root/logs/overnight_controller.log
mkdir -p "$root/logs"
exec > >(tee -a "$log") 2>&1

echo "OVERNIGHT_CONTROLLER_START=$(date -Is)"
echo "FREE_SPACE_BEFORE=$(df -B1 --output=avail "$repo" | tail -1)"

# The controller owns the complete training lifecycle. A prior partial run has
# no resumable checkpoint and is archived automatically by the runner.
if [[ ! -s "$root/eval/step10/node_summary.json" ]]; then
  bash "$repo/scripts/run_boundary_exact_rollout.sh" \
    --profile grpo_no_std --target-step 10 --n-gpus 4
fi

node10=$root/eval/step10/node_summary.json
test -s "$node10" || { echo "STOP_STEP10_NO_SUMMARY"; exit 10; }
py=/data1/hcc/eca-verl-vexact/.venv/bin/python
gate10=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$node10")
echo "STEP10_GATE=$gate10"
[[ "$gate10" == GRPO_NO_STD_DIRECTION_PASS ]] || { echo "STOP_STEP10_GATE_FAIL"; exit 0; }

avail=$(df -B1 --output=avail "$repo" | tail -1)
(( avail >= 150000000000 )) || { echo "STOP_LOW_DISK_BEFORE_STEP25=$avail"; exit 25; }
bash "$repo/scripts/run_boundary_exact_rollout.sh" --profile grpo_no_std --target-step 25 --n-gpus 4
gate25_file=$root/eval/step25/overnight_gate.json
test -s "$gate25_file" || { echo "STOP_STEP25_NO_GATE"; exit 25; }
gate25=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate"])' "$gate25_file")
echo "STEP25_GATE=$gate25"
[[ "$gate25" == GRPO_NO_STD_STEP25_PASS ]] || { echo "STOP_STEP25_GATE_FAIL"; exit 0; }

avail=$(df -B1 --output=avail "$repo" | tail -1)
(( avail >= 150000000000 )) || { echo "STOP_LOW_DISK_BEFORE_STEP50=$avail"; exit 50; }
bash "$repo/scripts/run_boundary_exact_rollout.sh" --profile grpo_no_std --target-step 50 --n-gpus 4
gate50_file=$root/eval/step50/overnight_gate.json
test -s "$gate50_file" || { echo "STOP_STEP50_NO_GATE"; exit 50; }
gate50=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate"])' "$gate50_file")
echo "STEP50_GATE=$gate50"

# Step 50 is the terminal training node regardless of its gate. Produce the
# requested development-only Agent evaluation; never continue to step 100.
model=$root/checkpoints/step50_hf
devout=$root/eval/step50/dev200
mkdir -p "$devout"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=4 \
  /data1/hcc/eca-verl-vexact/.venv/bin/python "$repo/scripts/run_agent_rollout_smoke.py" \
  --model-path "$model" --eval-file "$repo/data/eval/hotpotqa_200.jsonl" \
  --max-samples 200 --top-k 5 --max-search-turns 2 --temperature 0 \
  --max-new-tokens 512 --output-dir "$devout" --run-tag grpo_no_std_step50_dev200
echo "OVERNIGHT_CONTROLLER_COMPLETE=$(date -Is)"
