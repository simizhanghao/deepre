#!/usr/bin/env bash
# Execute the frozen RP-0 causal matrix. Stop before formal @10 unless the hard gate passes.
set -euo pipefail

repo=/data1/hcc/deepresearch
cd "$repo"
branch="$repo/scripts/run_root_pivot_rp0_branch.sh"
root="$repo/results/21_root_pivot/rp0"
mkdir -p "$root"

# A successfully completed branch is immutable and reusable after interruption.
run_branch() {
  local subset=$1 mode=$2 beta=$3
  local summary="$root/${subset}_${mode}/branch_summary.json"
  if [[ -s "$summary" ]]; then
    echo "RP0_REUSE subset=$subset mode=$mode"
  else
    "$branch" --subset "$subset" --mode "$mode" --beta "$beta"
  fi
}

run_branch all task_only 1
run_branch all route_only 1
beta=$(env -u LD_LIBRARY_PATH PYTHONPATH="$repo" \
  /data1/hcc/eca-verl-vexact/.venv/bin/python \
  "$repo/scripts/analyze_root_pivot_rp0.py" --root "$root" --beta-only)
printf '%s\n' "$beta" > "$root/beta.txt"
echo "RP0_FIXED_BETA=$beta"
run_branch all joint "$beta"

for subset in need no; do
  run_branch "$subset" task_only 1
  run_branch "$subset" route_only 1
  run_branch "$subset" joint "$beta"
done

env -u LD_LIBRARY_PATH PYTHONPATH="$repo" \
  /data1/hcc/eca-verl-vexact/.venv/bin/python \
  "$repo/scripts/analyze_root_pivot_rp0.py" --root "$root"
