#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

tag=${1:?usage: $0 <tag> <model_path>}
model=${2:?usage: $0 <tag> <model_path>}
require_file "$model/config.json"
require_file "$FROZEN_DEV"

tag_root="$PROJECT_ROOT/results/frozen_dev/$tag"
[[ ! -e "$tag_root/summary.json" ]] || {
  echo "ERROR frozen-dev result already exists for tag=$tag" >&2
  exit 1
}
mkdir -p "$tag_root/runs"
log="$PROJECT_ROOT/logs/dev_${tag}_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} \
  "$VEXACT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/run_agent_rollout_smoke.py" \
    --model-path "$model" \
    --eval-file "$FROZEN_DEV" \
    --max-samples 200 \
    --top-k 5 \
    --max-search-turns 2 \
    --temperature 0.0 \
    --max-new-tokens 512 \
    --output-dir "$tag_root/runs" \
    --run-tag "$tag" \
    2>&1 | tee "$log"

summary=$(find "$tag_root/runs" -type f -name summary.json -printf '%T@ %p\n' \
  | sort -n | tail -n 1 | cut -d' ' -f2-)
require_file "$summary"
cp "$summary" "$tag_root/summary.json"
printf '%s\n' "$summary" >"$tag_root/run_summary_path.txt"
sha256sum "$model/config.json" "$model/tokenizer.json" >"$tag_root/model_identity.sha256"
echo "FROZEN_DEV_PASS tag=$tag summary=$tag_root/summary.json"
