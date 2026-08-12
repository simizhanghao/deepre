#!/usr/bin/env bash
# Snapshot-safe CUR-1 feature extraction and frozen B0--B6 Validation gate.
set -euo pipefail

repo=/data1/hcc/deepresearch
run_id=${CUR1_OFFLINE_RUN_ID:-b0_b6_v1}
root=$repo/results/23_cur1/offline/$run_id
snapshot=$root/snapshot

if [[ "${CUR1_FROZEN_LAUNCH:-0}" != 1 ]]; then
  if [[ -e "$root" && "${CUR1_ALLOW_OVERWRITE:-0}" != 1 ]]; then
    echo "REFUSE_OVERWRITE=$root" >&2
    exit 5
  fi
  mkdir -p "$snapshot"
  cp "$repo/scripts/run_cur1_offline_b0_b6.sh" "$snapshot/run_frozen.sh"
  cp "$repo/scripts/extract_cur1_features.py" "$snapshot/extract_cur1_features.py"
  cp "$repo/scripts/fit_evaluate_cur1_b0_b6.py" "$snapshot/fit_evaluate_cur1_b0_b6.py"
  cp "$repo/docs/CUR1_PLAN.md" "$snapshot/CUR1_PLAN.md"
  cp "$repo/data/cur/cur1_fresh896/manifest.json" "$snapshot/split_manifest.json"
  git -C "$repo" rev-parse HEAD > "$snapshot/git_sha.txt"
  sha256sum "$snapshot"/* > "$snapshot/snapshot_sha256.txt"
  exec env CUR1_FROZEN_LAUNCH=1 CUR1_OFFLINE_RUN_ID="$run_id" \
    CUR_FEATURE_GPU="${CUR_FEATURE_GPU:-4}" bash "$snapshot/run_frozen.sh"
fi

test ! -e "$repo/results/23_cur1/capture/test/outcomes.jsonl" || {
  echo "TEST_OUTCOME_EXISTS_BEFORE_UNLOCK" >&2
  exit 9
}
mkdir -p "$root/features" "$root/models"
python=/data1/hcc/eca-verl-vexact/.venv/bin/python
model=$repo/outputs/rl/03_hf_evidence_step400
manifest=$repo/data/cur/cur1_fresh896/prompt_manifest.jsonl

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$CUR_FEATURE_GPU" "$python" \
  "$snapshot/extract_cur1_features.py" \
  --manifest "$manifest" --ids "$repo/data/cur/cur1_fresh896/train_ids.txt" \
  --split train --model "$model" --output "$root/features/train_features.npz" \
  --batch-size 32 2>&1 | tee "$root/features/train_extract.log"

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$CUR_FEATURE_GPU" "$python" \
  "$snapshot/extract_cur1_features.py" \
  --manifest "$manifest" --ids "$repo/data/cur/cur1_fresh896/validation_ids.txt" \
  --split validation --model "$model" --output "$root/features/validation_features.npz" \
  --batch-size 32 2>&1 | tee "$root/features/validation_extract.log"

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$CUR_FEATURE_GPU" "$python" \
  "$snapshot/fit_evaluate_cur1_b0_b6.py" \
  --train-features "$root/features/train_features.npz" \
  --validation-features "$root/features/validation_features.npz" \
  --train-outcomes "$repo/results/23_cur1/capture/train/outcomes.jsonl" \
  --validation-outcomes "$repo/results/23_cur1/capture/validation/outcomes.jsonl" \
  --train-split "$repo/data/cur/cur1_fresh896/train.parquet" \
  --validation-split "$repo/data/cur/cur1_fresh896/validation.parquet" \
  --output-dir "$root/models" 2>&1 | tee "$root/fit_evaluate.log"

test -s "$root/models/validation_decision.json"
test -s "$root/models/candidate_bundle_manifest.json"
echo "CUR1_OFFLINE_B0_B6_COMPLETE=$root"
