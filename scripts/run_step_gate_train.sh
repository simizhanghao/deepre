#!/usr/bin/env bash
set -euo pipefail

repo=/data1/hcc/deepresearch
vexact=/data1/hcc/eca-verl-vexact
out="$repo/results/25_step_adaptive/step_gate"
mkdir -p "$out/features" "$out/models"
cd "$repo"

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

env -u LD_LIBRARY_PATH "$vexact/.venv/bin/torchrun" \
  --standalone --nproc-per-node=4 \
  scripts/extract_step_gate_features.py \
  --model outputs/rl/03_hf_evidence_step400 \
  --parquet data/step_adaptive/s1_train640/base640.parquet \
  --base-dump results/25_step_adaptive/s1/base/step_records.jsonl \
  --selections data/step_adaptive/s1_train640/checkpoint_selections.jsonl \
  --pairs results/25_step_adaptive/s1/analysis/pairs.jsonl \
  --output "$out/features/train_features.npz" \
  --batch-size 16 \
  2>&1 | tee "$out/feature_extract.log"

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$vexact/.venv/bin/python" \
  scripts/fit_step_preference_gate.py \
  --features "$out/features/train_features.npz" \
  --root-oof results/24_dssr/models/train_oof_predictions.npz \
  --root-static-features results/23_cur1/offline/b0_b6_v1/features/train_features.npz \
  --root-model-dir results/23_cur1/offline/b0_b6_v1/models/b3 \
  --output-dir "$out/models" \
  2>&1 | tee "$out/gate_fit.log"

echo STEP_GATE_TRAIN_PIPELINE_PASS
