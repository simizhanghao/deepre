#!/usr/bin/env bash
# Evaluate one completed Boundary@50 segment after its training Ray job exits.
set -euo pipefail

repo=/data1/hcc/deepresearch
vexact_repo=/data1/hcc/eca-verl-vexact
step=${1:?usage: $0 10|25|50}
[[ "$step" == 10 || "$step" == 25 || "$step" == 50 ]]
model=$repo/results/18_boundary_exact_rollout/checkpoints/step${step}_hf
out=$repo/results/18_boundary_exact_rollout/eval/step${step}
capture=$out/route_capture
sentinel=$repo/outputs/boundary_exact_sentinel_step${step}
mkdir -p "$out" "$sentinel"
test -f "$model/config.json"

common_env=(env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$repo \
  INFER_FA_IMPL=triton-invariant VEOMNI_ATTN_IMPLEMENTATION=triton-invariant \
  MODELING_BACKEND=veomni VEOMNI_USE_LIGER_KERNEL=0 TOKENIZERS_PARALLELISM=false)

"${common_env[@]}" "$vexact_repo/.venv/bin/python" "$repo/scripts/capture_vexact_exact2.py" \
  --config "$repo/configs/rl/grpo_smoke128.yaml" --seed 42 \
  --output-dir "$capture" --max-samples 20 --n-rollouts 0 --route-probe-only \
  --model-path "$model" \
  --sample-manifest "$repo/results/16_audit_routing_exploration/worker_mismatch/sample_ids.json" \
  --attn-impl triton-invariant
env -u LD_LIBRARY_PATH "$vexact_repo/.venv/bin/python" \
  "$repo/scripts/summarize_boundary_route_margin.py" \
  --capture-dir "$capture" --output "$out/route_margin_summary.json" --step "$step"

"${common_env[@]}" "$vexact_repo/.venv/bin/python" "$repo/scripts/capture_vexact_exact2.py" \
  --config "$repo/configs/rl/grpo_smoke128.yaml" --seed 42 \
  --output-dir "$sentinel/capture" --max-samples 2 --debug \
  --model-path "$model" --attn-impl triton-invariant

verify=$vexact_repo/tests/scripts/verify_logits_vs_native_hf.py
"${common_env[@]}" "$vexact_repo/.venv/bin/python" "$verify" \
  --model_path "$model" --data_dir "$sentinel/capture" --rtol 1e-5 --atol 1e-6 \
  --log_file "$out/alignment_full.log" --attn_impl triton-invariant \
  --model_backend veomni --enable_batch_invariant --use_remove_padding \
  --logprobs_from_logits --skip_backward 2>&1 | tee "$out/alignment_full.log"
"${common_env[@]}" VEXACT_TESTS_MOE_IMPL=eager "$vexact_repo/.venv/bin/python" "$verify" \
  --model_path "$model" --data_dir "$sentinel/capture" --rtol 1e-5 --atol 1e-6 \
  --log_file "$out/alignment_fused_lce.log" --attn_impl triton-invariant \
  --model_backend veomni --enable_batch_invariant --use_remove_padding \
  --use_fused_lce --skip_backward 2>&1 | tee "$out/alignment_fused_lce.log"

env -u LD_LIBRARY_PATH "$vexact_repo/.venv/bin/python" \
  "$repo/scripts/summarize_boundary_node.py" \
  --step "$step" \
  --metrics "$repo/results/18_boundary_exact_rollout/train_metrics.jsonl" \
  --route-summary "$out/route_margin_summary.json" \
  --full-log "$out/alignment_full.log" --fused-log "$out/alignment_fused_lce.log" \
  --output "$out/node_summary.json"
echo "BOUNDARY_NODE_${step}_EVAL_COMPLETE"
