#!/usr/bin/env bash
# Evaluate the pre-registered GRPO-without-local-std step-10 fallback.
set -euo pipefail

repo=/data1/hcc/deepresearch
vexact_repo=/data1/hcc/eca-verl-vexact
step=${1:?usage: $0 10|25|50}
[[ "$step" == 10 || "$step" == 25 || "$step" == 50 ]]
model=$repo/results/20_grpo_no_std/checkpoints/step${step}_hf
out=$repo/results/20_grpo_no_std/eval/step${step}
capture=$out/route_capture
sentinel=$repo/outputs/grpo_no_std_sentinel_step${step}
mkdir -p "$out" "$sentinel"
test -f "$model/config.json"

common_env=(env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$repo
  INFER_FA_IMPL=triton-invariant VEOMNI_ATTN_IMPLEMENTATION=triton-invariant
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
  "$repo/scripts/summarize_boundary_node.py" --profile grpo_no_std \
  --step "$step" --metrics "$repo/results/20_grpo_no_std/train_metrics.jsonl" \
  --route-summary "$out/route_margin_summary.json" \
  --full-log "$out/alignment_full.log" --fused-log "$out/alignment_fused_lce.log" \
  --output "$out/node_summary.json"
echo "GRPO_NO_STD_NODE_${step}_EVAL_COMPLETE"

if [[ "$step" == 25 || "$step" == 50 ]]; then
  a4=$out/frozen32x4
  ECA_VISIBLE_GPUS=0,1,2,3 bash "$repo/scripts/run_vexact_a4_parity_32x4.sh" \
    --config configs/rl/grpo_smoke128.yaml --seed 42 \
    --output-dir "${a4#$repo/}" --max-samples 32 --model-path "$model"

  env -u LD_LIBRARY_PATH "$vexact_repo/.venv/bin/python" - \
    "$out/node_summary.json" "$a4/a4_parity_summary.json" "$out/overnight_gate.json" "$step" <<'PY'
import json,sys
from pathlib import Path
node=json.loads(Path(sys.argv[1]).read_text())
routing=json.loads(Path(sys.argv[2]).read_text())
step=int(sys.argv[4])
m=node["mechanism"]["route_margin"]
passed=(
    node["alignment"]["pass"]
    and node["trajectory_gate_pass"]
    and node["optimizer_health_pass"]
    and routing["finish_rate"] >= .95
    and routing["clip_ratio"] < .05
    and routing["mixed_action_group_rate"] > 0
    and routing["search_rate_NeedSearch"] >= .85
    and routing["search_rate_NoSearch"] <= .70
    and routing["delta_boundary"] >= .20
    and m.get("NoSearch", 1e9) < .864
    and m.get("NeedSearch", -1e9) >= 1.272
)
out={"step":step,"gate":f"GRPO_NO_STD_STEP{step}_{'PASS' if passed else 'FAIL'}",
     "route_margin":m,"frozen32x4":routing,"system":{
       "alignment":node["alignment"],"trajectory_gate_pass":node["trajectory_gate_pass"],
       "optimizer_health_pass":node["optimizer_health_pass"]}}
Path(sys.argv[3]).write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
PY
fi
