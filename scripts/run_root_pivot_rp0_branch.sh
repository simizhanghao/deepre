#!/usr/bin/env bash
# One Evidence@400 Root-Pivot counterfactual; evaluate, summarize, then delete model state.
set -euo pipefail

repo=/data1/hcc/deepresearch
vexact_repo=/data1/hcc/eca-verl-vexact
mode=
subset=all
beta=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode=$2; shift 2 ;;
    --subset) subset=$2; shift 2 ;;
    --beta) beta=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$mode" == task_only || "$mode" == route_only || "$mode" == joint ]]
[[ "$subset" == all || "$subset" == need || "$subset" == no ]]

root=$repo/results/21_root_pivot/rp0/${subset}_${mode}
state=$repo/outputs/rl/rp0_${subset}_${mode}_scratch
data=$repo/results/21_root_pivot/data/${subset}_16.parquet
manifest=$repo/results/21_root_pivot/data/${subset}_manifest.json
model=$repo/outputs/rl/03_hf_evidence_step400
boundary=$repo/outputs/rl/04_table_search_boundary/boundary_latest.json
mkdir -p "$root" "$state"

env -u LD_LIBRARY_PATH PYTHONPATH=$repo "$vexact_repo/.venv/bin/python" \
  "$repo/scripts/build_root_pivot_balanced.py" \
  --source "$repo/results/18_boundary_exact_rollout/data/train.parquet" \
  --boundary "$boundary" --output "$data" --manifest "$manifest" \
  --per-class 8 --subset "$subset"
batch_size=$(env -u LD_LIBRARY_PATH "$vexact_repo/.venv/bin/python" -c \
  'import pandas as p,sys;print(len(p.read_parquet(sys.argv[1])))' "$data")

rm -f "$root/train_metrics.jsonl" "$root/trajectories.jsonl"
rm -rf -- "$state"
mkdir -p "$state"

export PYTHONPATH=$repo
export ECA_REPO_ROOT=$repo
export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export ECA_ROLLOUT_BACKEND=vexact
export ECA_ROOT_PIVOT=1
export ECA_ROOT_PIVOT_MODE=$mode
export ECA_ROOT_PIVOT_BETA=$beta
export ECA_SCHEDULE_HORIZON=10
export ECA_TRAIN_METRICS_JSONL=$root/train_metrics.jsonl
export ECA_PARITY_DUMP=$root/trajectories.jsonl
export ECA_BOUNDARY_TABLE=$boundary
export ECA_BOUNDARY_STRICT=1
export ECA_EVIDENCE_WEIGHT=0.5
export ECA_SEARCH_COST_WEIGHT=0.30
export ECA_AUDIT_STOP_MODE=sequence
export ECA_MAX_ASSISTANT_TURN_TOKENS=256
export ECA_FINAL_ANSWER_RESERVE=256
export INFER_FA_IMPL=triton-invariant
export VEOMNI_ATTN_IMPLEMENTATION=triton-invariant
export MODELING_BACKEND=veomni
export VEOMNI_USE_LIGER_KERNEL=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "$vexact_repo/.venv/bin/python" "$repo/scripts/launch_grpo_main.py" \
  model_engine=veomni algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False algorithm.norm_adv_by_std_in_grpo=False \
  data.train_files="$data" data.val_files="$data" \
  data.train_batch_size="$batch_size" \
  data.max_prompt_length=1024 data.max_response_length=2048 \
  data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True data.seed=42 \
  actor_rollout_ref.model.path="$model" \
  actor_rollout_ref.model.external_lib=src.rl.root_pivot \
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=triton-invariant \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size="$batch_size" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.actor.veomni.fsdp_size=4 \
  actor_rollout_ref.actor.veomni.attn_implementation=triton-invariant \
  actor_rollout_ref.actor.veomni.cross_entropy_loss_implementation=eager \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vexact actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.seed=42 actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=0.9 actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks=512 \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=triton-invariant \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=384 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$repo/configs/rl/candidate_bm25_tool.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=eca_search_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$repo/configs/rl/eca_agent_loop.yaml" \
  reward.custom_reward_function.path="$repo/src/rl/rewards_boundary.py" \
  reward.custom_reward_function.name=compute_score trainer.nnodes=1 trainer.n_gpus_per_node=4 \
  trainer.total_epochs=1 trainer.total_training_steps=1 trainer.val_before_train=False \
  trainer.save_freq=1 trainer.test_freq=-1 trainer.resume_mode=disable trainer.max_actor_ckpt_to_keep=1 \
  'trainer.logger=[console]' trainer.project_name=eca_root_pivot \
  trainer.experiment_name="rp0_${subset}_${mode}" trainer.default_local_dir="$state" \
  2>&1 | tee "$root/run.log"

hf=$state/global_step_1/actor/huggingface
test -f "$hf/config.json"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$repo \
  INFER_FA_IMPL=triton-invariant VEOMNI_ATTN_IMPLEMENTATION=triton-invariant \
  MODELING_BACKEND=veomni VEOMNI_USE_LIGER_KERNEL=0 \
  "$vexact_repo/.venv/bin/python" "$repo/scripts/capture_vexact_exact2.py" \
  --config "$repo/configs/rl/grpo_smoke128.yaml" --seed 42 \
  --output-dir "$root/route_capture" --max-samples 20 --n-rollouts 0 --route-probe-only \
  --model-path "$hf" \
  --sample-manifest "$repo/results/16_audit_routing_exploration/worker_mismatch/sample_ids.json" \
  --attn-impl triton-invariant
env -u LD_LIBRARY_PATH "$vexact_repo/.venv/bin/python" \
  "$repo/scripts/summarize_boundary_route_margin.py" \
  --capture-dir "$root/route_capture" --output "$root/route_margin_summary.json" --step 1

env -u LD_LIBRARY_PATH "$vexact_repo/.venv/bin/python" - \
  "$root/train_metrics.jsonl" "$root/route_margin_summary.json" "$root/branch_summary.json" \
  "$mode" "$subset" "$beta" <<'PY'
import json,sys
from pathlib import Path
metric=json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])
route=json.loads(Path(sys.argv[2]).read_text())
out={"gate":"RP0_BRANCH_COMPLETE","mode":sys.argv[4],"subset":sys.argv[5],"beta":float(sys.argv[6]),
     "gradient_norm":metric["actor/grad_norm"],"route_margin":route["mean_route_margin"],
     "root_pivot_metrics":{k:v for k,v in metric.items() if k.startswith("actor/root_pivot/")},
     "trajectory":{"finish_rate":metric.get("agent/finish_rate"),
       "clip_ratio":metric.get("response_length/clip_ratio")}}
Path(sys.argv[3]).write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
PY

# RP-0 is attribution, not a resumable candidate. Keep only measurements.
rm -rf -- "$state"
echo "RP0_BRANCH_DONE mode=$mode subset=$subset"
