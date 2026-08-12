#!/usr/bin/env bash
# Frozen Evidence@400 CUR-1 paired potential-outcome acquisition.
set -euo pipefail

repo=/data1/hcc/deepresearch
vexact_repo=/data1/hcc/eca-verl-vexact
stage=${1:-}
[[ "$stage" == smoke || "$stage" == train || "$stage" == validation || "$stage" == test ]] || {
  echo "usage: $0 smoke|train|validation|test" >&2; exit 2;
}
gpus=${CUR_GPUS:-0,1,2,3}
data_dir=$repo/data/cur/cur1_fresh896
data=$data_dir/smoke_2.parquet
questions=2
rollouts=1
batch=4
steps=1
micro_batch=1
max_batched_tokens=8192
max_cache_blocks=512
case "$stage" in
  train)
    data=$data_dir/train.parquet; questions=640; rollouts=1; batch=64; steps=20
    micro_batch=4; max_batched_tokens=16384; max_cache_blocks=1024 ;;
  validation)
    data=$data_dir/validation.parquet; questions=128; rollouts=4; batch=32; steps=8
    micro_batch=4; max_batched_tokens=16384; max_cache_blocks=1024 ;;
  test)
    data=$data_dir/test.parquet; questions=128; rollouts=8; batch=16; steps=16
    lock=$repo/results/23_cur1/locks/final_candidate.lock.json
    test -s "$lock" || {
      echo "SEALED_TEST_LOCKED: missing final candidate lock $lock" >&2; exit 4;
    } ;;
esac

root=$repo/results/23_cur1/capture/$stage
if [[ -e "$root/outcomes.jsonl" && "${CUR1_ALLOW_OVERWRITE:-0}" != 1 ]]; then
  echo "REFUSE_OVERWRITE_CAPTURE=$root/outcomes.jsonl" >&2; exit 5
fi
mkdir -p "$root"
rm -f -- "$root/outcomes.jsonl" "$root/trajectories.jsonl" "$root/summary.json"

env -u LD_LIBRARY_PATH "$vexact_repo/.venv/bin/python" "$repo/scripts/audit_cur1_freeze.py" \
  --data-dir "$data_dir" --model "$repo/outputs/rl/03_hf_evidence_step400" \
  --output "$root/preflight_freeze_audit.json"
curl -fsS http://127.0.0.1:8003/health >/dev/null || {
  echo "CUR-1 Candidate-BM25 server is not healthy on :8003" >&2; exit 3;
}
# A health-only check cannot detect a missing BM25 runtime dependency. Exercise
# one real retrieval before allocating GPUs so infrastructure failures can
# never be recorded as policy outcomes.
/home/hanchengcheng/miniconda3/envs/deepresearch/bin/python - "$data_dir/contexts_index.jsonl" <<'PY'
import json, sys, urllib.request

with open(sys.argv[1], encoding="utf-8") as handle:
    sample_id = json.loads(next(handle))["sample_id"]
payload = json.dumps({"sample_id": sample_id, "query": "evidence", "topk": 1}).encode()
request = urllib.request.Request(
    "http://127.0.0.1:8003/retrieve",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=15) as response:
    result = json.load(response)
assert result.get("documents"), result
print("CUR1_RETRIEVER_CANARY_PASS")
PY

export PYTHONPATH=$repo${PYTHONPATH:+:$PYTHONPATH}
export ECA_REPO_ROOT=$repo
export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export ECA_ROLLOUT_BACKEND=vexact ECA_FORWARD_ONLY_CAPTURE=1
export ECA_CUR_CAPTURE_JSONL=$root/outcomes.jsonl
export ECA_PARITY_DUMP=$root/trajectories.jsonl
export ECA_AUDIT_STOP_MODE=sequence ECA_MAX_ASSISTANT_TURN_TOKENS=256 ECA_FINAL_ANSWER_RESERVE=256
export INFER_FA_IMPL=triton-invariant VEOMNI_ATTN_IMPLEMENTATION=triton-invariant MODELING_BACKEND=veomni
export VEOMNI_USE_LIGER_KERNEL=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset ECA_ATTRIBUTION_CAPTURE_DIR ECA_BOUNDARY_TABLE ECA_ROOT_PIVOT

set +e
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpus" \
  "$vexact_repo/.venv/bin/python" "$repo/scripts/launch_grpo_main.py" \
  model_engine=veomni algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files="$data" data.val_files="$data" data.train_batch_size="$batch" \
  data.max_prompt_length=1024 data.max_response_length=2048 data.filter_overlong_prompts=True \
  data.truncation=error data.return_raw_chat=True data.seed=2026081201 \
  actor_rollout_ref.model.path="$repo/outputs/rl/03_hf_evidence_step400" \
  actor_rollout_ref.model.external_lib=vexact.integrations.verl.fsdp_enable_invariant \
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch actor_rollout_ref.model.enable_gradient_checkpointing=False \
  +actor_rollout_ref.model.override_config.attn_implementation=triton-invariant \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size="$batch" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$micro_batch" actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.veomni.fsdp_size=4 actor_rollout_ref.actor.veomni.attn_implementation=triton-invariant \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$micro_batch" \
  actor_rollout_ref.rollout.name=vexact actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.seed=2026081201 actor_rollout_ref.rollout.n="$rollouts" \
  actor_rollout_ref.rollout.temperature=0.9 actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.max_num_batched_tokens="$max_batched_tokens" actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.calculate_log_probs=True actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$micro_batch" \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks="$max_cache_blocks" \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=triton-invariant \
  actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=4 actor_rollout_ref.rollout.multi_turn.max_tool_response_length=384 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$repo/configs/rl/cur1_candidate_bm25_tool.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=eca_search_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$repo/configs/rl/eca_agent_loop.yaml" \
  reward.custom_reward_function.path="$repo/src/rl/rewards_cur.py" reward.custom_reward_function.name=compute_score \
  trainer.nnodes=1 trainer.n_gpus_per_node=4 trainer.total_epochs=100 trainer.total_training_steps="$steps" \
  trainer.val_before_train=False trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
  'trainer.logger=[console]' trainer.project_name=eca_cur1 trainer.experiment_name="cur1_${stage}" \
  trainer.default_local_dir="$root/no_checkpoint" 2>&1 | tee "$root/run.log"
run_rc=${PIPESTATUS[0]}
set -e
echo "cur1_${stage}_run_rc=$run_rc"
[[ "$run_rc" -eq 0 ]]
grep -q "FORWARD_ONLY: actor backward/optimizer/scheduler are disabled" "$root/run.log"
env -u LD_LIBRARY_PATH "$vexact_repo/.venv/bin/python" "$repo/scripts/audit_cur0_capture.py" \
  --input "$root/outcomes.jsonl" --questions "$questions" --rollouts "$rollouts" \
  --output "$root/summary.json" | tee "$root/audit.log"
echo "CUR1_${stage^^}_CAPTURE_PASS=$root"
