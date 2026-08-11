#!/usr/bin/env bash
# A3/Gate B: real multi-turn EcaSearchAgentLoop -> VeXact, validation only.
set -euo pipefail

repo=/data1/hcc/deepresearch
vexact_repo=/data1/hcc/eca-verl-vexact
config=
seed=
output_dir=
max_samples=
debug=0
validate_only=0
stage=a3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --max-samples) max_samples=$2; shift 2 ;;
    --debug) debug=1; shift ;;
    --validate-only) validate_only=1; shift ;;
    --stage) stage=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$config" && -n "$seed" && -n "$output_dir" && -n "$max_samples" ]] || {
  echo "required: --config --seed --output-dir --max-samples --debug" >&2
  exit 2
}
if [[ "$stage" == a3 ]]; then
  [[ "$debug" -eq 1 && "$max_samples" -eq 8 ]] || {
    echo "A3 Gate B smoke is locked to --debug --max-samples 8" >&2
    exit 2
  }
  n_rollouts=2
  experiment_name=vexact_a3_gate_b
elif [[ "$stage" == a4 ]]; then
  [[ "$debug" -eq 0 && "$max_samples" -eq 32 ]] || {
    echo "A4 parity is locked to --max-samples 32 without --debug" >&2
    exit 2
  }
  n_rollouts=4
  experiment_name=vexact_a4_parity_32x4
else
  echo "unknown --stage: $stage" >&2
  exit 2
fi

config_abs=$repo/${config#./}
output_abs=$repo/${output_dir#./}
if [[ "$stage" == a3 ]]; then
  train_file=$output_abs/train_gate_b_smoke8.parquet
  dump_file=$output_abs/a3_multi_turn.jsonl
  parity_file=$output_abs/a3_trajectories.jsonl
  summary_file=$output_abs/gate_b_summary.json
else
  train_file=$repo/results/16_audit_routing_exploration/parity_sglang_32x4/train_parity_32.parquet
  dump_file=$output_abs/a4_multi_turn.jsonl
  parity_file=$output_abs/a4_trajectories.jsonl
  summary_file=$output_abs/a4_parity_summary.json
fi
run_log=$output_abs/run.log

test -f "$config_abs"
test -f "$train_file"
test -d "$repo/outputs/rl/03_hf_evidence_step400"
test -x "$vexact_repo/.venv/bin/python"
curl -sf http://127.0.0.1:8001/health >/dev/null || {
  echo "Candidate-BM25 server is not healthy on :8001" >&2
  exit 3
}
mkdir -p "$output_abs"
if [[ "$validate_only" -eq 0 ]]; then
  rm -f "$dump_file" "$parity_file" "$summary_file"
fi

export PYTHONPATH=$repo${PYTHONPATH:+:$PYTHONPATH}
export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export ECA_ROLLOUT_BACKEND=vexact
export ECA_ROUTING_MISMATCH_AUDIT=1
export ECA_AUDIT_FIRST_GENERATE_ONLY=0
export ECA_AUDIT_PATH=${stage^^}
export ECA_AUDIT_STOP_MODE=sequence
export ECA_MAX_ASSISTANT_TURN_TOKENS=256
export ECA_FINAL_ANSWER_RESERVE=256
export ECA_ROUTING_MISMATCH_DUMP=$dump_file
export ECA_PARITY_DUMP=$parity_file
export ECA_BOUNDARY_TABLE=$repo/outputs/rl/04_table_search_boundary/boundary_latest.json
export ECA_BOUNDARY_STRICT=1
export ECA_EVIDENCE_WEIGHT=0.5
export ECA_SEARCH_COST_WEIGHT=0.30
export INFER_FA_IMPL=triton-invariant
export VEOMNI_ATTN_IMPLEMENTATION=triton-invariant
export MODELING_BACKEND=veomni
export VEOMNI_USE_LIGER_KERNEL=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$repo"
runner=("$vexact_repo/.venv/bin/python" -m verl.trainer.main_ppo)
if [[ "$validate_only" -eq 1 ]]; then
  runner=("$vexact_repo/.venv/bin/python" "$repo/scripts/validate_verl_config.py")
fi

set +e
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=4,5,6,7 \
  "${runner[@]}" \
  model_engine=veomni \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$train_file" \
  data.val_files="$train_file" \
  data.train_batch_size="$max_samples" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.seed="$seed" \
  actor_rollout_ref.model.path="$repo/outputs/rl/03_hf_evidence_step400" \
  actor_rollout_ref.model.external_lib=vexact.integrations.verl.fsdp_enable_invariant \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=triton-invariant \
  actor_rollout_ref.actor.optim.lr=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.veomni.fsdp_size=4 \
  actor_rollout_ref.actor.veomni.attn_implementation=triton-invariant \
  actor_rollout_ref.actor.veomni.forward_only=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vexact \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.seed="$seed" \
  actor_rollout_ref.rollout.n="$n_rollouts" \
  actor_rollout_ref.rollout.temperature=0.9 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.n="$n_rollouts" \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.9 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks=256 \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=triton-invariant \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=384 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$repo/configs/rl/candidate_bm25_tool.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=eca_search_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$repo/configs/rl/eca_agent_loop.yaml" \
  reward.custom_reward_function.path="$repo/src/rl/rewards_boundary.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.logger='["console"]' \
  trainer.project_name=eca_rollout_alignment \
  trainer.experiment_name="$experiment_name" \
  trainer.default_local_dir="$output_abs/ckpt_scratch" \
  2>&1 | tee "$run_log"
run_rc=${PIPESTATUS[0]}
set -e

echo "${stage}_run_rc=$run_rc"
[[ "$run_rc" -eq 0 ]]
if [[ "$validate_only" -eq 1 ]]; then
  exit 0
fi
test -s "$dump_file"

"$vexact_repo/.venv/bin/python" - \
  "$dump_file" \
  "$repo/outputs/rl/04_table_search_boundary/boundary_latest.json" \
  "$summary_file" \
  "$stage" \
  "$max_samples" \
  "$n_rollouts" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
table = json.load(open(sys.argv[2], encoding="utf-8"))
table = table.get("boundary", table)
stage = sys.argv[4]
n_questions = int(sys.argv[5])
n_rollouts = int(sys.argv[6])
assert len(rows) == n_questions * n_rollouts, (
    f"expected {n_questions * n_rollouts} trajectories, got {len(rows)}"
)

def boundary(sample_id):
    value = table[sample_id]
    if isinstance(value, dict):
        return value.get("boundary") or value.get("label")
    return value

finish_rate = sum(row["finish"] for row in rows) / len(rows)
clip_ratio = sum(row["hit_response_cap"] for row in rows) / len(rows)
missing_rate = sum(row["final_answer_missing"] for row in rows) / len(rows)
reserve_violations = sum(row["final_answer_reserve_violations"] for row in rows)
max_assistant = max(row["max_assistant_turn_tokens"] for row in rows)
max_observation = max(row["max_observation_turn_tokens"] for row in rows)
all_closes = [tag for row in rows for tag in row["turn_close_tags"]]
continued_turns = sum(sum(row.get("turn_continued", [])) for row in rows)
unresolved_unclosed_turns = 0
for row in rows:
    tags = row["turn_close_tags"]
    continued = row.get("turn_continued", [False] * len(tags))
    assert len(tags) == len(continued), "turn close/continuation audit length mismatch"
    for tag, was_continued in zip(tags, continued):
        if tag is None and not was_continued:
            unresolved_unclosed_turns += 1
        elif tag is not None and tag not in ("</search>", "</internal>", "</answer>"):
            unresolved_unclosed_turns += 1
routes = Counter(row["route_first"] for row in rows)
nosearch = [row for row in rows if boundary(row["sample_id"]) == "NoSearch"]
p_internal_nosearch = sum(row["route_first"] == "internal" for row in nosearch) / len(nosearch)
groups = defaultdict(set)
for row in rows:
    groups[row["sample_id"]].add(row["route_first"])
mixed_rate = sum({"search", "internal"} <= actions for actions in groups.values()) / len(groups)

trajectory_pass = (
    finish_rate >= 0.95
    and clip_ratio < 0.05
    and missing_rate == 0
    and reserve_violations == 0
    and max_assistant <= 256
    and max_observation <= 384
    and unresolved_unclosed_turns == 0
)
gate_pass = trajectory_pass and (
    (p_internal_nosearch > 0) if stage == "a3"
    else (p_internal_nosearch > 0.10 and mixed_rate > 0)
)
summary = {
    "gate": (
        ("GATE_B_PASS" if gate_pass else "GATE_B_FAIL")
        if stage == "a3"
        else ("A4_EXACT_PARITY_PASS" if gate_pass else "A4_EXACT_PARITY_FAIL")
    ),
    "stage": stage,
    "n_questions": n_questions,
    "n_rollouts": n_rollouts,
    "finish_rate": finish_rate,
    "clip_ratio": clip_ratio,
    "final_answer_missing_rate": missing_rate,
    "final_answer_reserve_violations": reserve_violations,
    "max_assistant_turn_tokens": max_assistant,
    "max_observation_turn_tokens": max_observation,
    "complete_close_sequence_rate": (
        sum(tag is not None for tag in all_closes) / len(all_closes) if all_closes else 0.0
    ),
    "continued_capped_turns": continued_turns,
    "unresolved_unclosed_turns": unresolved_unclosed_turns,
    "routes": dict(routes),
    "p_internal_NoSearch": p_internal_nosearch,
    "mixed_action_group_rate": mixed_rate,
}
Path(sys.argv[3]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
assert gate_pass, summary
PY
