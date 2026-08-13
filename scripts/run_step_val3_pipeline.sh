#!/usr/bin/env bash
set -euo pipefail
repo=/data1/hcc/deepresearch
vexact=/data1/hcc/eca-verl-vexact
cd "$repo"
mkdir -p results/25_step_adaptive/val3

if ! tmux has-session -t step_gate_server 2>/dev/null; then
  tmux new-session -d -s step_gate_server "cd $repo && bash scripts/start_step_gate_server.sh 2>&1 | tee results/25_step_adaptive/val3/gate_server.log"
fi
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:8007/health >/dev/null && break; sleep 2; done
curl -fsS http://127.0.0.1:8007/health | tee results/25_step_adaptive/val3/gate_health.json

if [[ ! -f data/cur/step_val3_fresh128/manifest.json ]]; then
  env -u LD_LIBRARY_PATH "$vexact/.venv/bin/python" scripts/build_step_val3_dataset.py \
    | tee results/25_step_adaptive/val3/data_freeze.log
fi

mkdir -p results/25_step_adaptive/val3/root_static
if [[ ! -s results/25_step_adaptive/val3/root_static/root_scores.json ]]; then
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=6 "$vexact/.venv/bin/python" \
    scripts/extract_cur1_features.py \
    --manifest data/cur/step_val3_fresh128/prompt_manifest.jsonl \
    --ids data/cur/step_val3_fresh128/val3_ids.txt \
    --split validation --model outputs/rl/03_hf_evidence_step400 \
    --output results/25_step_adaptive/val3/root_static/features.npz --batch-size 32 \
    2>&1 | tee results/25_step_adaptive/val3/root_static/extract.log
  env -u LD_LIBRARY_PATH "$vexact/.venv/bin/python" scripts/freeze_step_val3_root_scores.py \
    --features results/25_step_adaptive/val3/root_static/features.npz \
    --root-model-dir results/23_cur1/offline/b0_b6_v1/models/b3 \
    --output results/25_step_adaptive/val3/root_static/root_scores.json \
    | tee results/25_step_adaptive/val3/root_static/freeze.log
fi

# Restart once so every Val3 request consumes the fixed ordered-batch32 root
# score map; this prevents BF16 batch-shape drift in the inherited B3 scalar.
tmux kill-session -t step_gate_server 2>/dev/null || true
tmux new-session -d -s step_gate_server "cd $repo && bash scripts/start_step_gate_server.sh 2>&1 | tee results/25_step_adaptive/val3/gate_server.log"
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:8007/health >/dev/null && break; sleep 2; done
curl -fsS http://127.0.0.1:8007/health | tee results/25_step_adaptive/val3/gate_health.json
env -u LD_LIBRARY_PATH "$vexact/.venv/bin/python" - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('data/cur/step_val3_fresh128/manifest.json')
x=json.loads(p.read_text()); freeze=Path('results/25_step_adaptive/step_gate/models/deployment_freeze.json')
x['runtime_deployment_freeze_sha256']=hashlib.sha256(freeze.read_bytes()).hexdigest()
x['root_score_map_sha256']=hashlib.sha256(Path('results/25_step_adaptive/val3/root_static/root_scores.json').read_bytes()).hexdigest()
x['runtime_frozen_before_outcomes']=True
p.write_text(json.dumps(x,indent=2)+'\n')
print('STEP_VAL3_RUNTIME_FREEZE_PASS')
PY

if ! tmux has-session -t step_val3_retriever 2>/dev/null; then
  tmux new-session -d -s step_val3_retriever "cd $repo && bash scripts/start_step_val3_retriever.sh 2>&1 | tee results/25_step_adaptive/val3/retriever.log"
fi
for _ in $(seq 1 30); do curl -fsS http://127.0.0.1:8006/health >/dev/null && break; sleep 1; done
curl -fsS http://127.0.0.1:8006/health >/dev/null

bash scripts/run_step_val3_arm.sh step_gate_smoke | tee results/25_step_adaptive/val3/step_gate_smoke_launcher.log
env -u LD_LIBRARY_PATH "$vexact/.venv/bin/python" - <<'PY'
import json
p='results/25_step_adaptive/val3/step_gate_smoke/step_records.jsonl'
rows=[json.loads(x) for x in open(p) if x.strip()]
assert len(rows)==4
assert all(r['metrics']['step_policy']=='frozen_gate' for r in rows)
assert all(r['finish']==1 for r in rows)
assert sum(len(r['metrics']['gate_probabilities']) for r in rows)>0
print('STEP_VAL3_ONLINE_GATE_SMOKE_PASS')
PY

bash scripts/run_step_val3_arm.sh root | tee results/25_step_adaptive/val3/root_launcher.log
bash scripts/run_step_val3_arm.sh step_allsearch | tee results/25_step_adaptive/val3/step_allsearch_launcher.log
bash scripts/run_step_val3_arm.sh step_gate | tee results/25_step_adaptive/val3/step_gate_launcher.log

env -u LD_LIBRARY_PATH "$vexact/.venv/bin/python" scripts/analyze_step_val3.py \
  --data-dir data/cur/step_val3_fresh128 \
  --root-outcomes results/25_step_adaptive/val3/root/outcomes.jsonl \
  --step-allsearch results/25_step_adaptive/val3/step_allsearch/step_records.jsonl \
  --step-gate results/25_step_adaptive/val3/step_gate/step_records.jsonl \
  --model outputs/rl/03_hf_evidence_step400 \
  --output-dir results/25_step_adaptive/val3/analysis \
  | tee results/25_step_adaptive/val3/analysis.log

echo STEP_VAL3_PIPELINE_COMPLETE
