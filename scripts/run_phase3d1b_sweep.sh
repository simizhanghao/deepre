#!/usr/bin/env bash
# Phase 3D1b — short online Uniform Cost λ phase diagram (NOT long train).
#
# Usage:
#   bash scripts/run_phase3d1b_sweep.sh
#   LAMBDAS="0.05 0.10 0.15 0.20" STEPS=60 bash scripts/run_phase3d1b_sweep.sh
#   tmux new -s eca-3d1b 'bash scripts/run_phase3d1b_sweep.sh; exec bash'
#
# Each λ: fresh SFT-v1, ~40–60 steps, SAVE_FREQ=999, early-stop watcher.
set -euo pipefail

# tmux servers often drop supplementary groups → docker.sock permission denied.
# Re-exec under `sg docker` once if needed.
if ! docker info >/dev/null 2>&1; then
  if sg docker -c 'docker info' >/dev/null 2>&1; then
    exec sg docker -c "cd \"${PWD}\" && bash \"${BASH_SOURCE[0]}\" ${*@Q}"
  fi
  echo "ERROR: cannot talk to docker (not in docker group / daemon down)" >&2
  exit 1
fi

REPO=${REPO:-/data1/hcc/deepresearch}
CONTAINER=${CONTAINER:-eca-verl}
STEPS=${STEPS:-60}
SAVE_FREQ=${SAVE_FREQ:-999}
LAMBDAS=${LAMBDAS:-"0.05 0.10 0.15 0.20"}
ECA_EVIDENCE_WEIGHT=${ECA_EVIDENCE_WEIGHT:-0.5}
# Throughput knobs (same for all λ). 3B+multi-turn leaves headroom on 80G;
# screenshot ~12G is usually ref_log_prob phase — bump util/batch for rollout.
BATCH=${BATCH:-16}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}
MICRO_BATCH=${MICRO_BATCH:-2}
N=${N:-4}
STAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR=${RESULT_DIR:-$REPO/results/phase3d1b_online_lambda_${STAMP}}
mkdir -p "$RESULT_DIR" "$REPO/logs"
ln -sfn "$RESULT_DIR" "$REPO/results/phase3d1b_latest"

cd "$REPO"
command -v docker >/dev/null
docker start "$CONTAINER" >/dev/null

if ! curl -sf http://127.0.0.1:8001/health >/dev/null; then
  echo "Starting Candidate-BM25 on :8001 ..."
  nohup bash -lc "cd $REPO; (conda activate deepresearch 2>/dev/null || true); python scripts/start_candidate_retrieval_server.py --index data/rl/grpo_smoke_128/contexts_index.jsonl --port 8001" \
    >"$REPO/logs/retriever_8001.log" 2>&1 &
  sleep 3
fi
curl -sf http://127.0.0.1:8001/health >/dev/null || { echo "retriever failed"; exit 1; }
echo RETRIEVER_OK

echo "lambda,max_step,mean_search_last10,mean_answer_last10,mean_kl_last10,stop_reason" \
  >"$RESULT_DIR/phase_diagram.csv"

summarize_log() {
  local log="$1" lam="$2" reason="$3"
  python3 - "$log" "$lam" "$reason" "$RESULT_DIR" <<'PY'
import json, re, sys
from pathlib import Path
log, lam, reason, root = sys.argv[1:5]
root = Path(root)
pat = re.compile(
    r"\[phase3c\] step=(\d+) answer=([0-9.]+).*?search_rate=([0-9.]+).*?kl=([0-9.eE+-]+)"
)
rows = []
for line in Path(log).open(errors="ignore"):
    m = pat.search(line)
    if m:
        rows.append(
            {
                "step": int(m.group(1)),
                "answer": float(m.group(2)),
                "search_rate": float(m.group(3)),
                "kl": float(m.group(4)),
            }
        )
tag = f"{float(lam):.2f}".replace(".", "p")
if not rows:
    open(root / "phase_diagram.csv", "a").write(f"{lam},0,,,,{reason}|no_metrics\n")
    print(f"{lam}: no metrics")
else:
    last = rows[-10:] if len(rows) >= 10 else rows
    ms = sum(r["search_rate"] for r in last) / len(last)
    ma = sum(r["answer"] for r in last) / len(last)
    mk = sum(r["kl"] for r in last) / len(last)
    mx = rows[-1]["step"]
    open(root / "phase_diagram.csv", "a").write(
        f"{lam},{mx},{ms:.4f},{ma:.4f},{mk:.4f},{reason}\n"
    )
    (root / f"lambda_{tag}.json").write_text(
        json.dumps(
            {
                "lambda": float(lam),
                "n": len(rows),
                "last10_search": ms,
                "last10_answer": ma,
                "last10_kl": mk,
                "max_step": mx,
                "stop_reason": reason,
                "series": rows,
            },
            indent=2,
        )
    )
    print(f"λ={lam} max_step={mx} last10_search={ms:.3f} answer={ma:.3f} kl={mk:.3f} ({reason})")
PY
}

for lam in $LAMBDAS; do
  tag=$(python3 -c "print(f'{float(\"$lam\"):.2f}'.replace('.',''))")
  EXP="grpo_sftv1_cost_3d1b_ls${tag}"
  OUT_C="/workspace/deepresearch/outputs/rl/${EXP}"
  LOG_NAME="grpo_${EXP}_to${STEPS}_${STAMP}.log"
  LOG="$REPO/logs/$LOG_NAME"
  CONTAINER_LOG="/workspace/deepresearch/logs/$LOG_NAME"
  mkdir -p "$REPO/outputs/rl/${EXP}"
  : >"$LOG"
  ln -sfn "$LOG" "$REPO/logs/grpo_3d1b_latest.log"

  echo "======== 3D1b λ=$lam STEPS<=$STEPS OUT=outputs/rl/${EXP} ========"

  docker exec "$CONTAINER" bash -lc \
    'ray stop --force >/dev/null 2>&1 || true; pkill -9 -f launch_grpo_main >/dev/null 2>&1 || true; pkill -9 -f run_grpo_cost >/dev/null 2>&1 || true; pkill -9 -f sglang.launch_server >/dev/null 2>&1 || true; exit 0' \
    || true
  sleep 2

  docker exec "$CONTAINER" bash -lc \
    "mkdir -p /workspace/deepresearch/outputs/rl/tensorboard/${EXP} ${OUT_C} /workspace/deepresearch/logs && chmod -R a+rwX /workspace/deepresearch/outputs/rl /workspace/deepresearch/logs" \
    || true

  docker exec -d "$CONTAINER" bash -lc "\
    export PYTHONPATH=/workspace/deepresearch:/workspace/verl; \
    export TENSORBOARD_DIR=/workspace/deepresearch/outputs/rl/tensorboard/${EXP}; \
    export ECA_EVIDENCE_WEIGHT=${ECA_EVIDENCE_WEIGHT}; \
    export ECA_SEARCH_COST_WEIGHT=${lam}; \
    cd /workspace/deepresearch; \
    STEPS=${STEPS} SAVE_FREQ=${SAVE_FREQ} OUT_DIR=${OUT_C} \
    EXPERIMENT_NAME=${EXP} RESUME_MODE=disable \
    TOTAL_EPOCHS=${STEPS} \
    BATCH=${BATCH} N=${N} GPU_MEM_UTIL=${GPU_MEM_UTIL} MICRO_BATCH=${MICRO_BATCH} \
    ECA_EVIDENCE_WEIGHT=${ECA_EVIDENCE_WEIGHT} \
    ECA_SEARCH_COST_WEIGHT=${lam} \
    bash /workspace/deepresearch/scripts/run_grpo_cost.sh \
      >${CONTAINER_LOG} 2>&1"

  sleep 10
  if ! docker exec "$CONTAINER" bash -lc 'pgrep -f launch_grpo_main >/dev/null || pgrep -f run_grpo_cost >/dev/null'; then
    echo "WARN: train not visible for λ=$lam"
    tail -50 "$LOG" || true
  fi

  rm -f "${LOG}.stop_reason"
  set +e
  bash "$REPO/scripts/watch_3d1b_early_stop.sh" "$LOG" "$STEPS"
  set -e

  docker exec "$CONTAINER" bash -lc \
    'pkill -9 -f launch_grpo_main >/dev/null 2>&1; pkill -9 -f run_grpo_cost >/dev/null 2>&1; ray stop --force >/dev/null 2>&1; exit 0' \
    || true
  sleep 3

  reason="max_step"
  [[ -f "${LOG}.stop_reason" ]] && reason=$(cat "${LOG}.stop_reason")
  summarize_log "$LOG" "$lam" "$reason"
done

python3 - "$RESULT_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
data = []
for p in sorted(root.glob("lambda_*.json")):
    data.append(json.loads(p.read_text()))
notes, healthy = [], []
for d in data:
    sr = float(d.get("last10_search") or 0)
    lam = d["lambda"]
    if sr < 0.05:
        notes.append(f"λ={lam}: COLLAPSE search≈{sr:.2f}")
    elif sr > 0.95:
        notes.append(f"λ={lam}: STILL_ALWAYS_SEARCH search≈{sr:.2f}")
    elif 0.4 <= sr <= 0.9:
        healthy.append(lam)
        notes.append(f"λ={lam}: CANDIDATE search≈{sr:.2f}")
    else:
        notes.append(f"λ={lam}: TRANSITION search≈{sr:.2f}")
verdict = (
    "HAS_PARETO_CANDIDATE → pick best λ for formal 3D1@400"
    if healthy
    else "NO_STABLE_UNIFORM_WINDOW → trigger 3D2 Capability-Aware Cost"
)
summary = {
    "phase": "3D1b",
    "purpose": "online λ phase diagram; not long-train",
    "lambdas": data,
    "notes": notes,
    "verdict": verdict,
    "also_known": "λ=0.40 already COLLAPSE (3D1 FAIL)",
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "======== 3D1b DONE ========"
echo "results: $RESULT_DIR"
echo "diagram: $RESULT_DIR/phase_diagram.csv"
echo "summary: $RESULT_DIR/summary.json"
