#!/usr/bin/env python3
"""Freeze the one-shot Fresh Step-Val3 split without opening original Test outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from build_cur1_dataset import canonical_id, collect_json_ids, prompt_digest, sha256_file
from build_grpo_smoke_dataset import _read_jsonl, _to_verl_row
from src.rl.candidate_index import write_contexts_jsonl


def known_historical_ids(out_dir: Path):
    """Collect identifiers while explicitly refusing sealed Test parquet reads."""
    ids: set[str] = set()
    paths: set[Path] = set()
    for root in (REPO / "data/eval", REPO / "data/rl", REPO / "data/cur"):
        if not root.exists():
            continue
        paths.update(
            path for path in root.rglob("*")
            if path.suffix in {".txt", ".json", ".jsonl", ".parquet"}
            and not path.is_relative_to(out_dir)
            and path.name not in {"test.parquet"}
        )
    for phase in range(16, 26):
        for root in (REPO / "results").glob(f"{phase}_*"):
            paths.update(path for path in root.rglob("manifest.json") if path.is_file())
    counts = {}
    for path in sorted(paths):
        before = len(ids)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            for row in pq.read_table(path).to_pylist():
                collect_json_ids(row, ids)
        elif path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                if line.strip():
                    collect_json_ids(json.loads(line), ids)
        elif path.suffix == ".json":
            collect_json_ids(json.loads(path.read_text()), ids)
        else:
            for line in path.read_text().splitlines():
                value = canonical_id(line)
                if value:
                    ids.add(value)
        counts[str(path.relative_to(REPO))] = len(ids) - before
    return ids, counts


def make_row(sample, index, agent, policy, root_arm=""):
    row = _to_verl_row(sample, split="validation", idx=index)
    row["agent_name"] = agent
    extra = dict(row["extra_info"])
    extra.update({
        "step_val3": True,
        "step_val3_question_index": index,
        "step_val3_arm": policy,
        "step_policy": policy if agent == "eca_step_adaptive_agent" else "fixed",
        "cur_forced_arm": root_arm,
    })
    row["extra_info"] = extra
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026081203)
    ap.add_argument("--n-questions", type=int, default=128)
    ap.add_argument("--pool", type=Path, default=REPO / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl")
    ap.add_argument("--model", type=Path, default=REPO / "outputs/rl/03_hf_evidence_step400")
    ap.add_argument("--gate-dir", type=Path, default=REPO / "results/25_step_adaptive/step_gate/models")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/cur/step_val3_fresh128")
    args = ap.parse_args()
    if (args.out_dir / "manifest.json").exists():
        raise SystemExit(f"REFUSE_REFREEZE={args.out_dir / 'manifest.json'}")
    excluded, exclusion_counts = known_historical_ids(args.out_dir.resolve())
    pool = _read_jsonl(args.pool)
    eligible = [row for row in pool if canonical_id(str(row.get("sample_id"))) not in excluded]
    if len(eligible) < args.n_questions:
        raise SystemExit(f"eligible={len(eligible)} required={args.n_questions}")
    random.Random(args.seed).shuffle(eligible)
    selected = eligible[:args.n_questions]
    selected_ids = [str(row["sample_id"]) for row in selected]
    if len(set(selected_ids)) != args.n_questions:
        raise RuntimeError("Val3 duplicate IDs")

    from datasets import Dataset
    from transformers import AutoTokenizer
    from verl.utils.tokenizer.chat_template import apply_chat_template

    args.out_dir.mkdir(parents=True, exist_ok=False)
    root_rows, allsearch_rows, gate_rows = [], [], []
    for index, sample in enumerate(selected):
        root_rows.extend([
            make_row(sample, index, "eca_search_agent", "no_search", "internal"),
            make_row(sample, index, "eca_search_agent", "old_always_search", "search"),
        ])
        allsearch_rows.append(make_row(sample, index, "eca_step_adaptive_agent", "all_search"))
        gate_rows.append(make_row(sample, index, "eca_step_adaptive_agent", "frozen_gate"))
    Dataset.from_list(root_rows).to_parquet(str(args.out_dir / "root_baselines.parquet"))
    Dataset.from_list(allsearch_rows).to_parquet(str(args.out_dir / "step_allsearch.parquet"))
    Dataset.from_list(gate_rows).to_parquet(str(args.out_dir / "step_gate.parquet"))
    Dataset.from_list(root_rows[:4]).to_parquet(str(args.out_dir / "root_smoke2.parquet"))
    Dataset.from_list(allsearch_rows[:2]).to_parquet(str(args.out_dir / "step_allsearch_smoke2.parquet"))
    Dataset.from_list(gate_rows[:2]).to_parquet(str(args.out_dir / "step_gate_smoke2.parquet"))
    write_contexts_jsonl(selected, args.out_dir / "contexts_index.jsonl")
    (args.out_dir / "val3_ids.txt").write_text("\n".join(selected_ids) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    prompt_rows = []
    for row in gate_rows:
        ids = list(apply_chat_template(tokenizer, row["prompt"], tools=None, tokenize=True, add_generation_prompt=True))
        prompt_rows.append({
            "sample_id": row["extra_info"]["sample_id"], "split": "validation",
            "canonical_prompt_sha256": prompt_digest(ids), "canonical_prompt_len": len(ids),
            "canonical_prompt_ids": ids,
        })
    (args.out_dir / "prompt_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in prompt_rows))

    artifact_names = [
        "root_baselines.parquet", "step_allsearch.parquet", "step_gate.parquet",
        "root_smoke2.parquet", "step_allsearch_smoke2.parquet", "step_gate_smoke2.parquet",
        "contexts_index.jsonl", "val3_ids.txt", "prompt_manifest.jsonl",
    ]
    gate_files = ["deployment_freeze.json", "summary.json", "threshold.json", "pca_l27.npz", "scaler.npz", "seed1.pt", "seed2.pt", "seed3.pt"]
    manifest = {
        "gate": "STEP_VAL3_SPLIT_FREEZE_PASS",
        "seed": args.seed,
        "questions": args.n_questions,
        "question_ids": selected_ids,
        "question_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
        "source_pool": str(args.pool), "source_pool_sha256": sha256_file(args.pool),
        "historical_id_count": len(excluded),
        "historical_exclusion_new_id_counts": exclusion_counts,
        "eligible_pool_count": len(eligible),
        "arms": ["no_search", "old_always_search", "step_all_search", "frozen_step_gate"],
        "rollouts_per_question_per_arm": 1,
        "sampling": {"temperature": 0, "top_p": 1, "seed": args.seed},
        "scientific_pass": {"calls_ratio_vs_step_allsearch_max": 0.75, "f1_delta_floor": -0.02},
        "project_pass": {"token_cost_strictly_lower_than_old_alwayssearch": True, "f1_delta_floor": -0.02},
        "gate_artifact_sha256": {name: sha256_file(args.gate_dir / name) for name in gate_files},
        "frozen_artifact_sha256": {name: sha256_file(args.out_dir / name) for name in artifact_names},
        "original_test_parquet_read": False,
        "original_test_outcomes_read": False,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"question_ids", "historical_exclusion_new_id_counts"}}, indent=2))


if __name__ == "__main__":
    main()
