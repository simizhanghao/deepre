#!/usr/bin/env python3
"""Freeze the fresh DSSR Val2 split without touching the sealed CUR-1 Test."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from build_cur1_dataset import canonical_id, load_historical_ids, prompt_digest, sha256_file
from build_grpo_smoke_dataset import _read_jsonl, _to_verl_row
from src.rl.candidate_index import write_contexts_jsonl


def hash_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def search_row(sample: dict[str, Any], index: int) -> dict[str, Any]:
    row = _to_verl_row(sample, split="validation", idx=index)
    row["extra_info"].update(
        {
            "cur_phase": "dssr",
            "cur_split": "val2",
            "cur_pair_id": str(sample["sample_id"]),
            "cur_forced_arm": "search",
            "cur_question_index": index,
        }
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026081202)
    ap.add_argument("--n-questions", type=int, default=128)
    ap.add_argument(
        "--pool",
        type=Path,
        default=REPO / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl",
    )
    ap.add_argument(
        "--cur1-manifest", type=Path, default=REPO / "data/cur/cur1_fresh896/manifest.json"
    )
    ap.add_argument(
        "--model", type=Path, default=REPO / "outputs/rl/03_hf_evidence_step400"
    )
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/cur/dssr_val2_fresh128")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    excluded, exclusion_counts = load_historical_ids(out_dir)
    cur1 = json.loads(args.cur1_manifest.read_text())
    cur1_ids = {
        str(sid) for split_ids in cur1["question_ids"].values() for sid in split_ids
    }
    excluded.update(filter(None, (canonical_id(sid) for sid in cur1_ids)))

    excluded_sorted = sorted(excluded)
    pool = _read_jsonl(args.pool)
    eligible = [
        row for row in pool if canonical_id(str(row.get("sample_id"))) not in excluded
    ]
    if len(eligible) < args.n_questions:
        raise SystemExit(
            f"eligible={len(eligible)} required={args.n_questions} excluded={len(excluded)}"
        )
    random.Random(args.seed).shuffle(eligible)
    selected = eligible[: args.n_questions]
    selected_ids = [str(row["sample_id"]) for row in selected]
    if len(set(selected_ids)) != args.n_questions or set(selected_ids) & cur1_ids:
        raise SystemExit("Val2 duplicates or overlaps CUR-1")

    from datasets import Dataset
    from transformers import AutoTokenizer
    from verl.utils.tokenizer.chat_template import apply_chat_template

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [search_row(sample, i) for i, sample in enumerate(selected)]
    Dataset.from_list(rows).to_parquet(str(args.out_dir / "search.parquet"))
    Dataset.from_list(rows[:8]).to_parquet(str(args.out_dir / "smoke_8.parquet"))
    write_contexts_jsonl(selected, args.out_dir / "contexts_index.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    prompt_rows = []
    for row in rows:
        ids = list(
            apply_chat_template(
                tokenizer, row["prompt"], tools=None, tokenize=True, add_generation_prompt=True
            )
        )
        prompt_rows.append(
            {
                "sample_id": row["extra_info"]["sample_id"],
                "split": "val2",
                "canonical_prompt_sha256": prompt_digest(ids),
                "canonical_prompt_len": len(ids),
                "canonical_prompt_ids": ids,
            }
        )
    (args.out_dir / "prompt_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prompt_rows)
    )
    (args.out_dir / "val2_ids.txt").write_text("\n".join(selected_ids) + "\n")

    model_files = [
        args.model / name
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    ]
    artifact_names = [
        "search.parquet",
        "smoke_8.parquet",
        "contexts_index.jsonl",
        "prompt_manifest.jsonl",
        "val2_ids.txt",
    ]
    manifest = {
        "gate": "DSSR_VAL2_SPLIT_FREEZE_PASS",
        "seed": args.seed,
        "source_pool": str(args.pool),
        "source_pool_sha256": sha256_file(args.pool),
        "historical_id_count": len(excluded),
        "historical_exclusion_sha256": hash_lines(excluded_sorted),
        "historical_exclusion_new_id_counts": exclusion_counts,
        "cur1_manifest": str(args.cur1_manifest),
        "cur1_manifest_sha256": sha256_file(args.cur1_manifest),
        "cur1_all_ids_excluded": True,
        "eligible_pool_count": len(eligible),
        "split_size": args.n_questions,
        "split_id_sha256": hash_lines(selected_ids),
        "question_ids": selected_ids,
        "model_path": str(args.model),
        "model_contract_sha256": {path.name: sha256_file(path) for path in model_files},
        "frozen_artifact_sha256": {
            name: sha256_file(args.out_dir / name) for name in artifact_names
        },
        "acquisition_plan": {
            "probe": {"questions": args.n_questions, "rollouts": 1, "trajectories": 128},
            "search": {"questions": args.n_questions, "rollouts": 4, "trajectories": 512},
            "total_trajectories": 640,
        },
        "probe_contract": {
            "forced_prefix": "<internal>",
            "temperature": 0,
            "max_total_response_tokens": 96,
            "tool_enabled": False,
            "closed_answer_required": True,
        },
        "original_test_sealed": True,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in {"question_ids", "historical_exclusion_new_id_counts"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
