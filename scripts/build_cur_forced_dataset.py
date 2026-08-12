#!/usr/bin/env python3
"""Build fresh paired CUR-0 do(search)/do(internal) data."""

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

from build_grpo_smoke_dataset import _read_jsonl, _to_verl_row
from src.rl.candidate_index import write_contexts_jsonl


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return {str(x) for key in ("train_ids", "val_ids") for x in obj.get(key, [])}
    return {x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--pool", type=Path, default=REPO / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/cur/cur0_fresh128")
    args = ap.parse_args()

    exclusion_paths = [
        REPO / "data/eval/hotpotqa_200_ids.txt",
        REPO / "data/rl/train_smoke_128/train_ids.txt",
        REPO / "data/rl/train_smoke_128/manifest.json",
    ]
    excluded: set[str] = set()
    for path in exclusion_paths:
        excluded |= load_ids(path)
    pool = [r for r in _read_jsonl(args.pool) if str(r.get("sample_id")) not in excluded]
    random.Random(args.seed).shuffle(pool)
    selected = pool[: args.n_questions]
    if len(selected) != args.n_questions:
        raise SystemExit(f"eligible={len(pool)} requested={args.n_questions}")

    rows = []
    for qidx, sample in enumerate(selected):
        for arm_idx, arm in enumerate(("search", "internal")):
            row = _to_verl_row(sample, split="cur0", idx=qidx * 2 + arm_idx)
            row["extra_info"].update({
                "cur_pair_id": str(sample["sample_id"]),
                "cur_forced_arm": arm,
                "cur_question_index": qidx,
            })
            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import datasets

    datasets.Dataset.from_list(rows).to_parquet(str(args.out_dir / "paired_128.parquet"))
    datasets.Dataset.from_list(rows[:4]).to_parquet(str(args.out_dir / "smoke_2.parquet"))
    write_contexts_jsonl(selected, args.out_dir / "contexts_index.jsonl")
    ids = [str(r["sample_id"]) for r in selected]
    manifest = {
        "purpose": "CUR-0 fresh paired action-level intervention",
        "seed": args.seed,
        "n_questions": len(selected),
        "n_rows": len(rows),
        "arms": ["search", "internal"],
        "excluded_ids": len(excluded),
        "exclusion_paths": [str(x) for x in exclusion_paths],
        "question_ids": ids,
        "question_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "boundary_v1_used": False,
        "canonical_prompt_intervention": "after prompt, before first generated token",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "question_ids"}, indent=2))


if __name__ == "__main__":
    main()
