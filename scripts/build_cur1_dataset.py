#!/usr/bin/env python3
"""Freeze CUR-1 fresh train/validation/test splits and paired arm datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from build_grpo_smoke_dataset import _read_jsonl, _to_verl_row
from src.rl.candidate_index import write_contexts_jsonl

HEX_ID = re.compile(r"([0-9a-f]{24})$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = HEX_ID.search(value.strip())
    return match.group(1).lower() if match else None


def collect_json_ids(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            collect_json_ids(value, out)
    elif isinstance(obj, list):
        for value in obj:
            collect_json_ids(value, out)
    else:
        cid = canonical_id(obj)
        if cid:
            out.add(cid)


def history_paths(exclude_root: Path) -> list[Path]:
    paths: set[Path] = set()
    for root in (REPO / "data/eval", REPO / "data/rl", REPO / "data/cur"):
        if root.exists():
            paths.update(
                p for p in root.rglob("*")
                if p.suffix in {".txt", ".json", ".jsonl", ".parquet"}
                and not p.is_relative_to(exclude_root)
            )
    for phase in range(16, 23):
        for root in (REPO / "results").glob(f"{phase}_*"):
            paths.update(
                p for p in root.rglob("*")
                if p.suffix == ".parquet" or (p.suffix in {".json", ".jsonl", ".txt"} and "manifest" in p.name)
            )
    return sorted(paths)


def load_historical_ids(exclude_root: Path) -> tuple[set[str], dict[str, int]]:
    excluded: set[str] = set()
    counts: dict[str, int] = {}
    for path in history_paths(exclude_root):
        before = len(excluded)
        try:
            if path.suffix == ".parquet":
                import pyarrow.parquet as pq

                for row in pq.read_table(path).to_pylist():
                    collect_json_ids(row, excluded)
            elif path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            collect_json_ids(json.loads(line), excluded)
            elif path.suffix == ".json":
                collect_json_ids(json.loads(path.read_text(encoding="utf-8")), excluded)
            else:
                for line in path.read_text(encoding="utf-8").splitlines():
                    cid = canonical_id(line)
                    if cid:
                        excluded.add(cid)
        except Exception as exc:
            raise RuntimeError(f"failed historical-ID audit for {path}: {exc}") from exc
        counts[str(path.relative_to(REPO))] = len(excluded) - before
    return excluded, counts


def prompt_digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()


def paired_rows(samples: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows = []
    for qidx, sample in enumerate(samples):
        for arm_idx, arm in enumerate(("search", "internal")):
            row = _to_verl_row(sample, split=split, idx=qidx * 2 + arm_idx)
            row["extra_info"].update(
                {
                    "cur_phase": "cur1",
                    "cur_split": split,
                    "cur_pair_id": str(sample["sample_id"]),
                    "cur_forced_arm": arm,
                    "cur_question_index": qidx,
                }
            )
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026081201)
    ap.add_argument("--n-train", type=int, default=640)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--pool", type=Path, default=REPO / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl")
    ap.add_argument("--model", type=Path, default=REPO / "outputs/rl/03_hf_evidence_step400")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/cur/cur1_fresh896")
    args = ap.parse_args()

    excluded, exclusion_counts = load_historical_ids(args.out_dir.resolve())
    pool = _read_jsonl(args.pool)
    eligible = [row for row in pool if canonical_id(str(row.get("sample_id"))) not in excluded]
    required = args.n_train + args.n_val + args.n_test
    if len(eligible) < required:
        raise SystemExit(f"eligible={len(eligible)} required={required} excluded={len(excluded)}")
    random.Random(args.seed).shuffle(eligible)
    selected = eligible[:required]
    split_raw = {
        "train": selected[: args.n_train],
        "validation": selected[args.n_train : args.n_train + args.n_val],
        "test": selected[args.n_train + args.n_val :],
    }
    split_ids = {key: [str(x["sample_id"]) for x in rows] for key, rows in split_raw.items()}
    if len(set().union(*(set(x) for x in split_ids.values()))) != required:
        raise SystemExit("split overlap or duplicate sample IDs")

    from datasets import Dataset
    from transformers import AutoTokenizer
    from verl.utils.tokenizer.chat_template import apply_chat_template

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {key: paired_rows(value, key) for key, value in split_raw.items()}
    for split, rows in rows_by_split.items():
        Dataset.from_list(rows).to_parquet(str(args.out_dir / f"{split}.parquet"))
    Dataset.from_list(rows_by_split["train"][:4]).to_parquet(str(args.out_dir / "smoke_2.parquet"))
    write_contexts_jsonl(selected, args.out_dir / "contexts_index.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    prompt_rows = []
    for split, rows in rows_by_split.items():
        for row in rows[::2]:
            ids = list(
                apply_chat_template(
                    tokenizer, row["prompt"], tools=None, tokenize=True, add_generation_prompt=True
                )
            )
            prompt_rows.append(
                {
                    "sample_id": row["extra_info"]["sample_id"],
                    "split": split,
                    "canonical_prompt_sha256": prompt_digest(ids),
                    "canonical_prompt_len": len(ids),
                    "canonical_prompt_ids": ids,
                }
            )
    (args.out_dir / "prompt_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prompt_rows)
    )

    model_files = [args.model / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")]
    frozen_artifacts = [
        args.out_dir / "train.parquet",
        args.out_dir / "validation.parquet",
        args.out_dir / "test.parquet",
        args.out_dir / "smoke_2.parquet",
        args.out_dir / "contexts_index.jsonl",
        args.out_dir / "prompt_manifest.jsonl",
    ]
    manifest = {
        "gate": "CUR1_SPLIT_FREEZE_PASS",
        "seed": args.seed,
        "source_pool": str(args.pool),
        "source_pool_sha256": sha256_file(args.pool),
        "historical_id_count": len(excluded),
        "historical_exclusion_new_id_counts": exclusion_counts,
        "eligible_pool_count": len(eligible),
        "split_sizes": {key: len(value) for key, value in split_ids.items()},
        "split_id_sha256": {
            key: hashlib.sha256("\n".join(value).encode()).hexdigest() for key, value in split_ids.items()
        },
        "all_selected_id_sha256": hashlib.sha256(
            "\n".join(split_ids["train"] + split_ids["validation"] + split_ids["test"]).encode()
        ).hexdigest(),
        "model_path": str(args.model),
        "model_contract_sha256": {path.name: sha256_file(path) for path in model_files},
        "frozen_artifact_sha256": {
            str(path.relative_to(args.out_dir)): sha256_file(path) for path in frozen_artifacts
        },
        "rollout_plan": {
            "train": {"questions": args.n_train, "rollouts_per_arm": 1, "trajectories": args.n_train * 2},
            "validation": {"questions": args.n_val, "rollouts_per_arm": 4, "trajectories": args.n_val * 8},
            "test": {"questions": args.n_test, "rollouts_per_arm": 8, "trajectories": args.n_test * 16},
            "total_new_trajectories": 4352,
        },
        "test_sealed": True,
        "boundary_v1_used": False,
        "cur0_ids_excluded": True,
        "question_ids": split_ids,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for split, ids in split_ids.items():
        (args.out_dir / f"{split}_ids.txt").write_text("\n".join(ids) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"question_ids", "historical_exclusion_new_id_counts"}}, indent=2))


if __name__ == "__main__":
    main()
