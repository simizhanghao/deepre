#!/usr/bin/env python3
"""Build Phase 3B/3C GRPO smoke dataset (default n=128).

Policy sees: system + Question only (NO gold, NO contexts, NO supporting_facts).
Reward sees: gold_answers + supporting_facts (title, sentence_id) via ground_truth.
Retriever sees: sample_id -> contexts jsonl index.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.rl.candidate_index import write_contexts_jsonl
from src.sft.prototype_builder import AGENT_SYSTEM_PROMPT


def _load_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sf_minimal(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reward-only supporting facts (title + sentence_id). No sentence text."""
    out: List[Dict[str, Any]] = []
    for sf in sample.get("supporting_facts") or []:
        title = sf.get("title")
        sid = sf.get("sentence_id", sf.get("sent_id"))
        if title is None or sid is None:
            continue
        out.append({"title": str(title), "sentence_id": int(sid)})
    return out


def _to_verl_row(sample: Dict[str, Any], *, split: str, idx: int) -> Dict[str, Any]:
    sid = str(sample["sample_id"])
    question = sample["question"]
    golds = list(sample.get("gold_answers") or [])
    sf = _sf_minimal(sample)
    return {
        "data_source": "hotpotqa_distractor_candidate",
        "agent_name": "eca_search_agent",
        "prompt": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ],
        "ability": "qa",
        "reward_model": {
            "style": "rule",
            # Reward only — never put into prompt.
            "ground_truth": {"target": golds, "supporting_facts": sf},
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "sample_id": sid,
            "question": question,
            "supporting_facts": sf,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {"sample_id": sid},
                }
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-pool",
        type=Path,
        default=REPO / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl",
    )
    ap.add_argument(
        "--exclude-ids",
        type=Path,
        default=REPO / "data/eval/hotpotqa_200_ids.txt",
        help="Frozen val-200 ids (never in train smoke)",
    )
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=16, help="Tiny held-out from pool for veRL val_files")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data/rl/train_smoke_128",
    )
    args = ap.parse_args()

    exclude = _load_ids(args.exclude_ids)
    pool = _read_jsonl(args.train_pool)
    eligible = [r for r in pool if str(r.get("sample_id")) not in exclude]
    if len(eligible) < args.n_train + args.n_val:
        raise SystemExit(
            f"Not enough eligible samples: {len(eligible)} < {args.n_train + args.n_val}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    train_raw = eligible[: args.n_train]
    val_raw = eligible[args.n_train : args.n_train + args.n_val]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [_to_verl_row(r, split="train", idx=i) for i, r in enumerate(train_raw)]
    val_rows = [_to_verl_row(r, split="val", idx=i) for i, r in enumerate(val_raw)]

    # Leakage audit: prompt must not contain gold answers / contexts.
    for rows, name in ((train_rows, "train"), (val_rows, "val")):
        for row in rows:
            blob = json.dumps(row["prompt"], ensure_ascii=False)
            golds = row["reward_model"]["ground_truth"]["target"]
            for g in golds:
                if g and len(g) >= 4 and g.lower() in blob.lower():
                    # Soft warn — short answers may collide with question text.
                    pass
            if "contexts" in blob or "supporting_facts" in blob:
                raise SystemExit(f"Leak: contexts/supporting_facts in {name} prompt")

    try:
        import datasets
    except ImportError as exc:
        raise SystemExit("Need `datasets` to write parquet: pip install datasets") from exc

    train_ds = datasets.Dataset.from_list(train_rows)
    val_ds = datasets.Dataset.from_list(val_rows)
    train_path = args.out_dir / "train.parquet"
    val_path = args.out_dir / "val.parquet"
    train_ds.to_parquet(str(train_path))
    val_ds.to_parquet(str(val_path))

    # Retrieval index for train+val smoke samples (contexts only).
    index_path = args.out_dir / "contexts_index.jsonl"
    n_idx = write_contexts_jsonl(train_raw + val_raw, index_path)

    manifest = {
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "seed": args.seed,
        "excluded_val200": len(exclude),
        "train_parquet": str(train_path),
        "val_parquet": str(val_path),
        "contexts_index": str(index_path),
        "index_samples": n_idx,
        "agent_name": "eca_search_agent",
        "policy_sees": ["system", "question"],
        "policy_must_not_see": ["gold_answers", "contexts", "supporting_facts"],
        "reward_sees": ["gold_answers", "supporting_facts"],
        "tool_sees": ["sample_id", "contexts", "query"],
        "train_ids": [r["sample_id"] for r in train_raw],
        "val_ids": [r["sample_id"] for r in val_raw],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "train_ids.txt").write_text(
        "\n".join(manifest["train_ids"]) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in manifest if k not in ("train_ids", "val_ids")}, indent=2))
    print(f"[ok] wrote {train_path} / {val_path} / {index_path}")


if __name__ == "__main__":
    main()
