"""Phase 2C: prepare HotpotQA distractor/train pool for cold-start SFT.

Writes a seed-permuted train JSONL and asserts ZERO overlap with the frozen
Phase-1 validation-200 eval set (by sample_id and raw_id).

Usage (repo root, deepresearch env; may need HF network / mirror):
    export HF_ENDPOINT=https://hf-mirror.com   # if needed
    python scripts/prepare_hotpotqa_train.py --seed 42 --max-samples 8000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prepare_hotpotqa as ph  # noqa: E402

HF_CONFIG = ph.HF_CONFIG
HF_DATASET = ph.HF_DATASET
convert_row = ph.convert_row
load_hotpotqa = ph.load_hotpotqa
permute_indices = ph.permute_indices
validate_sample = ph.validate_sample
write_id_list = ph.write_id_list
write_jsonl = ph.write_jsonl

DEFAULT_FROZEN_VAL_IDS = REPO_ROOT / "data" / "eval" / "hotpotqa_200_ids.txt"
DEFAULT_OUT = REPO_ROOT / "data" / "sft" / "source"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare HotpotQA train pool for Phase 2C.")
    p.add_argument("--config", type=str, default=HF_CONFIG)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=8000)
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument(
        "--frozen-val-ids",
        type=str,
        default=str(DEFAULT_FROZEN_VAL_IDS),
        help="Phase-1 frozen validation sample_id list (must not overlap).",
    )
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args()


def load_id_set(path: Path) -> Set[str]:
    if not path.is_file():
        raise SystemExit(f"frozen val id file not found: {path}")
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def raw_id_from_sample_id(sample_id: str) -> str:
    # hotpotqa_{config}_{split}_{raw_id}
    parts = sample_id.split("_", 3)
    if len(parts) < 4:
        return sample_id
    return parts[3]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    frozen_path = Path(args.frozen_val_ids)
    if not frozen_path.is_absolute():
        frozen_path = REPO_ROOT / frozen_path

    frozen_ids = load_id_set(frozen_path)
    frozen_raw = {raw_id_from_sample_id(x) for x in frozen_ids}

    print(f"Loading {HF_DATASET} config={args.config!r} split='train' ...")
    rows = load_hotpotqa(args.config, "train", args.cache_dir)
    print(f"Loaded {len(rows)} train rows.")

    order = permute_indices(len(rows), args.seed)
    need = min(args.max_samples, len(rows))

    converted: List[Dict[str, Any]] = []
    skipped: List[str] = []
    seen: Set[str] = set()
    # HotpotQA train has rare broken supporting_facts (sentence_id OOR).
    # Skip them and keep walking the seeded permutation until we fill `need`.
    for idx in order:
        if len(converted) >= need:
            break
        sample = convert_row(rows[idx], config=args.config, split="train")
        sid = sample["sample_id"]
        if sid in seen:
            skipped.append(f"duplicate sample_id: {sid}")
            continue
        errs = validate_sample(sample)
        if errs:
            skipped.extend(errs)
            continue
        seen.add(sid)
        converted.append(sample)

    if len(converted) < need:
        raise SystemExit(
            f"only collected {len(converted)} valid train samples "
            f"(need {need}); skipped={len(skipped)}"
        )

    # Leakage gate
    overlap_sid = sorted(seen & frozen_ids)
    overlap_raw = sorted(
        {
            (sample.get("metadata") or {}).get("raw_id", "")
            for sample in converted
        }
        & frozen_raw
    )
    if overlap_sid or overlap_raw:
        raise SystemExit(
            f"LEAKAGE: overlap with frozen validation-200 "
            f"(sample_id={len(overlap_sid)}, raw_id={len(overlap_raw)}). "
            f"Examples sid={overlap_sid[:3]} raw={overlap_raw[:3]}"
        )

    if skipped:
        print(f"Skipped {len(skipped)} invalid/duplicate row(s); examples:")
        for e in skipped[:10]:
            print(f"  - {e}")

    stem = f"hotpotqa_distractor_train_pool_n{len(converted)}"
    jsonl_path = out_dir / f"{stem}.jsonl"
    ids_path = out_dir / f"{stem}_ids.txt"
    manifest_path = out_dir / f"{stem}_manifest.json"

    write_jsonl(jsonl_path, converted)
    write_id_list(ids_path, [s["sample_id"] for s in converted])
    manifest = {
        "dataset": HF_DATASET,
        "config": args.config,
        "split": "train",
        "seed": args.seed,
        "n_raw_total": len(rows),
        "n_pool": len(converted),
        "n_skipped_invalid": len(skipped),
        "frozen_val_ids_file": str(frozen_path),
        "n_frozen_val_ids": len(frozen_ids),
        "overlap_sample_id": 0,
        "overlap_raw_id": 0,
        "output_jsonl": str(jsonl_path),
        "phase": "2C",
        "note": "Train pool only; never merge Phase-1 validation-200 into SFT.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"wrote {jsonl_path} n={len(converted)}")
    print(f"wrote {ids_path}")
    print(f"wrote {manifest_path}")
    print("overlap with frozen validation-200: 0 (pass)")


if __name__ == "__main__":
    main()
