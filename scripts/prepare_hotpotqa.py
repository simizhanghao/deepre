"""Prepare HotpotQA distractor/validation eval subsets (Phase 1B Fast Track).

Downloads official HF `hotpotqa/hotpot_qa` (config=distractor, split=validation),
converts each row to the ECA eval sample contract, then writes nested subsets:

    data/eval/hotpotqa_8.jsonl
    data/eval/hotpotqa_50.jsonl
    data/eval/hotpotqa_200.jsonl

Sampling: seed-fixed permutation of the full validation split, then prefixes
so that 8 ⊂ 50 ⊂ 200. sample_id uses the stable HotpotQA raw id.

Usage (repo root, deepresearch env):
    python scripts/prepare_hotpotqa.py --seed 42
    python scripts/prepare_hotpotqa.py --seed 42 --debug --max-samples 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "eval"
HF_DATASET = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"
HF_SPLIT = "validation"
SUBSET_SIZES = (8, 50, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HotpotQA distractor/validation into ECA eval JSONL."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=HF_CONFIG,
        help="HF dataset config name (default: distractor).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for hotpotqa_{8,50,200}.jsonl",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=max(SUBSET_SIZES),
        help="Max samples after permutation (default 200). "
        "Subset files larger than this are skipped.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print first converted sample and validation details.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional HF datasets cache dir.",
    )
    return parser.parse_args()


def join_sentences(sentences: Sequence[str]) -> str:
    return " ".join(sentences)


def build_sample_id(raw_id: str, config: str, split: str) -> str:
    return f"hotpotqa_{config}_{split}_{raw_id}"


def convert_row(row: Dict[str, Any], config: str, split: str) -> Dict[str, Any]:
    raw_id = str(row["id"])
    sample_id = build_sample_id(raw_id, config=config, split=split)

    context = row["context"]
    titles: Sequence[str] = list(context["title"])
    sentence_lists: Sequence[Sequence[str]] = list(context["sentences"])

    contexts: List[Dict[str, Any]] = []
    title_to_ctx: Dict[str, Dict[str, Any]] = {}
    for k, (title, sentences) in enumerate(zip(titles, sentence_lists)):
        sent_list = [str(s) for s in sentences]
        doc = {
            "document_id": f"{sample_id}_ctx_{k}",
            "title": str(title),
            "sentences": sent_list,
            "text": join_sentences(sent_list),
        }
        contexts.append(doc)
        # First occurrence wins if duplicate titles appear.
        title_to_ctx.setdefault(str(title), doc)

    sf = row["supporting_facts"]
    sf_titles: Sequence[str] = list(sf["title"])
    sf_sent_ids: Sequence[int] = list(sf["sent_id"])
    supporting_facts: List[Dict[str, Any]] = []
    for title, sent_id in zip(sf_titles, sf_sent_ids):
        title_s = str(title)
        sid = int(sent_id)
        item: Dict[str, Any] = {"title": title_s, "sentence_id": sid}
        ctx = title_to_ctx.get(title_s)
        if ctx is not None and 0 <= sid < len(ctx["sentences"]):
            item["sentence"] = ctx["sentences"][sid]
        supporting_facts.append(item)

    answer = row["answer"]
    if isinstance(answer, list):
        gold_answers = [str(a) for a in answer]
    else:
        gold_answers = [str(answer)]

    return {
        "sample_id": sample_id,
        "question": str(row["question"]),
        "gold_answers": gold_answers,
        "supporting_facts": supporting_facts,
        "contexts": contexts,
        "metadata": {
            "dataset": "hotpotqa",
            "split": split,
            "source_split": split,
            "source": f"hf:{HF_DATASET}",
            "config": config,
            "raw_id": raw_id,
            "level": str(row.get("level", "")),
            "type": str(row.get("type", "")),
        },
    }


def validate_sample(sample: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    sid = sample.get("sample_id")
    if not sid or not isinstance(sid, str):
        errors.append("sample_id missing or not str")
    if not sample.get("question"):
        errors.append(f"{sid}: empty question")
    gold = sample.get("gold_answers")
    if not isinstance(gold, list) or not gold or any(
        not isinstance(x, str) or not x for x in gold
    ):
        errors.append(f"{sid}: gold_answers must be non-empty List[str]")

    contexts = sample.get("contexts") or []
    doc_ids = [c.get("document_id") for c in contexts]
    if len(doc_ids) != len(set(doc_ids)):
        errors.append(f"{sid}: duplicate document_id in contexts")
    title_to_sents = {c["title"]: c["sentences"] for c in contexts}

    for c in contexts:
        joined = join_sentences(c.get("sentences") or [])
        if c.get("text") != joined:
            errors.append(
                f"{sid}: text != join(sentences) for {c.get('document_id')}"
            )

    for i, sf in enumerate(sample.get("supporting_facts") or []):
        title = sf.get("title")
        sent_id = sf.get("sentence_id")
        if title not in title_to_sents:
            errors.append(
                f"{sid}: supporting_facts[{i}] title not in contexts: {title!r}"
            )
            continue
        sents = title_to_sents[title]
        if not isinstance(sent_id, int) or sent_id < 0 or sent_id >= len(sents):
            errors.append(
                f"{sid}: supporting_facts[{i}] sentence_id={sent_id} "
                f"out of range for title={title!r} (n={len(sents)})"
            )
    return errors


def load_hotpotqa(
    config: str,
    split: str,
    cache_dir: str | None,
) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `datasets`. Install in deepresearch env:\n"
            "  pip install datasets\n"
            "Then re-run this script."
        ) from exc

    kwargs: Dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset(HF_DATASET, config, split=split, **kwargs)
    return [dict(row) for row in ds]


def permute_indices(n: int, seed: int) -> List[int]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    return idx


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_id_list(path: Path, sample_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sample_ids) + ("\n" if sample_ids else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    print(f"Loading {HF_DATASET} config={args.config!r} split={HF_SPLIT!r} ...")
    rows = load_hotpotqa(args.config, HF_SPLIT, args.cache_dir)
    n_total = len(rows)
    print(f"Loaded {n_total} raw rows.")

    order = permute_indices(n_total, args.seed)
    take_n = min(args.max_samples, n_total, max(SUBSET_SIZES))
    selected_raw = [rows[i] for i in order[:take_n]]

    converted: List[Dict[str, Any]] = []
    all_errors: List[str] = []
    seen_ids: set[str] = set()
    for row in selected_raw:
        sample = convert_row(row, config=args.config, split=HF_SPLIT)
        if sample["sample_id"] in seen_ids:
            all_errors.append(f"duplicate sample_id: {sample['sample_id']}")
        seen_ids.add(sample["sample_id"])
        errs = validate_sample(sample)
        all_errors.extend(errs)
        converted.append(sample)

    if all_errors:
        print(f"VALIDATION FAILED: {len(all_errors)} error(s)", file=sys.stderr)
        for e in all_errors[:30]:
            print(f"  - {e}", file=sys.stderr)
        if len(all_errors) > 30:
            print(f"  ... and {len(all_errors) - 30} more", file=sys.stderr)
        raise SystemExit(1)

    manifest = {
        "dataset": HF_DATASET,
        "config": args.config,
        "split": HF_SPLIT,
        "seed": args.seed,
        "n_raw_total": n_total,
        "n_converted": len(converted),
        "subset_sizes": [s for s in SUBSET_SIZES if s <= len(converted)],
        "sample_id_rule": "hotpotqa_{config}_{split}_{raw_id}",
        "nested_prefix": True,
        "output_dir": str(output_dir),
    }

    written: List[Tuple[int, Path]] = []
    for size in SUBSET_SIZES:
        if size > len(converted):
            print(f"Skip hotpotqa_{size}.jsonl (only {len(converted)} available)")
            continue
        subset = converted[:size]
        out_path = output_dir / f"hotpotqa_{size}.jsonl"
        write_jsonl(out_path, subset)
        write_id_list(
            output_dir / f"hotpotqa_{size}_ids.txt",
            [s["sample_id"] for s in subset],
        )
        written.append((size, out_path))
        print(f"Wrote {size} -> {out_path}")

    # Nesting check: ids of smaller are prefixes of larger.
    ids_by_size = {
        size: [s["sample_id"] for s in converted[:size]]
        for size in SUBSET_SIZES
        if size <= len(converted)
    }
    if 8 in ids_by_size and 50 in ids_by_size:
        assert ids_by_size[8] == ids_by_size[50][:8]
    if 50 in ids_by_size and 200 in ids_by_size:
        assert ids_by_size[50] == ids_by_size[200][:50]

    manifest_path = output_dir / "hotpotqa_prepare_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote manifest -> {manifest_path}")
    print("Validation: PASS")

    if args.debug and converted:
        first = converted[0]
        print("--- debug first sample ---")
        print(json.dumps(first, ensure_ascii=False, indent=2)[:4000])
        print(
            f"oracle titles: "
            f"{sorted({sf['title'] for sf in first['supporting_facts']})}"
        )
        print(f"n_contexts: {len(first['contexts'])}")


if __name__ == "__main__":
    main()
