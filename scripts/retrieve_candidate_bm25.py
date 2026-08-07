"""Phase 1E: Candidate-BM25 retrieval over HotpotQA per-sample contexts.

For each eval sample, BM25s ranks ONLY that sample's candidate contexts
(diagnostic retrieval; NOT full-corpus Wiki search). Writes:

  results/{run}/retrieval_results.jsonl
  results/{run}/retrieval_metrics.json
  results/{run}/run_info.json

No GPU. Requires: pip install bm25s  (use: python -m pip install bm25s)

Usage:
  python scripts/retrieve_candidate_bm25.py \
    --eval-file data/eval/hotpotqa_50.jsonl \
    --max-samples 50 --top-k 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Candidate-BM25s over HotpotQA distractor contexts."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results"),
    )
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--eval-file",
        type=str,
        required=True,
        help="HotpotQA eval JSONL",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_eval_jsonl(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    return rows


def gold_titles(sample: Dict[str, Any]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for sf in sample.get("supporting_facts") or []:
        t = sf["title"]
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def title_recall_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    hit = sum(1 for t in gold if t in top)
    return hit / len(gold)


def all_gold_in_topk(gold: Sequence[str], retrieved: Sequence[str], k: int) -> bool:
    if not gold:
        return False
    top = set(retrieved[:k])
    return all(t in top for t in gold)


def retrieve_one(
    sample: Dict[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    try:
        import bm25s
    except ImportError as exc:
        raise SystemExit(
            "Missing bm25s. Install with:\n"
            "  python -m pip install bm25s\n"
            "Then re-run."
        ) from exc

    contexts = sample.get("contexts") or []
    if not contexts:
        return {
            "sample_id": sample["sample_id"],
            "query": sample["question"],
            "retriever": {
                "name": "bm25s",
                "scope": "candidate",
                "top_k": top_k,
                "config": {"stopwords": "en"},
            },
            "documents": [],
        }

    # Index title + text so title tokens help ranking.
    corpus_texts = [f"{c['title']} {c['text']}" for c in contexts]
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    k = min(top_k, len(contexts))
    query = sample["question"]
    query_tokens = bm25s.tokenize(query, stopwords="en")
    # results: doc indices (when corpus not passed as return docs)
    results, scores = retriever.retrieve(query_tokens, k=k)

    documents: List[Dict[str, Any]] = []
    for rank_i in range(results.shape[1]):
        idx = int(results[0, rank_i])
        score = float(scores[0, rank_i])
        ctx = contexts[idx]
        documents.append(
            {
                "document_id": ctx["document_id"],
                "title": ctx["title"],
                "text": ctx["text"],
                "rank": rank_i + 1,
                "score": score,
                "metadata": {"sentences": list(ctx.get("sentences") or [])},
            }
        )

    return {
        "sample_id": sample["sample_id"],
        "query": query,
        "retriever": {
            "name": "bm25s",
            "scope": "candidate",
            "top_k": top_k,
            "config": {"stopwords": "en", "indexed_field": "title+text"},
        },
        "documents": documents,
    }


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_file)
    if not eval_path.is_absolute():
        eval_path = REPO_ROOT / eval_path

    samples = load_eval_jsonl(eval_path, args.max_samples)
    if not samples:
        raise SystemExit(f"no samples in {eval_path}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"retrieval_candidate_bm25_n{len(samples)}_{stamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[candidate_bm25] eval={eval_path} n={len(samples)} top_k={args.top_k}")
    print(f"[candidate_bm25] run_dir={run_dir}")

    cache_path = run_dir / "retrieval_results.jsonl"
    per_sample_metrics: List[Dict[str, Any]] = []
    ks = (1, 3, 5)

    t0 = time.perf_counter()
    with cache_path.open("w", encoding="utf-8") as out:
        for i, sample in enumerate(samples):
            packed = retrieve_one(sample, args.top_k)
            out.write(json.dumps(packed, ensure_ascii=False) + "\n")

            gold = gold_titles(sample)
            retrieved_titles = [d["title"] for d in packed["documents"]]
            row_m = {
                "sample_id": sample["sample_id"],
                "n_gold_titles": len(gold),
                "n_retrieved": len(retrieved_titles),
                "gold_titles": gold,
                "retrieved_titles": retrieved_titles,
            }
            for k in ks:
                row_m[f"title_recall@{k}"] = title_recall_at_k(
                    gold, retrieved_titles, k
                )
                row_m[f"title_hit_all@{k}"] = all_gold_in_topk(
                    gold, retrieved_titles, k
                )
            per_sample_metrics.append(row_m)

            if args.debug:
                print(
                    f"[debug] {i+1}/{len(samples)} {sample['sample_id']} "
                    f"R@5={row_m['title_recall@5']:.2f} "
                    f"hit_all@5={row_m['title_hit_all@5']} "
                    f"top={retrieved_titles[:3]}"
                )
            elif (i + 1) % 10 == 0 or (i + 1) == len(samples):
                print(f"[candidate_bm25] {i+1}/{len(samples)}")

    elapsed = time.perf_counter() - t0
    n = len(per_sample_metrics)
    summary = {
        "num_samples": n,
        "top_k": args.top_k,
        "scope": "candidate",
        "retriever": "bm25s",
        "elapsed_seconds": round(elapsed, 2),
    }
    for k in ks:
        summary[f"mean_title_recall@{k}"] = round(
            sum(r[f"title_recall@{k}"] for r in per_sample_metrics) / n, 4
        )
        summary[f"title_hit_all_rate@{k}"] = round(
            sum(1 for r in per_sample_metrics if r[f"title_hit_all@{k}"]) / n, 4
        )

    (run_dir / "retrieval_metrics.json").write_text(
        json.dumps(per_sample_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "eval_file": str(eval_path),
                "args": vars(args),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[candidate_bm25] cache -> {cache_path}")


if __name__ == "__main__":
    main()
