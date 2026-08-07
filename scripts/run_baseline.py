"""Phase 1C/D/E: unified Direct / Oracle / Candidate-BM25 RAG on HotpotQA.

Same data, model, tokenizer, system-prompt style, generation config, TraceRecord,
and metrics. Only the injected context differs:

    --method direct          : question only
    --method oracle          : gold supporting documents
    --method candidate_bm25  : docs from retrieval_results.jsonl cache

Usage (repo root, deepresearch env):
    CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python scripts/run_baseline.py \
        --method direct \
        --eval-file data/eval/hotpotqa_8.jsonl \
        --max-samples 8 --seed 42

    CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python scripts/run_baseline.py \
        --method candidate_bm25 \
        --eval-file data/eval/hotpotqa_50.jsonl \
        --retrieval-cache results/.../retrieval_results.jsonl \
        --max-samples 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import basic_metrics
from src.eval.trace_schema import (
    CostInfo,
    Document,
    TraceRecord,
    TraceStep,
    validate_trace_record,
)

DEFAULT_MODEL_DIR = "/data1/hcc/.hf_home/Qwen2.5-3B-Instruct"
METHODS = ("direct", "oracle", "candidate_bm25")
RAG_METHODS = frozenset({"oracle", "candidate_bm25"})

DIRECT_SYSTEM_PROMPT = (
    "You are a question answering assistant. "
    "Answer the question with a short answer only, no explanation."
)
RAG_SYSTEM_PROMPT = (
    "You are a question answering assistant. Answer the question using ONLY "
    "the provided context. Give a short answer only, no explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HotpotQA Direct / Oracle baseline runner."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON overlay; unused fields ignored. Prefer --eval-file.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=METHODS,
        help="Baseline method: direct | oracle | candidate_bm25.",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        required=True,
        help="Path to HotpotQA eval JSONL (e.g. data/eval/hotpotqa_8.jsonl).",
    )
    parser.add_argument(
        "--retrieval-cache",
        type=str,
        default=None,
        help="retrieval_results.jsonl (required for candidate_bm25).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results"),
    )
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag appended to run_name.",
    )
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


def load_retrieval_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["sample_id"]] = row
    return out


def documents_from_cache(cache_row: Dict[str, Any]) -> List[Document]:
    docs: List[Document] = []
    for d in cache_row.get("documents") or []:
        docs.append(
            Document(
                document_id=d["document_id"],
                title=d.get("title", ""),
                text=d.get("text", ""),
                source="candidate_bm25",
                rank=d.get("rank"),
                score=d.get("score"),
                metadata=dict(d.get("metadata") or {}),
            )
        )
    return docs


def git_commit_short() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def oracle_documents(sample: Dict[str, Any]) -> List[Document]:
    """Full docs for unique supporting_facts titles, first-seen order."""
    contexts = sample.get("contexts") or []
    title_to_ctx = {c["title"]: c for c in contexts}
    docs: List[Document] = []
    seen = set()
    rank = 1
    for sf in sample.get("supporting_facts") or []:
        title = sf["title"]
        if title in seen:
            continue
        seen.add(title)
        ctx = title_to_ctx.get(title)
        if ctx is None:
            raise ValueError(
                f"{sample['sample_id']}: oracle title missing in contexts: {title!r}"
            )
        docs.append(
            Document(
                document_id=ctx["document_id"],
                title=ctx["title"],
                text=ctx["text"],
                source="oracle_supporting_docs",
                rank=rank,
                score=None,
                metadata={"sentences": list(ctx.get("sentences") or [])},
            )
        )
        rank += 1
    return docs


def format_observation(documents: Sequence[Document]) -> str:
    return "\n".join(
        f"[{doc.document_id}] {doc.title}: {doc.text}" for doc in documents
    )


def build_messages(
    method: str,
    question: str,
    observation_text: Optional[str],
) -> List[Dict[str, str]]:
    if method == "direct":
        return [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    assert observation_text is not None
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{observation_text}\n\nQuestion: {question}",
        },
    ]


def retrieval_metadata(
    method: str,
    cache_row: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if method == "oracle":
        return {
            "name": "oracle",
            "scope": "oracle_supporting_docs",
            "top_k": None,
        }
    if method == "candidate_bm25":
        meta = {
            "name": "bm25s",
            "scope": "candidate",
            "top_k": None,
        }
        if cache_row and isinstance(cache_row.get("retriever"), dict):
            meta.update(cache_row["retriever"])
        return meta
    return None


def aggregate_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"num_samples": 0}

    def mean(key: str) -> float:
        return sum(r["metrics"][key] for r in rows) / n

    def mean_field(key: str) -> float:
        return sum(r[key] for r in rows) / n

    format_ok = sum(1 for r in rows if r["metrics"]["format_valid"]) / n
    return {
        "num_samples": n,
        "mean_em": round(mean("exact_match"), 4),
        "mean_token_f1": round(mean("token_f1"), 4),
        "format_valid_rate": round(format_ok, 4),
        "mean_prompt_tokens": round(mean_field("prompt_tokens"), 1),
        "mean_observation_tokens": round(mean_field("observation_tokens"), 1),
        "mean_generated_tokens": round(mean_field("generated_tokens"), 1),
        "mean_latency_ms": round(mean_field("latency_ms"), 1),
        "mean_retrieved_document_count": round(
            mean_field("retrieved_document_count"), 2
        ),
    }


def main() -> None:
    args = parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    eval_path = Path(args.eval_file)
    if not eval_path.is_absolute():
        eval_path = REPO_ROOT / eval_path
    if not eval_path.is_file():
        raise SystemExit(f"eval file not found: {eval_path}")

    samples = load_eval_jsonl(eval_path, args.max_samples)
    if not samples:
        raise SystemExit(f"no samples loaded from {eval_path}")

    retrieval_cache: Dict[str, Dict[str, Any]] = {}
    cache_path: Optional[Path] = None
    if args.method == "candidate_bm25":
        if not args.retrieval_cache:
            raise SystemExit(
                "--retrieval-cache is required for --method candidate_bm25"
            )
        cache_path = Path(args.retrieval_cache)
        if not cache_path.is_absolute():
            cache_path = REPO_ROOT / cache_path
        if not cache_path.is_file():
            raise SystemExit(f"retrieval cache not found: {cache_path}")
        retrieval_cache = load_retrieval_cache(cache_path)
        missing = [s["sample_id"] for s in samples if s["sample_id"] not in retrieval_cache]
        if missing:
            raise SystemExit(
                f"retrieval cache missing {len(missing)} sample_ids "
                f"(e.g. {missing[0]})"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.run_tag}" if args.run_tag else ""
    run_name = f"baseline_{args.method}_n{len(samples)}_{stamp}{tag}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_baseline] method={args.method} device={device}")
    print(f"[run_baseline] eval_file={eval_path} n={len(samples)}")
    if cache_path is not None:
        print(f"[run_baseline] retrieval_cache={cache_path}")
    print(f"[run_baseline] model={args.model_path}")
    print(f"[run_baseline] run_dir={run_dir}")

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    model_load_s = time.perf_counter() - load_start
    print(f"[run_baseline] model loaded in {model_load_s:.1f}s")

    all_metrics: List[Dict[str, Any]] = []

    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as trace_file:
        for i, sample in enumerate(samples):
            sample_id = sample["sample_id"]
            documents: List[Document] = []
            observation_text: Optional[str] = None
            observation_tokens = 0
            cache_row: Optional[Dict[str, Any]] = None

            if args.method == "oracle":
                documents = oracle_documents(sample)
            elif args.method == "candidate_bm25":
                cache_row = retrieval_cache[sample_id]
                documents = documents_from_cache(cache_row)

            if args.method in RAG_METHODS:
                observation_text = format_observation(documents)
                observation_tokens = len(
                    tokenizer(observation_text, add_special_tokens=False)["input_ids"]
                )

            messages = build_messages(
                args.method, sample["question"], observation_text
            )
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            prompt_tokens = int(inputs["input_ids"].shape[-1])

            gen_start = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            latency_ms = (time.perf_counter() - gen_start) * 1000.0

            new_token_ids = output_ids[0, prompt_tokens:]
            answer_text = tokenizer.decode(
                new_token_ids, skip_special_tokens=True
            ).strip()
            generated_tokens = int(new_token_ids.shape[-1])

            if args.debug:
                print(f"[debug] [{i+1}/{len(samples)}] {sample_id}")
                if documents:
                    print(
                        f"[debug] docs="
                        f"{[(d.title, d.rank, d.score) for d in documents]}"
                    )
                print(f"[debug] answer={answer_text!r} gold={sample['gold_answers']}")

            steps: List[TraceStep] = []
            if args.method in RAG_METHODS:
                steps.append(
                    TraceStep(
                        step_id=0,
                        step_type="observation",
                        content=observation_text or "",
                        loss_mask=False,
                        document_ids=[d.document_id for d in documents],
                    )
                )
                steps.append(
                    TraceStep(
                        step_id=1,
                        step_type="answer",
                        content=answer_text,
                        loss_mask=True,
                    )
                )
            else:
                steps.append(
                    TraceStep(
                        step_id=0,
                        step_type="answer",
                        content=answer_text,
                        loss_mask=True,
                    )
                )

            record = TraceRecord(
                question=sample["question"],
                gold_answers=list(sample["gold_answers"]),
                sample_id=sample_id,
                trace_id=f"{sample_id}_{args.method}_0",
                steps=steps,
                documents=documents,
                cost_info=CostInfo(
                    retrieved_document_count=len(documents),
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    observation_tokens=observation_tokens,
                    latency_ms=latency_ms,
                ),
                metadata={
                    "method": args.method,
                    "model_path": args.model_path,
                    "eval_file": str(eval_path),
                    "retrieval_cache": str(cache_path) if cache_path else None,
                    "supporting_facts": sample.get("supporting_facts"),
                    "sample_metadata": sample.get("metadata"),
                    "generation": {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False,
                        "seed": args.seed,
                    },
                    "retrieval": retrieval_metadata(args.method, cache_row),
                },
            )

            validation_errors = validate_trace_record(record)
            metrics = basic_metrics(record)
            if validation_errors:
                metrics["format_valid"] = False

            trace_file.write(
                json.dumps(record.to_jsonl_dict(), ensure_ascii=False) + "\n"
            )
            row = {
                "trace_id": record.trace_id,
                "sample_id": sample_id,
                "validation_errors": validation_errors,
                "metrics": metrics,
                "prediction": answer_text,
                "gold_answers": list(sample["gold_answers"]),
                "prompt_tokens": prompt_tokens,
                "observation_tokens": observation_tokens,
                "generated_tokens": generated_tokens,
                "retrieved_document_count": len(documents),
                "latency_ms": round(latency_ms, 1),
            }
            all_metrics.append(row)

            print(
                f"[{args.method}] {i+1}/{len(samples)} {sample_id} "
                f"EM={metrics['exact_match']} F1={metrics['token_f1']:.2f} "
                f"lat={latency_ms:.0f}ms"
            )

    summary = aggregate_summary(all_metrics)
    (run_dir / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    run_info = {
        "run_name": run_name,
        "git_commit": git_commit_short(),
        "method": args.method,
        "args": vars(args),
        "eval_file": str(eval_path),
        "device": device,
        "model_load_seconds": round(model_load_s, 1),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "num_samples": len(samples),
        "summary": summary,
    }
    (run_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[run_baseline] summary={summary}")
    print(f"[run_baseline] artifacts -> {run_dir}")


if __name__ == "__main__":
    main()
