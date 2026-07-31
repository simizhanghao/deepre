"""Phase 0.2B RAG smoke: fixed retrieval + real model through the trace pipeline.

Chain under test:
    question -> tiny built-in corpus -> deterministic word-overlap retrieval
    -> Document[] -> observation step (loss_mask=False)
    -> context + question fed to Qwen2.5-3B-Instruct -> answer step
    -> TraceRecord -> validate_trace_record -> basic_metrics
    -> results/{run_name}/ (trace.jsonl + metrics.json + run_info.json)

Fixed RAG: the system retrieves before generation; the model issues no search
action, so the trace is [observation -> answer] with NO search step.
Expected: search_count=0 but retrieved_document_count>0 (not a contradiction).

Usage (repo root, `deepresearch` conda env):
    CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python scripts/smoke_rag.py --max-samples 1 --debug
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import basic_metrics, normalize_answer
from src.eval.trace_schema import (
    CostInfo,
    Document,
    TraceRecord,
    TraceStep,
    validate_trace_record,
)

DEFAULT_MODEL_DIR = "/data1/hcc/.hf_home/Qwen2.5-3B-Instruct"

# Tiny fixed corpus: interface validation only, not a real retrieval algorithm.
CORPUS: List[Dict[str, str]] = [
    {
        "document_id": "doc_001",
        "title": "James Madison",
        "text": "James Madison served as the fourth president of the United States from 1809 to 1817.",
    },
    {
        "document_id": "doc_002",
        "title": "Thomas Jefferson",
        "text": "Thomas Jefferson served as president from 1801 to 1809.",
    },
    {
        "document_id": "doc_003",
        "title": "James Monroe",
        "text": "James Monroe served as president from 1817 to 1825.",
    },
]

DEBUG_SAMPLES: List[Dict[str, Any]] = [
    {
        "sample_id": "smoke_rag_q0",
        "question": "Who was president of the United States in 1812?",
        "gold_answers": ["James Madison"],
    },
]

SYSTEM_PROMPT = (
    "You are a question answering assistant. Answer the question using ONLY "
    "the provided context. Give a short answer only, no explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal fixed-RAG smoke run.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional JSON config with a 'samples' list.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results"))
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    return parser.parse_args()


def load_samples(config_path: str) -> List[Dict[str, Any]]:
    if config_path is None:
        return DEBUG_SAMPLES
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return config["samples"]


def retrieve(question: str, top_k: int) -> List[Document]:
    """Deterministic word-overlap retrieval over the built-in corpus."""
    question_tokens = set(normalize_answer(question).split())
    scored = []
    for doc in CORPUS:
        doc_tokens = set(normalize_answer(doc["title"] + " " + doc["text"]).split())
        scored.append((len(question_tokens & doc_tokens), doc))
    # Stable sort: ties keep corpus order, so results are fully deterministic.
    scored.sort(key=lambda pair: -pair[0])
    return [
        Document(
            document_id=doc["document_id"],
            title=doc["title"],
            text=doc["text"],
            source="builtin_corpus",
            rank=rank,
            score=float(overlap),
        )
        for rank, (overlap, doc) in enumerate(scored[:top_k])
    ]


def format_observation(documents: List[Document]) -> str:
    lines = [
        f"[{doc.document_id}] {doc.title}: {doc.text}"
        for doc in documents
    ]
    return "\n".join(lines)


def git_commit_short() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"smoke_rag_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke_rag] device={device} model={args.model_path}")
    print(f"[smoke_rag] run_dir={run_dir}")

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    model_load_s = time.perf_counter() - load_start
    print(f"[smoke_rag] model loaded in {model_load_s:.1f}s")

    samples = load_samples(args.config)[: args.max_samples]
    all_metrics: List[Dict[str, Any]] = []

    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as trace_file:
        for sample in samples:
            documents = retrieve(sample["question"], args.top_k)
            observation_text = format_observation(documents)
            observation_tokens = len(tokenizer(observation_text)["input_ids"])

            if args.debug:
                print(f"[debug] retrieved: "
                      f"{[(d.document_id, d.rank, d.score) for d in documents]}")
                print(f"[debug] observation:\n{observation_text}")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Context:\n{observation_text}\n\n"
                    f"Question: {sample['question']}"
                )},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            prompt_tokens = inputs["input_ids"].shape[-1]

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

            if args.debug:
                raw_output = tokenizer.decode(new_token_ids, skip_special_tokens=False)
                print(f"[debug] prompt_tokens={prompt_tokens}")
                print(f"[debug] raw_output={raw_output!r}")
                print(f"[debug] answer_text={answer_text!r}")

            record = TraceRecord(
                question=sample["question"],
                gold_answers=list(sample["gold_answers"]),
                sample_id=sample["sample_id"],
                trace_id=f"{sample['sample_id']}_rag_0",
                steps=[
                    TraceStep(
                        step_id=0,
                        step_type="observation",
                        content=observation_text,
                        loss_mask=False,
                        document_ids=[doc.document_id for doc in documents],
                    ),
                    TraceStep(
                        step_id=1,
                        step_type="answer",
                        content=answer_text,
                        loss_mask=True,
                    ),
                ],
                documents=documents,
                cost_info=CostInfo(
                    retrieved_document_count=len(documents),
                    prompt_tokens=prompt_tokens,
                    generated_tokens=int(new_token_ids.shape[-1]),
                    observation_tokens=observation_tokens,
                    latency_ms=latency_ms,
                ),
                metadata={
                    "method": "rag_fixed",
                    "model_path": args.model_path,
                    "retrieval": {
                        "type": "builtin_word_overlap",
                        "top_k": args.top_k,
                        "corpus_size": len(CORPUS),
                    },
                    "generation": {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False,
                        "seed": args.seed,
                    },
                },
            )

            validation_errors = validate_trace_record(record)
            metrics = basic_metrics(record)

            trace_file.write(
                json.dumps(record.to_jsonl_dict(), ensure_ascii=False) + "\n"
            )
            all_metrics.append({
                "trace_id": record.trace_id,
                "sample_id": record.sample_id,
                "validation_errors": validation_errors,
                "metrics": metrics,
                "retrieved_document_count": len(documents),
                "latency_ms": round(latency_ms, 1),
            })

            print(f"[smoke_rag] {record.trace_id}")
            print(f"  question:          {sample['question']}")
            print(f"  retrieved docs:    {[doc.document_id for doc in documents]}")
            print(f"  model answer:      {answer_text}")
            print(f"  gold answers:      {sample['gold_answers']}")
            print(f"  validation_errors: {validation_errors}")
            print(f"  metrics:           {metrics}")

    (run_dir / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    import transformers

    run_info = {
        "run_name": run_name,
        "git_commit": git_commit_short(),
        "args": vars(args),
        "device": device,
        "model_load_seconds": round(model_load_s, 1),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "num_samples": len(samples),
    }
    (run_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[smoke_rag] artifacts written to {run_dir}")


if __name__ == "__main__":
    main()