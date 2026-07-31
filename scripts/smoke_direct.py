"""Phase 0.2A Direct smoke: one real model answer through the full trace pipeline.

Chain under test:
    question -> local Qwen2.5-3B-Instruct generation -> answer TraceStep
    -> TraceRecord -> validate_trace_record -> basic_metrics
    -> results/{run_name}/ (trace.jsonl + metrics.json + run_info.json)

Direct baseline only: no search, no observation, no evidence, no reward, no agent loop.

Usage (run from repo root, in the `deepresearch` conda env):
    CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python scripts/smoke_direct.py --max-samples 1 --debug
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

from src.eval.metrics import basic_metrics
from src.eval.trace_schema import (
    CostInfo,
    TraceRecord,
    TraceStep,
    validate_trace_record,
)

DEFAULT_MODEL_DIR = "/data1/hcc/.hf_home/Qwen2.5-3B-Instruct"

# Built-in handcrafted sample, same as the Phase 0.1 acceptance case.
DEBUG_SAMPLES: List[Dict[str, Any]] = [
    {
        "sample_id": "smoke_direct_q0",
        "question": "Who was president of the United States in 1812?",
        "gold_answers": ["James Madison"],
    },
]

SYSTEM_PROMPT = (
    "You are a question answering assistant. "
    "Answer the question with a short answer only, no explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Direct-answer smoke run.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional JSON config with a 'samples' list; "
                             "defaults to the built-in debug sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results"))
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--debug", action="store_true",
                        help="Print raw model output and extra details.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def load_samples(config_path: str) -> List[Dict[str, Any]]:
    if config_path is None:
        return DEBUG_SAMPLES
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return config["samples"]


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
    run_name = f"smoke_direct_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke_direct] device={device} model={args.model_path}")
    print(f"[smoke_direct] run_dir={run_dir}")

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    model_load_s = time.perf_counter() - load_start
    print(f"[smoke_direct] model loaded in {model_load_s:.1f}s")

    samples = load_samples(args.config)[: args.max_samples]
    all_metrics: List[Dict[str, Any]] = []

    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as trace_file:
        for sample in samples:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["question"]},
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
            answer_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

            if args.debug:
                raw_output = tokenizer.decode(new_token_ids, skip_special_tokens=False)
                print(f"[debug] prompt_tokens={prompt_tokens}")
                print(f"[debug] raw_output={raw_output!r}")
                print(f"[debug] answer_text={answer_text!r}")

            record = TraceRecord(
                question=sample["question"],
                gold_answers=list(sample["gold_answers"]),
                sample_id=sample["sample_id"],
                trace_id=f"{sample['sample_id']}_direct_0",
                steps=[
                    TraceStep(
                        step_id=0,
                        step_type="answer",
                        content=answer_text,
                        loss_mask=True,
                    ),
                ],
                cost_info=CostInfo(
                    prompt_tokens=prompt_tokens,
                    generated_tokens=int(new_token_ids.shape[-1]),
                    latency_ms=latency_ms,
                ),
                metadata={
                    "method": "direct",
                    "model_path": args.model_path,
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
                "latency_ms": round(latency_ms, 1),
            })

            print(f"[smoke_direct] {record.trace_id}")
            print(f"  question:          {sample['question']}")
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

    print(f"[smoke_direct] artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
