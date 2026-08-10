"""Offline smoke: load merged HF checkpoint and run one CUDA generate.

Usage (deepresearch env, after 2D3-A merge):
    CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python scripts/smoke_merged_model.py \
      --model-path outputs/00_sft_v1_merged
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-load a merged HF CausalLM.")
    p.add_argument(
        "--model-path",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "00_sft_v1_merged"
        ),
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.model_path).resolve()
    if not path.is_dir():
        print(f"[smoke] missing model dir: {path}", file=sys.stderr)
        return 1
    if not (path / "config.json").exists():
        print(f"[smoke] no config.json under {path}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("[smoke] CUDA not available", file=sys.stderr)
        return 1

    print(f"[smoke] loading tokenizer from {path}")
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    print(f"[smoke] loading model (bf16) from {path}")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map={"": 0},
    )
    print(f"[smoke] model loaded in {time.perf_counter() - t0:.1f}s")

    messages = [
        {
            "role": "system",
            "content": "You are a question answering assistant. Answer briefly.",
        },
        {"role": "user", "content": "What is the capital of France?"},
    ]
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = "What is the capital of France?\n"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    print("[smoke] generation OK")
    print(f"[smoke] output: {text!r}")
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
