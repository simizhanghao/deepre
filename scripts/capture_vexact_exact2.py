#!/usr/bin/env python3
"""Capture VeXact route-root logits and sampling for exact-contract calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model-path", default=str(REPO / "outputs/rl/03_hf_evidence_step400"))
    parser.add_argument(
        "--source-dump",
        default=str(REPO / "results/16_audit_routing_exploration/worker_mismatch/dump_pathC.jsonl"),
    )
    parser.add_argument(
        "--sample-manifest",
        default=str(REPO / "results/17_rollout_alignment/calibration/sample_ids_exact2.json"),
    )
    parser.add_argument("--attn-impl", default="triton-invariant")
    parser.add_argument("--n-rollouts", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_digest(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids).encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    if args.debug:
        if args.max_samples != 2 or args.n_rollouts != 0:
            raise SystemExit("exact smoke requires --max-samples 2 --n-rollouts 0")
    elif args.max_samples != 20 or args.n_rollouts != 16:
        raise SystemExit("Gate A1 requires --max-samples 20 --n-rollouts 16")
    if args.attn_impl != "triton-invariant":
        raise SystemExit("A100 exact smoke requires triton-invariant")

    config_path = Path(args.config).resolve()
    model_path = Path(args.model_path).resolve()
    source_path = Path(args.source_dump).resolve()
    manifest_path = Path(args.sample_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_root = (
        (REPO / "outputs").resolve()
        if args.debug
        else (REPO / "results/17_rollout_alignment/calibration").resolve()
    )
    if output_dir != output_root and output_root not in output_dir.parents:
        raise SystemExit(f"output must be under {output_root}")
    for path in (config_path, model_path, source_path, manifest_path):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted = {str(row["sample_id"]): row for row in manifest["samples"]}
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(source_path):
        sample_id = str(row["sample_id"])
        if sample_id in wanted and sample_id not in source_by_id:
            source_by_id[sample_id] = row
    if set(source_by_id) != set(wanted):
        raise SystemExit(f"missing samples: {sorted(set(wanted) - set(source_by_id))}")

    selected: list[dict[str, Any]] = []
    for item in manifest["samples"]:
        source_row = source_by_id[item["sample_id"]]
        prompt_ids = list(map(int, source_row["canonical_prompt_ids"]))
        expected_digest = item.get("canonical_prompt_sha256", source_row["canonical_prompt_sha256"])
        if prompt_digest(prompt_ids) != expected_digest:
            raise SystemExit(f"prompt hash mismatch: {item['sample_id']}")
        selected.append({**item, "canonical_prompt_sha256": expected_digest, "prompt_ids": prompt_ids})
    if len(selected) != args.max_samples:
        raise SystemExit(f"wanted {args.max_samples} samples, found {len(selected)}")

    import torch
    from transformers import GenerationConfig
    from vexact.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VeXactConfig
    from vexact.core.request import InferenceRequest
    from vexact.worker.driver_worker import DriverWorker

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "8576")
    engine_config = VeXactConfig(
        model=ModelConfig(
            model_path=str(model_path),
            attn_impl=args.attn_impl,
            enable_batch_invariant=True,
            max_model_len=2048,
            enforce_eager=True,
        ),
        parallel=ParallelConfig(pipeline_parallel_size=1),
        scheduler=SchedulerConfig(
            max_num_batched_tokens=256 if args.debug else 4096,
            max_num_seqs=args.max_samples,
            enable_chunked_prefill=True,
        ),
        cache=CacheConfig(page_size=64, max_cache_blocks=16 if args.debug else 64),
    )
    engine = DriverWorker(config=engine_config)
    engine.start()

    def collect(requests: list[Any], timeout_seconds: float = 180.0) -> dict[str, Any]:
        request_ids = [request.request_id for request in requests]
        for request in requests:
            engine.submit_request(request)
        pending = set(request_ids)
        completed: dict[str, Any] = {}
        deadline = time.monotonic() + timeout_seconds
        while pending and time.monotonic() < deadline:
            for result in engine.poll_results(timeout=1.0):
                if result.request_id in pending:
                    completed[result.request_id] = result
                    pending.remove(result.request_id)
            if pending and not engine._processing_thread.is_alive():
                raise RuntimeError(
                    "VeXact generation worker exited before completing requests; "
                    "inspect the run log for the worker traceback"
                )
        if pending:
            raise SystemExit(f"VeXact requests timed out: {sorted(pending)}")
        return completed

    request_ids = [item["sample_id"] for item in selected]
    natural_rows: list[dict[str, Any]] = []
    try:
        greedy_requests = [
            InferenceRequest(
                request_id=item["sample_id"],
                input_ids_list=item["prompt_ids"],
                generation_config=GenerationConfig(
                    max_length=len(item["prompt_ids"]) + 1,
                    max_new_tokens=1,
                    do_sample=False,
                    top_p=1.0,
                    output_logits=True,
                    output_scores=True,
                ),
            )
            for item in selected
        ]
        completed = collect(greedy_requests)

        for rollout_index in range(args.n_rollouts):
            natural_requests = [
                InferenceRequest(
                    request_id=f"{item['sample_id']}::natural::{rollout_index}",
                    input_ids_list=item["prompt_ids"],
                    generation_config=GenerationConfig(
                        max_length=len(item["prompt_ids"]) + 1,
                        max_new_tokens=1,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        output_scores=True,
                    ),
                )
                for item in selected
            ]
            natural_completed = collect(natural_requests)
            for item, request in zip(selected, natural_requests, strict=True):
                result = natural_completed[request.request_id]
                if len(result.generated_tokens) != 1:
                    raise SystemExit(f"invalid natural result: {request.request_id}")
                token_id = int(result.generated_tokens[0])
                action = "search" if token_id == 27 else "internal" if token_id == 4159 else "other"
                natural_rows.append(
                    {
                        "sample_id": item["sample_id"],
                        "boundary": item["boundary"],
                        "rollout_index": rollout_index,
                        "run_seed": args.seed,
                        "token_id": token_id,
                        "action": action,
                        "generated_logprob": (
                            float(result.generated_logprobs[0]) if result.generated_logprobs else None
                        ),
                    }
                )
    finally:
        engine.stop()

    all_logits: dict[str, list[Any]] = {}
    all_logprobs: dict[str, list[float]] = {}
    all_token_ids: dict[str, list[int]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for item in selected:
        request_id = item["sample_id"]
        result = completed[request_id]
        if len(result.generated_tokens) != 1 or len(result.generated_logits) != 1:
            raise SystemExit(f"invalid capture for {request_id}")
        all_logits[request_id] = [tensor.detach().cpu() for tensor in result.generated_logits]
        all_logprobs[request_id] = [float(value) for value in result.generated_logprobs]
        all_token_ids[request_id] = item["prompt_ids"] + list(map(int, result.generated_tokens))
        metadata[request_id] = {
            "sample_id": request_id,
            "boundary": item["boundary"],
            "canonical_prompt_sha256": item["canonical_prompt_sha256"],
            "prompt_len": len(item["prompt_ids"]),
            "response_len": 1,
            "generated_token_id": int(result.generated_tokens[0]),
            "seed": args.seed,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(all_logits, output_dir / "all_logits.pt")
    torch.save(all_logprobs, output_dir / "all_logprobs.pt")
    (output_dir / "all_token_ids_list.json").write_text(
        json.dumps(all_token_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    natural_path = output_dir / "natural_samples.jsonl"
    natural_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in natural_rows), encoding="utf-8"
    )
    action_by_boundary: dict[str, Counter[str]] = defaultdict(Counter)
    action_by_sample: dict[str, Counter[str]] = defaultdict(Counter)
    for row in natural_rows:
        action_by_boundary[row["boundary"]][row["action"]] += 1
        action_by_sample[row["sample_id"]][row["action"]] += 1
    no_search = action_by_boundary.get("NoSearch", Counter())
    no_search_total = sum(no_search.values())
    mixed_groups = sum(
        counts["search"] > 0 and counts["internal"] > 0 for counts in action_by_sample.values()
    )
    summary = {
        "purpose": "vexact_exact2_full_logits_capture",
        "gate": "CAPTURE_PASS",
        "model_path": str(model_path),
        "attn_impl": args.attn_impl,
        "dtype": "bfloat16",
        "n_samples": len(selected),
        "n_rollouts": args.n_rollouts,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "sampling_rng": "torch global CUDA RNG (no per-request seed)",
        "run_seed": args.seed,
        "request_ids": request_ids,
        "natural_samples_path": str(natural_path),
        "by_boundary": {key: dict(value) for key, value in action_by_boundary.items()},
        "p_internal_NoSearch": no_search["internal"] / no_search_total if no_search_total else None,
        "mixed_action_group_rate": mixed_groups / len(selected) if natural_rows else None,
    }
    (output_dir / "capture_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
