#!/usr/bin/env python3
"""Minimal Evidence@400 dense rollout smoke for the official VeXact stack."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO / "outputs/rl/03_hf_evidence_step400"
DEFAULT_SOURCE = REPO / "results/16_audit_routing_exploration/worker_mismatch/dump_pathC.jsonl"
DEFAULT_SAMPLES = REPO / "results/17_rollout_alignment/calibration/sample_ids_exact2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--source-dump", default=str(DEFAULT_SOURCE))
    parser.add_argument("--sample-manifest", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--attn-impl", default="triton-invariant")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_digest(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids).encode("utf-8")).hexdigest()


async def generate_all(engine: Any, requests: list[Any]) -> list[Any]:
    return await asyncio.gather(*(engine.generate(request, timeout=180.0) for request in requests))


def main() -> None:
    args = parse_args()
    if not args.debug:
        raise SystemExit("this entry point is smoke-only; pass --debug")
    if args.max_samples != 2:
        raise SystemExit("VeXact dense smoke requires --max-samples 2")
    if args.attn_impl != "triton-invariant":
        raise SystemExit("A100 smoke requires --attn-impl triton-invariant")

    config_path = Path(args.config).resolve()
    model_path = Path(args.model_path).resolve()
    source_path = Path(args.source_dump).resolve()
    manifest_path = Path(args.sample_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    expected_output_root = (REPO / "outputs").resolve()
    if output_dir != expected_output_root and expected_output_root not in output_dir.parents:
        raise SystemExit(f"smoke output must be under {expected_output_root}")
    for path in (config_path, model_path, source_path, manifest_path):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted = {row["sample_id"]: row for row in manifest["samples"]}
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(source_path):
        sample_id = str(row["sample_id"])
        if sample_id in wanted and sample_id not in source_by_id:
            source_by_id[sample_id] = row
    if set(source_by_id) != set(wanted):
        raise SystemExit(f"missing frozen samples: {sorted(set(wanted) - set(source_by_id))}")

    selected: list[dict[str, Any]] = []
    for item in manifest["samples"]:
        row = source_by_id[item["sample_id"]]
        prompt_ids = list(map(int, row["canonical_prompt_ids"]))
        digest = prompt_digest(prompt_ids)
        if digest != item["canonical_prompt_sha256"]:
            raise SystemExit(f"prompt hash mismatch: {item['sample_id']}")
        selected.append({**item, "prompt_ids": prompt_ids})

    from transformers import GenerationConfig
    from vexact.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VeXactConfig
    from vexact.core.request import DriverRequest
    from vexact.engine import VeXact

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
            max_num_batched_tokens=256,
            max_num_seqs=2,
            enable_chunked_prefill=True,
        ),
        cache=CacheConfig(page_size=64, max_cache_blocks=16),
    )

    engine = VeXact(engine_config)
    try:
        requests = [
            DriverRequest(
                request_id=item["sample_id"],
                generation_config=GenerationConfig(
                    max_length=len(item["prompt_ids"]) + 1,
                    max_new_tokens=1,
                    do_sample=False,
                    top_p=1.0,
                    output_scores=True,
                    seed=args.seed,
                ),
                input_ids_list=item["prompt_ids"],
            )
            for item in selected
        ]
        results = asyncio.run(generate_all(engine, requests))
    finally:
        engine.close()

    rows: list[dict[str, Any]] = []
    for item, result in zip(selected, results, strict=True):
        if len(result.new_token_ids) != 1 or result.new_logprobs is None or len(result.new_logprobs) != 1:
            raise SystemExit(f"invalid VeXact result for {item['sample_id']}: {result}")
        rows.append(
            {
                "sample_id": item["sample_id"],
                "boundary": item["boundary"],
                "canonical_prompt_sha256": item["canonical_prompt_sha256"],
                "canonical_prompt_len": len(item["prompt_ids"]),
                "generated_token_id": int(result.new_token_ids[0]),
                "generated_logprob": float(result.new_logprobs[0]),
                "status": str(result.status),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "vexact_dense_smoke.jsonl"
    details_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "purpose": "vexact_dense_rollout_smoke",
        "gate": "PASS",
        "config": str(config_path),
        "model_path": str(model_path),
        "sample_manifest": str(manifest_path),
        "seed": args.seed,
        "max_samples": args.max_samples,
        "dtype": "bfloat16",
        "attn_impl": args.attn_impl,
        "batch_invariant": True,
        "details_path": str(details_path),
        "rows": rows,
    }
    summary_path = output_dir / "vexact_dense_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
