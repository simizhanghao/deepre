#!/usr/bin/env python3
"""Calibrate vLLM route-token probabilities against the frozen HF reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO / "outputs/rl/03_hf_evidence_step400"
DEFAULT_DUMP = (
    REPO
    / "results/16_audit_routing_exploration/worker_mismatch/dump_pathC.jsonl"
)
DEFAULT_SAMPLE_IDS = (
    REPO
    / "results/16_audit_routing_exploration/worker_mismatch/sample_ids.json"
)
DEFAULT_HF = (
    REPO
    / "results/16_audit_routing_exploration/worker_mismatch/hf_route_scores.jsonl"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO / "configs/rl/grpo_smoke128.yaml"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL))
    p.add_argument("--source-dump", default=str(DEFAULT_DUMP))
    p.add_argument("--sample-ids", default=str(DEFAULT_SAMPLE_IDS))
    p.add_argument("--hf-reference", default=str(DEFAULT_HF))
    p.add_argument("--n-rollouts", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    p.add_argument("--max-model-len", type=int, default=2048)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def logprob_value(entry: Any) -> float | None:
    if entry is None:
        return None
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    if isinstance(entry, dict) and "logprob" in entry:
        return float(entry["logprob"])
    if isinstance(entry, (int, float)):
        return float(entry)
    return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def main() -> None:
    args = parse_args()
    if args.debug and args.max_samples > 2:
        raise SystemExit("--debug requires --max-samples <= 2")
    if args.max_samples < 1 or args.n_rollouts < 1:
        raise SystemExit("max-samples and n-rollouts must be positive")

    model_path = Path(args.model_path).resolve()
    source_dump = Path(args.source_dump).resolve()
    sample_ids_path = Path(args.sample_ids).resolve()
    hf_path = Path(args.hf_reference).resolve()
    config_path = Path(args.config).resolve()
    out_dir = Path(args.output_dir).resolve()
    for path in (model_path, source_dump, sample_ids_path, hf_path, config_path):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    boundary_by_id = {
        str(row["sample_id"]): str(row["boundary"])
        for row in json.loads(sample_ids_path.read_text(encoding="utf-8"))["samples"]
    }
    hf_by_id = {str(row["sample_id"]): row for row in read_jsonl(hf_path)}

    unique: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(source_dump):
        sid = str(row["sample_id"])
        if sid not in unique:
            unique[sid] = row
    selected = list(unique.values())[: args.max_samples]
    if len(selected) != args.max_samples:
        raise SystemExit(f"wanted {args.max_samples} unique prompts, found {len(selected)}")

    from transformers import AutoTokenizer, __version__ as transformers_version
    from vllm import LLM, SamplingParams, __version__ as vllm_version
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    search_ids = tokenizer.encode("<search>", add_special_tokens=False)
    internal_ids = tokenizer.encode("<internal>", add_special_tokens=False)
    if search_ids != [27, 1836, 29] or internal_ids != [4159, 2978, 29]:
        raise SystemExit(
            f"route tokenization mismatch: search={search_ids} internal={internal_ids}"
        )
    search_tok, internal_tok = search_ids[0], internal_ids[0]

    prompts = [list(map(int, row["canonical_prompt_ids"])) for row in selected]
    for row, prompt in zip(selected, prompts):
        digest = hashlib.sha256(json.dumps(prompt).encode("utf-8")).hexdigest()
        if digest != row["canonical_prompt_sha256"]:
            raise SystemExit(f"canonical prompt hash mismatch: {row['sample_id']}")

    engine = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_logprobs=50,
        trust_remote_code=False,
    )

    # vLLM reports model logprobs for the generated token and requested top-k.
    # Greedy generation avoids stochastic action choice in this execution gate.
    raw_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        logprobs=50,
    )
    raw_outputs = engine.generate(
        prompt_token_ids=prompts, sampling_params=raw_params, use_tqdm=True
    )

    natural_params = SamplingParams(
        n=args.n_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        max_tokens=1,
        logprobs=5,
    )
    natural_outputs = engine.generate(
        prompt_token_ids=prompts, sampling_params=natural_params, use_tqdm=True
    )

    detail_rows: list[dict[str, Any]] = []
    abs_delta_search: list[float] = []
    action_by_boundary: dict[str, Counter[str]] = defaultdict(Counter)
    mixed_groups = 0
    for src, raw_req, natural_req in zip(selected, raw_outputs, natural_outputs):
        sid = str(src["sample_id"])
        boundary = boundary_by_id.get(sid, "Unknown")
        lp_map = raw_req.outputs[0].logprobs[0] or {}
        lp_search = logprob_value(lp_map.get(search_tok))
        lp_internal = logprob_value(lp_map.get(internal_tok))
        hf = hf_by_id.get(sid, {})
        hf_search = hf.get("L_search_first_tok")
        hf_internal = hf.get("L_internal_first_tok")
        if lp_search is not None and hf_search is not None:
            abs_delta_search.append(abs(lp_search - float(hf_search)))

        counts: Counter[str] = Counter()
        sampled: list[dict[str, Any]] = []
        for output in natural_req.outputs:
            tok = int(output.token_ids[0]) if output.token_ids else -1
            if tok == search_tok:
                action = "search"
            elif tok == internal_tok:
                action = "internal"
            else:
                action = "other"
            counts[action] += 1
            action_by_boundary[boundary][action] += 1
            sampled.append({"token_id": tok, "action": action, "text": output.text})
        if counts["search"] > 0 and counts["internal"] > 0:
            mixed_groups += 1

        denom = None
        p_internal_pair = None
        if lp_search is not None and lp_internal is not None:
            denom = math.exp(lp_search) + math.exp(lp_internal)
            p_internal_pair = math.exp(lp_internal) / denom
        detail_rows.append(
            {
                "sample_id": sid,
                "boundary": boundary,
                "canonical_prompt_sha256": src["canonical_prompt_sha256"],
                "canonical_prompt_len": len(prompts[len(detail_rows)]),
                "vllm_logp_search_tok0": lp_search,
                "vllm_logp_internal_tok0": lp_internal,
                "vllm_pair_normalized_p_internal": p_internal_pair,
                "hf_logp_search_tok0": hf_search,
                "hf_logp_internal_tok0": hf_internal,
                "abs_delta_search_tok0": (
                    abs(lp_search - float(hf_search))
                    if lp_search is not None and hf_search is not None
                    else None
                ),
                "natural_counts": dict(counts),
                "samples": sampled,
            }
        )

    detail_path = out_dir / "vllm_route_calibration.jsonl"
    with detail_path.open("w", encoding="utf-8") as f:
        for row in detail_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    no_search_counts = action_by_boundary.get("NoSearch", Counter())
    no_search_total = sum(no_search_counts.values())
    p_internal_no_search = (
        no_search_counts["internal"] / no_search_total if no_search_total else 0.0
    )
    median_delta = statistics.median(abs_delta_search) if abs_delta_search else None
    p95_delta = percentile(abs_delta_search, 0.95)
    summary = {
        "purpose": "vllm_route_token_calibration",
        "mode": "debug_smoke" if args.debug else "gate_a_calibration",
        "model_path": str(model_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
        "source_dump": str(source_dump),
        "hf_reference": str(hf_path),
        "versions": {
            "vllm": vllm_version,
            "transformers": transformers_version,
            "torch": torch.__version__,
        },
        "protocol": {
            "seed": args.seed,
            "n_questions": len(selected),
            "n_rollouts": args.n_rollouts,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "natural_temperature": args.temperature,
            "natural_top_p": args.top_p,
            "raw_probe": "greedy generation; model top-50 logprobs",
        },
        "route_token_ids": {"search": search_tok, "internal": internal_tok},
        "by_boundary": {k: dict(v) for k, v in action_by_boundary.items()},
        "metrics": {
            "median_abs_delta_search_tok0": median_delta,
            "p95_abs_delta_search_tok0": p95_delta,
            "p_internal_NoSearch": p_internal_no_search,
            "mixed_action_group_rate": mixed_groups / len(selected),
        },
        "gate": (
            "SMOKE_ONLY"
            if args.debug
            else (
                "PASS"
                if median_delta is not None
                and p95_delta is not None
                and median_delta <= 0.02
                and p95_delta <= 0.05
                and p_internal_no_search > 0.10
                and mixed_groups > 0
                else "FAIL"
            )
        ),
        "detail_path": str(detail_path),
    }
    summary_path = out_dir / "vllm_route_calibration_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
