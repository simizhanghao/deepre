"""Phase 2E2: call Kimi teacher for grounded <think> on hard HotpotQA train items.

Smoke (20, concurrent):
  python scripts/generate_teacher_reasoning.py --max-samples 20 --run-tag smoke20 --concurrency 16

Full (~400):
  python scripts/generate_teacher_reasoning.py --n-persistent 320 --n-other 80 --concurrency 32

Env (optional overrides):
  KIMI_BASE_URL   default http://10.16.137.2:8000/v1
  KIMI_API_KEY    default EMPTY
  KIMI_MODEL      default Kimi-K2.6-CT-FP8KV
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.sft.prototype_builder import (  # noqa: E402
    gold_answer_of,
    load_jsonl,
    resolve_evidence_refs,
)
from src.sft.teacher_reasoning import (  # noqa: E402
    DEFAULT_TEACHER_MODEL,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    format_teacher_user_prompt,
    mine_hard_candidates,
    oracle_em_map_from_metrics,
    validate_teacher_think,
)

DEFAULT_TRAIN = (
    REPO_ROOT / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl"
)
DEFAULT_DIRECT = (
    REPO_ROOT
    / "results/phase2e1_direct_label_n8000_20260807_202826_phase2e1/labels.jsonl"
)
DEFAULT_BASE_ORACLE = (
    REPO_ROOT
    / "results/phase2e1_base_oracle_n8000_20260807_205154/merged/metrics.json"
)
DEFAULT_SFT_ORACLE = (
    REPO_ROOT
    / "results/phase2e1_sftv0_oracle_n8000_20260807_211627/merged/metrics.json"
)

_print_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Kimi grounded <think> cache.")
    p.add_argument("--train-file", type=str, default=str(DEFAULT_TRAIN))
    p.add_argument("--direct-labels", type=str, default=str(DEFAULT_DIRECT))
    p.add_argument("--base-oracle-metrics", type=str, default=str(DEFAULT_BASE_ORACLE))
    p.add_argument("--sft-oracle-metrics", type=str, default=str(DEFAULT_SFT_ORACLE))
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results"))
    p.add_argument("--run-tag", type=str, default="phase2e2_teacher")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-persistent", type=int, default=320)
    p.add_argument("--n-other", type=int, default=80)
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap chosen candidates (use 20 for smoke).",
    )
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-backoff", type=float, default=3.0)
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("KIMI_CONCURRENCY", "32")),
        help="Parallel request workers (default 32).",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("KIMI_BASE_URL", "http://10.16.137.2:8000/v1"),
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("KIMI_API_KEY", "EMPTY"),
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("KIMI_MODEL", DEFAULT_TEACHER_MODEL),
    )
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def load_direct(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows[r["sample_id"]] = r
    return rows


def chat_complete(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int = 3,
    retry_backoff: float = 3.0,
    quiet: bool = False,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt >= retries:
                break
            sleep_s = retry_backoff * attempt
            if not quiet:
                with _print_lock:
                    print(
                        f"[teacher] retry {attempt}/{retries} after error: {exc}; "
                        f"sleep {sleep_s}s"
                    )
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def _pool_tag(sid: str, direct: Dict[str, Any], base_oracle: Dict[str, Any]) -> str:
    d = direct.get(sid) or {}
    d_ok = bool(d.get("direct_correct")) or float(d.get("exact_match") or 0) >= 1.0 - 1e-9
    o_ok = float((base_oracle.get(sid) or {}).get("exact_match") or 0) >= 1.0 - 1e-9
    if (not d_ok) and (not o_ok):
        return "persistent_c_like"
    return "other_hard"


def process_one(
    idx: int,
    total: int,
    sid: str,
    sample: Dict[str, Any],
    args: argparse.Namespace,
    direct: Dict[str, Dict[str, Any]],
    base_oracle: Dict[str, Dict[str, Any]],
) -> Tuple[int, Dict[str, Any]]:
    refs = resolve_evidence_refs(sample)
    gold = gold_answer_of(sample)
    user = format_teacher_user_prompt(sample, refs, gold)
    t0 = time.time()
    err = None
    raw = ""
    try:
        raw = chat_complete(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            retry_backoff=args.retry_backoff,
            quiet=args.concurrency > 1,
        )
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
    latency_ms = round((time.time() - t0) * 1000, 1)
    validation = (
        validate_teacher_think(
            raw,
            gold_answer=gold,
            question=sample["question"],
            refs=refs,
        )
        if err is None
        else {
            "format_valid": False,
            "answer_consistent": False,
            "grounding_valid": False,
            "length_valid": False,
            "n_words": 0,
            "novel_proper_nouns": [],
            "errors": [err or "api_error"],
            "accepted": False,
            "think": None,
        }
    )
    row = {
        "sample_id": sid,
        "question": sample["question"],
        "gold_answer": gold,
        "gold_answers": list(sample.get("gold_answers") or []),
        "evidence_refs": refs,
        "q_type": (sample.get("metadata") or {}).get("type"),
        "teacher_model": args.model,
        "teacher_prompt_version": PROMPT_VERSION,
        "teacher_base_url": args.base_url,
        "teacher_raw_output": raw,
        "teacher_validation": {
            k: validation[k]
            for k in (
                "format_valid",
                "answer_consistent",
                "grounding_valid",
                "length_valid",
                "n_words",
                "novel_proper_nouns",
                "errors",
                "accepted",
            )
        },
        "think": validation.get("think"),
        "latency_ms": latency_ms,
        "api_error": err,
        "reasoning_source": "kimi2.6",
        "pool": _pool_tag(sid, direct, base_oracle),
    }
    status = "OK" if validation["accepted"] else "REJECT"
    with _print_lock:
        print(
            f"[{idx}/{total}] {sid} {status} "
            f"words={validation.get('n_words')} lat={latency_ms}ms "
            f"err={validation.get('errors')[:2]}",
            flush=True,
        )
    return idx, row


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    train_path = resolve(args.train_file)
    samples = load_jsonl(str(train_path))
    by_id = {s["sample_id"]: s for s in samples}
    direct = load_direct(resolve(args.direct_labels))
    base_oracle = oracle_em_map_from_metrics(
        json.loads(resolve(args.base_oracle_metrics).read_text(encoding="utf-8"))
    )
    sft_oracle = oracle_em_map_from_metrics(
        json.loads(resolve(args.sft_oracle_metrics).read_text(encoding="utf-8"))
    )

    chosen, mine_stats = mine_hard_candidates(
        samples_by_id=by_id,
        direct=direct,
        base_oracle=base_oracle,
        sft_oracle=sft_oracle,
        seed=args.seed,
        n_persistent=args.n_persistent,
        n_other=args.n_other,
    )
    if args.max_samples is not None:
        chosen = chosen[: args.max_samples]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = resolve(args.output_dir) / (
        f"teacher_reasoning_n{len(chosen)}_{stamp}_{args.run_tag}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "reasoning_cache.jsonl"
    summary_path = run_dir / "summary.json"

    workers = min(args.concurrency, max(len(chosen), 1))
    print(f"[teacher] model={args.model} base_url={args.base_url}", flush=True)
    print(f"[teacher] run_dir={run_dir}", flush=True)
    print(f"[teacher] mine_stats={mine_stats}", flush=True)
    print(
        f"[teacher] generating n={len(chosen)} concurrency={workers} "
        f"(progress prints when each request returns)",
        flush=True,
    )

    t_all = time.time()
    results: Dict[int, Dict[str, Any]] = {}
    n_ok = 0
    n_fail = 0
    n_done = 0
    with cache_path.open("w", encoding="utf-8") as out, ThreadPoolExecutor(
        max_workers=workers
    ) as ex:
        futs = [
            ex.submit(
                process_one,
                i,
                len(chosen),
                sid,
                by_id[sid],
                args,
                direct,
                base_oracle,
            )
            for i, sid in enumerate(chosen, 1)
        ]
        print(f"[teacher] submitted {len(futs)} jobs", flush=True)
        for fut in as_completed(futs):
            idx, row = fut.result()
            results[idx] = row
            n_done += 1
            if row["teacher_validation"]["accepted"]:
                n_ok += 1
            else:
                n_fail += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if n_done % max(1, min(5, len(chosen))) == 0 or n_done == len(chosen):
                elapsed_so_far = time.time() - t_all
                print(
                    f"[teacher] progress {n_done}/{len(chosen)} "
                    f"ok={n_ok} reject={n_fail} "
                    f"elapsed={elapsed_so_far:.1f}s",
                    flush=True,
                )

    elapsed = round(time.time() - t_all, 2)
    summary = {
        "num_requested": len(chosen),
        "num_accepted": n_ok,
        "num_rejected": n_fail,
        "accept_rate": round(n_ok / max(len(chosen), 1), 4),
        "concurrency": workers,
        "elapsed_seconds": elapsed,
        "throughput_qps": round(len(chosen) / max(elapsed, 1e-6), 3),
        "mine_stats": mine_stats,
        "teacher_model": args.model,
        "teacher_prompt_version": PROMPT_VERSION,
        "cache_path": str(cache_path),
        "run_dir": str(run_dir),
        "phase": "2E2",
        "purpose": "kimi_grounded_think_smoke"
        if (args.max_samples or 0) <= 50
        else "kimi_grounded_think",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[teacher] artifacts -> {run_dir}")


if __name__ == "__main__":
    main()
