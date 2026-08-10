#!/usr/bin/env python3
"""Routing Sampler Alignment Audit — HF vs SGLang route distribution.

NO TRAINING. Extends results/16_audit_routing_exploration/worker_mismatch/.

Locked order (after Path C + HF score + Path B-current):
  branch-nucleus → HF logits at route branch-point + top_p membership
  sample-hf      → Exact-ID HF sampling (top_p=.95 vs 1.0)
  sampler-align  → both of the above in one model load (preferred)
  (B-none deferred; TIM logprob / greedy parity only if needed)

Example:
  python scripts/audit_routing_worker_mismatch.py --phase sampler-align \\
    --n-rollouts 16 --max-new-tokens 64 --temperature 0.9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results/16_audit_routing_exploration/worker_mismatch"
PARITY_META = REPO / "results/16_audit_routing_exploration/parity_sglang_32x4/prepare_meta.json"
PARITY_PQ = REPO / "results/16_audit_routing_exploration/parity_sglang_32x4/train_parity_32.parquet"
VALID = ("NoSearch", "NeedSearch", "Undetermined")
OPEN_TAGS = ("<search>", "<internal>")
STOP_STRINGS = ("</search>", "</answer>", "</internal>")


def _stop_tokenization_report(tokenizer: Any) -> dict[str, Any]:
    """Host-side copy of Eca stop forensic (no verl import)."""
    tags: dict[str, Any] = {}
    last_ids: List[int] = []
    for s in STOP_STRINGS:
        ids = list(tokenizer.encode(s, add_special_tokens=False))
        last = ids[-1] if ids else None
        tags[s] = {
            "full_token_ids": ids,
            "last_token_id": last,
            "decoded_last_token": (
                tokenizer.decode([last], skip_special_tokens=False) if last is not None else None
            ),
        }
        if last is not None:
            last_ids.append(int(last))
    collision: dict[str, list[str]] = {}
    for s, info in tags.items():
        lid = info["last_token_id"]
        if lid is None:
            continue
        collision.setdefault(str(lid), []).append(s)
    collision = {k: v for k, v in collision.items() if len(v) > 1}
    return {
        "closing_tags": tags,
        "last_token_collision": collision,
        "unique_last_token_ids": sorted(set(last_ids)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Routing Sampler Alignment Audit")
    p.add_argument(
        "--phase",
        choices=(
            "prepare",
            "print-cmd",
            "score",
            "branch-nucleus",
            "sample-hf",
            "sampler-align",
            "greedy-tim",
            "aggregate",
        ),
        required=True,
    )
    p.add_argument("--config", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument(
        "--boundary-table",
        type=str,
        default=str(REPO / "outputs/rl/04_table_search_boundary/boundary_latest.json"),
    )
    p.add_argument("--parity-meta", type=str, default=str(PARITY_META))
    p.add_argument("--parity-parquet", type=str, default=str(PARITY_PQ))
    p.add_argument("--model-path", type=str, default=str(REPO / "outputs/rl/03_hf_evidence_step400"))
    p.add_argument("--n-rollouts", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--top-p-list",
        type=str,
        default="0.95,1.0",
        help="comma-separated top_p values for sample-hf / sampler-align",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--path", choices=("B", "C"), default="C", help="print-cmd path")
    p.add_argument("--stop-mode", choices=("current", "none"), default="current")
    p.add_argument("--container", type=str, default="eca-verl")
    p.add_argument("--cuda-devices", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--n-gpus", type=int, default=8)
    p.add_argument("--n-nosearch", type=int, default=12)
    p.add_argument("--n-needsearch", type=int, default=8)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--max-samples", type=int, default=0, help="debug shrink")
    return p.parse_args()


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO / path


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(REPO),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_boundary(path: Path) -> Dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    m = raw["boundary"] if isinstance(raw, dict) and isinstance(raw.get("boundary"), dict) else raw
    out: Dict[str, str] = {}
    for k, v in m.items():
        if isinstance(v, dict):
            lab = str(v.get("boundary") or v.get("label") or "Undetermined")
        else:
            lab = str(v)
        out[str(k)] = lab if lab in VALID else "Undetermined"
    return out


def _ws(host_path: Path) -> str:
    try:
        rel = host_path.resolve().relative_to(REPO.resolve())
        return f"/workspace/deepresearch/{rel.as_posix()}"
    except ValueError:
        return str(host_path)


def _sha_ids(ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode("utf-8")).hexdigest()


def _normalize_route(text: str) -> str:
    import re

    has_s = bool(re.search(r"<search>.*?</search>", text, re.DOTALL | re.I))
    has_i = bool(re.search(r"<internal>.*?</internal>", text, re.DOTALL | re.I))
    has_a = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL | re.I))
    if has_s and has_i:
        return "both"
    if has_s:
        return "search"
    if has_i:
        return "internal"
    if has_a:
        return "answer"
    if "<search>" in text.lower():
        return "search"
    if "<internal>" in text.lower():
        return "internal"
    return "other"


def phase_prepare(args: argparse.Namespace) -> None:
    import pandas as pd

    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(resolve(args.parity_meta).read_text(encoding="utf-8"))
    labels = load_boundary(resolve(args.boundary_table))
    ids = [str(x) for x in meta["selected_ids"]]
    by: Dict[str, List[str]] = defaultdict(list)
    for sid in ids:
        by[labels.get(sid, "Undetermined")].append(sid)

    n_no = int(args.n_nosearch)
    n_need = int(args.n_needsearch)
    if args.debug:
        n_no, n_need = min(2, n_no), min(2, n_need)
    # Parity-32 only has 11 NoSearch — take all available, never invent IDs.
    nosearch = by["NoSearch"][:n_no]
    need = by["NeedSearch"][:n_need]
    if not nosearch:
        raise SystemExit(f"NoSearch available={len(by['NoSearch'])} want={n_no}")
    want = list(nosearch) + list(need)
    # veRL: (BATCH * N) must be divisible by n_gpus. Pad with extra NeedSearch
    # (never invent IDs outside the frozen parity-32 pool).
    n_roll = int(args.n_rollouts)
    n_gpus = int(args.n_gpus)
    while want and (len(want) * n_roll) % n_gpus != 0:
        extras = [s for s in by["NeedSearch"] if s not in want]
        if not extras:
            extras = [s for s in by["NoSearch"] if s not in want]
        if not extras:
            # last resort: duplicate last id (same prompt twice) to satisfy divisor
            want.append(want[-1])
            print(
                f"[mismatch] WARN padded by duplicating {want[-1]} for batch divisor",
                flush=True,
            )
            break
        want.append(extras[0])
        print(f"[mismatch] pad +1 {extras[0]} so BATCH*N % n_gpus == 0", flush=True)
    if args.max_samples and args.max_samples > 0:
        want = want[: int(args.max_samples)]
        while want and (len(want) * n_roll) % n_gpus != 0:
            want.append(want[-1])

    pq_src = resolve(args.parity_parquet)
    df = pd.read_parquet(pq_src)

    def sid_of(row: Any) -> str:
        ei = row.get("extra_info") if hasattr(row, "get") else None
        if ei is None and "extra_info" in row:
            ei = row["extra_info"]
        if isinstance(ei, str):
            ei = json.loads(ei)
        if not isinstance(ei, dict):
            ei = dict(ei) if ei is not None else {}
        return str(ei.get("sample_id"))

    mask = df.apply(lambda r: sid_of(r) in set(want), axis=1)
    pool = df.loc[mask].copy()
    sid_to_row = {sid_of(r): r for _, r in pool.iterrows()}
    missing = [s for s in want if s not in sid_to_row]
    if missing:
        raise SystemExit(f"missing ids in parquet: {missing[:5]}")
    sub = pd.DataFrame([sid_to_row[s] for s in want])
    if len(sub) != len(want):
        raise SystemExit(f"parquet subset size={len(sub)} want={len(want)}")

    pq_out = out_dir / f"train_mismatch_{len(sub)}.parquet"
    sub.to_parquet(pq_out, index=False)

    frozen = []
    for sid in want:
        frozen.append(
            {
                "sample_id": sid,
                "boundary": labels.get(sid, "Undetermined"),
                "parity_t0.9_route": "search",
                "parity_t1.3_route": "search",
                "note": "from failed SGLang parity 32x4 (all-search)",
            }
        )
    n_no_used = sum(1 for s in want if labels.get(s) == "NoSearch")
    n_need_used = sum(1 for s in want if labels.get(s) == "NeedSearch")
    (out_dir / "sample_ids.json").write_text(
        json.dumps(
            {
                "purpose": "routing_worker_mismatch_frozen_ids",
                "git_commit": git_commit(),
                "n_NoSearch_requested": n_no,
                "n_NoSearch_available_in_parity32": len(by["NoSearch"]),
                "n_NoSearch_used": n_no_used,
                "n_NeedSearch_used": n_need_used,
                "n_total": len(want),
                "n_rollouts": n_roll,
                "n_gpus": n_gpus,
                "batch_times_n": len(want) * n_roll,
                "samples": frozen,
                "parquet": str(pq_out),
                "parity_meta": str(resolve(args.parity_meta)),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n": len(want),
                "parquet": str(pq_out),
                "hist": dict(Counter(labels[s] for s in want)),
                "batch_times_n": len(want) * n_roll,
            },
            indent=2,
        )
    )


def phase_print_cmd(args: argparse.Namespace) -> None:
    out_dir = resolve(args.output_dir)
    sid_meta = json.loads((out_dir / "sample_ids.json").read_text(encoding="utf-8"))
    pq_host = Path(sid_meta["parquet"])
    n_q = int(sid_meta.get("n_total") or (int(sid_meta.get("n_NoSearch_used", 0)) + int(sid_meta.get("n_NeedSearch_used", 0))))
    if args.debug:
        n_q = min(n_q, 4)
    n = int(args.n_rollouts)
    if (n_q * n) % int(args.n_gpus) != 0:
        raise SystemExit(
            f"BATCH*N={n_q*n} not divisible by n_gpus={args.n_gpus}; re-run --phase prepare"
        )
    path = args.path
    first_only = "1" if path == "B" else "0"
    stop_mode = args.stop_mode
    # Path B stop ablation writes separate dumps so current vs none do not clobber.
    if path == "B":
        dump_host = out_dir / f"dump_pathB_stop_{stop_mode}.jsonl"
        dump_tag = f"B_stop_{stop_mode}"
        audit_max_new = int(args.max_new_tokens) if int(args.max_new_tokens) > 0 else 128
    else:
        dump_host = out_dir / "dump_pathC.jsonl"
        dump_tag = "C"
        audit_max_new = 0  # 0 → unset; use remaining response budget
    dump_ws = _ws(dump_host)
    pq_ws = _ws(pq_host)
    out_ws = _ws(out_dir / f"ckpt_scratch_path{dump_tag}")
    log_host = out_dir / f"run_path{dump_tag}.log"
    n = int(args.n_rollouts)
    t = float(args.temperature)
    max_new_export = (
        f"export ECA_AUDIT_MAX_NEW_TOKENS={audit_max_new}"
        if path == "B"
        else "unset ECA_AUDIT_MAX_NEW_TOKENS || true"
    )

    cmd = f"""# === Routing Worker Mismatch — Path {dump_tag} ===
# Path B: first-generate-only (+ stop ablation) | Path C: full EcaSearchAgentLoop
cd /data1/hcc/deepresearch
mkdir -p "{out_dir}"
docker exec {args.container} bash -lc 'rm -f {dump_ws}' 2>/dev/null || rm -f "{dump_host}" || true
curl -sf http://127.0.0.1:8001/health || echo "START RETRIEVER first"
docker start {args.container} >/dev/null

docker exec -e CUDA_VISIBLE_DEVICES={args.cuda_devices} {args.container} bash -lc '
set -euo pipefail
test -d /workspace/verl || {{ echo MISSING_VERL; exit 1; }}
export PYTHONPATH=/workspace/deepresearch:/workspace/verl
export ECA_BOUNDARY_TABLE=/workspace/deepresearch/outputs/rl/04_table_search_boundary/boundary_latest.json
export ECA_BOUNDARY_STRICT=1
export ECA_EVIDENCE_WEIGHT=0.5
export ECA_SEARCH_COST_WEIGHT=0.30
export ECA_ROUTING_MISMATCH_AUDIT=1
export ECA_AUDIT_FIRST_GENERATE_ONLY={first_only}
export ECA_AUDIT_STOP_MODE={stop_mode}
export ECA_AUDIT_PATH={path}
{max_new_export}
export ECA_ROUTING_MISMATCH_DUMP={dump_ws}
cd /workspace/deepresearch
rm -f {dump_ws}
STEPS=1 TOTAL_EPOCHS=1 BATCH={n_q} N={n} \\
  ROLLOUT_TEMP={t} ROLLOUT_TOP_P={args.top_p} ACTOR_LR=0 N_GPUS={args.n_gpus} \\
  TRAIN_FILE={pq_ws} VAL_FILE={pq_ws} \\
  MODEL_PATH=/workspace/deepresearch/outputs/rl/03_hf_evidence_step400 \\
  OUT_DIR={out_ws} EXPERIMENT_NAME=mismatch_path{dump_tag} \\
  SAVE_FREQ=9999 RESUME_MODE=disable VAL_BEFORE_TRAIN=False \\
  GPU_MEM_UTIL=0.55 MICRO_BATCH=1 \\
  bash scripts/run_grpo_boundary.sh
' 2>&1 | tee "{log_host}"

echo "DUMP_LINES=$(wc -l < "{dump_host}" 2>/dev/null || echo 0)"
echo "DUMP_PATH={dump_host}"
"""
    cmd_path = out_dir / f"run_cmd_path{dump_tag}.sh"
    cmd_path.write_text(cmd, encoding="utf-8")
    print(cmd)
    print(f"\n# saved: {cmd_path}", flush=True)


def _load_canonical_prompts(dump_path: Path) -> Dict[str, dict[str, Any]]:
    """One canonical prompt per sample_id (first dump row wins)."""
    by: Dict[str, dict[str, Any]] = {}
    if not dump_path.is_file():
        return by
    with dump_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row["sample_id"])
            if sid in by:
                # verify equality
                if row.get("canonical_prompt_ids") != by[sid].get("canonical_prompt_ids"):
                    by[sid]["_prompt_conflict"] = True
                continue
            by[sid] = row
    return by


def _sequence_logprob(
    model: Any,
    input_ids: List[int],
    cont_ids: List[int],
    device: str,
) -> Tuple[float, float]:
    import torch
    import torch.nn.functional as F

    if not cont_ids:
        return 0.0, 0.0
    full = torch.tensor([input_ids + cont_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(full)
        logits = out.logits[0]  # [T, V]
    # Predict token at position len(input_ids)+j from prefix ending at that index-1
    start = len(input_ids) - 1
    logps: List[float] = []
    for j, tid in enumerate(cont_ids):
        pos = start + j
        lp = F.log_softmax(logits[pos], dim=-1)[int(tid)].item()
        logps.append(float(lp))
    s = float(sum(logps))
    m = s / len(logps)
    return s, m


def _lcp_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and int(a[i]) == int(b[i]):
        i += 1
    return i


def _nucleus_membership(
    probs_1d: Any,
    token_id: int,
    top_p: float,
) -> dict[str, Any]:
    """HF-compatible top-p keep-set check (temperature already applied upstream)."""
    import torch

    probs = probs_1d.detach().float()
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    remove = cumsum > float(top_p)
    # Keep the first token that crosses top_p (HF shift).
    remove_shifted = remove.clone()
    remove_shifted[1:] = remove[:-1].clone()
    remove_shifted[0] = False
    kept = ~remove_shifted
    pos_matches = (sorted_idx == int(token_id)).nonzero(as_tuple=False)
    if pos_matches.numel() == 0:
        return {
            "token_id": int(token_id),
            "p": 0.0,
            "rank": -1,
            "cum_mass_before": 1.0,
            "in_top_p": False,
        }
    pos = int(pos_matches[0].item())
    p = float(sorted_probs[pos].item())
    cum_before = float(cumsum[pos].item() - p)
    return {
        "token_id": int(token_id),
        "p": p,
        "rank": pos + 1,
        "cum_mass_before": cum_before,
        "in_top_p": bool(kept[pos].item()),
    }


def _route_open_ids(tokenizer: Any) -> Dict[str, List[int]]:
    return {tag: list(tokenizer.encode(tag, add_special_tokens=False)) for tag in OPEN_TAGS}


def _load_prompt_source(out_dir: Path) -> Tuple[Path, Dict[str, dict[str, Any]]]:
    dump_c = out_dir / "dump_pathC.jsonl"
    dump_b = out_dir / "dump_pathB_stop_current.jsonl"
    dump_b_any = sorted(out_dir.glob("dump_pathB*.jsonl"))
    if dump_c.is_file():
        source = dump_c
    elif dump_b.is_file():
        source = dump_b
    elif dump_b_any:
        source = dump_b_any[0]
    else:
        raise SystemExit("need dump_pathC.jsonl or dump_pathB*.jsonl")
    by = _load_canonical_prompts(source)
    if not by:
        raise SystemExit(f"empty dump: {source}")
    return source, by


def _parse_top_p_list(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise SystemExit("empty --top-p-list")
    return vals


def _top_p_tag(top_p: float) -> str:
    # Stable filename tag: 0.95 -> 0p95, 1.0 -> 1p0
    s = f"{top_p:g}".replace(".", "p")
    return s


def _load_hf_model(model_path: Path, device: str) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=__import__("torch").bfloat16, local_files_only=True
    ).to(device).eval()
    return model, tokenizer


def _branch_nucleus_for_prompt(
    model: Any,
    tokenizer: Any,
    prompt_ids: List[int],
    search_ids: List[int],
    internal_ids: List[int],
    temperature: float,
    top_p_probe: float,
    device: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    lcp = _lcp_len(search_ids, internal_ids)
    shared = list(search_ids[:lcp])
    if lcp >= len(search_ids) or lcp >= len(internal_ids):
        raise SystemExit("search/internal tags do not diverge; cannot branch-nucleus")
    search_branch = int(search_ids[lcp])
    internal_branch = int(internal_ids[lcp])
    state_ids = list(prompt_ids) + shared
    inp = torch.tensor([state_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(inp).logits[0, -1].float()
    T = float(temperature) if float(temperature) > 0 else 1.0
    probs = F.softmax(logits / T, dim=-1)
    search_m = _nucleus_membership(probs, search_branch, top_p_probe)
    internal_m = _nucleus_membership(probs, internal_branch, top_p_probe)
    # also top_p=1.0 sanity (always true if p>0)
    search_m1 = _nucleus_membership(probs, search_branch, 1.0)
    internal_m1 = _nucleus_membership(probs, internal_branch, 1.0)
    return {
        "lcp_len": lcp,
        "shared_prefix_ids": shared,
        "shared_prefix_text": tokenizer.decode(shared, skip_special_tokens=False) if shared else "",
        "search_ids": search_ids,
        "internal_ids": internal_ids,
        "search_branch_token_id": search_branch,
        "internal_branch_token_id": internal_branch,
        "search_branch_decoded": tokenizer.decode([search_branch], skip_special_tokens=False),
        "internal_branch_decoded": tokenizer.decode([internal_branch], skip_special_tokens=False),
        "temperature": T,
        "top_p_probe": float(top_p_probe),
        "p_search_branch": search_m["p"],
        "p_internal_branch": internal_m["p"],
        "rank_search": search_m["rank"],
        "rank_internal": internal_m["rank"],
        "cum_mass_before_search": search_m["cum_mass_before"],
        "cum_mass_before_internal": internal_m["cum_mass_before"],
        "search_in_top_p": search_m["in_top_p"],
        "internal_in_top_p": internal_m["in_top_p"],
        "search_in_top_p_1": search_m1["in_top_p"],
        "internal_in_top_p_1": internal_m1["in_top_p"],
        "log_odds_search_minus_internal": float(
            math.log(max(search_m["p"], 1e-30)) - math.log(max(internal_m["p"], 1e-30))
        ),
    }


def _summarize_nucleus_rows(rows: List[dict[str, Any]], boundary: str) -> dict[str, Any]:
    xs = [r for r in rows if r.get("boundary") == boundary]
    n = len(xs)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "frac_internal_in_top_p_095": sum(1 for r in xs if r.get("internal_in_top_p")) / n,
        "frac_search_in_top_p_095": sum(1 for r in xs if r.get("search_in_top_p")) / n,
        "median_rank_internal": sorted(int(r["rank_internal"]) for r in xs)[n // 2],
        "median_rank_search": sorted(int(r["rank_search"]) for r in xs)[n // 2],
        "median_p_internal_branch": sorted(float(r["p_internal_branch"]) for r in xs)[n // 2],
        "median_p_search_branch": sorted(float(r["p_search_branch"]) for r in xs)[n // 2],
        "median_cum_before_internal": sorted(float(r["cum_mass_before_internal"]) for r in xs)[n // 2],
        "lcp_len_mode": Counter(int(r["lcp_len"]) for r in xs).most_common(1)[0][0],
    }


def phase_branch_nucleus(args: argparse.Namespace, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
    import torch

    out_dir = resolve(args.output_dir)
    source, by = _load_prompt_source(out_dir)
    model_path = resolve(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    own_model = model is None
    if model is None or tokenizer is None:
        model, tokenizer = _load_hf_model(model_path, device)

    open_ids = _route_open_ids(tokenizer)
    search_ids = open_ids["<search>"]
    internal_ids = open_ids["<internal>"]
    labels = load_boundary(resolve(args.boundary_table))
    sid_meta = json.loads((out_dir / "sample_ids.json").read_text(encoding="utf-8"))
    frozen = {s["sample_id"]: s for s in sid_meta["samples"]}
    T = float(args.temperature)
    top_p = float(args.top_p)

    rows: List[dict[str, Any]] = []
    for sid, crow in sorted(by.items()):
        prompt_ids = list(crow["canonical_prompt_ids"])
        stats = _branch_nucleus_for_prompt(
            model, tokenizer, prompt_ids, search_ids, internal_ids, T, top_p, device
        )
        rows.append(
            {
                "sample_id": sid,
                "boundary": frozen.get(sid, {}).get("boundary") or labels.get(sid),
                "canonical_prompt_sha256": crow.get("canonical_prompt_sha256") or _sha_ids(prompt_ids),
                "canonical_prompt_len": len(prompt_ids),
                **stats,
            }
        )

    out_jsonl = out_dir / "hf_branch_nucleus.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    nos = _summarize_nucleus_rows(rows, "NoSearch")
    need = _summarize_nucleus_rows(rows, "NeedSearch")
    # Gate: if most NoSearch have internal pruned by top_p=0.95
    prune = bool(nos.get("n", 0) > 0 and nos.get("frac_internal_in_top_p_095", 1.0) <= 0.20)
    support_in_nucleus = bool(
        nos.get("n", 0) > 0 and nos.get("frac_internal_in_top_p_095", 0.0) >= 0.50
    )
    if prune:
        layer = "NUCLEUS_TRUNCATION_CAUSES_ROUTING_COLLAPSE"
    elif support_in_nucleus:
        layer = "NUCLEUS_DOES_NOT_EXPLAIN_COLLAPSE"
    else:
        layer = "NUCLEUS_INCONCLUSIVE"

    summary = {
        "purpose": "hf_branch_point_nucleus_audit",
        "git_commit": git_commit(),
        "model_path": str(model_path),
        "source_dump": str(source),
        "temperature": T,
        "top_p_probe": top_p,
        "open_tag_ids": open_ids,
        "note_lcp": (
            "For Qwen2.5, encode('<search>') vs encode('<internal>') often diverge at "
            "token0 (no shared prefix). Branch stats are still valid at that first token."
        ),
        "NoSearch": nos,
        "NeedSearch": need,
        "layer_verdict": layer,
        "n_rows": len(rows),
        "path": str(out_jsonl),
    }
    summary_path = out_dir / "hf_branch_nucleus_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "layer_verdict": layer,
                "NoSearch": nos,
                "NeedSearch": need,
                "summary_path": str(summary_path),
            },
            indent=2,
        ),
        flush=True,
    )
    if own_model:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary


def phase_sample_hf(args: argparse.Namespace, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
    import torch

    out_dir = resolve(args.output_dir)
    source, by = _load_prompt_source(out_dir)
    if args.max_samples and args.max_samples > 0:
        # keep insertion order from sorted ids
        keep = sorted(by.keys())[: int(args.max_samples)]
        by = {k: by[k] for k in keep}

    model_path = resolve(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    own_model = model is None
    if model is None or tokenizer is None:
        model, tokenizer = _load_hf_model(model_path, device)

    labels = load_boundary(resolve(args.boundary_table))
    sid_meta = json.loads((out_dir / "sample_ids.json").read_text(encoding="utf-8"))
    frozen = {s["sample_id"]: s for s in sid_meta["samples"]}
    n = int(args.n_rollouts)
    T = float(args.temperature)
    max_new = int(args.max_new_tokens)
    top_ps = _parse_top_p_list(args.top_p_list)

    by_top: Dict[str, List[dict[str, Any]]] = {}
    for top_p in top_ps:
        tag = _top_p_tag(top_p)
        out_path = out_dir / f"dump_pathA_hf_top_p{tag}.jsonl"
        records: List[dict[str, Any]] = []
        with out_path.open("w", encoding="utf-8") as f:
            for qi, (sid, crow) in enumerate(sorted(by.items())):
                prompt_ids = list(crow["canonical_prompt_ids"])
                for k in range(n):
                    seed = int(args.seed) + int(round(top_p * 1e6)) + qi * 1009 + k * 17
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                    gen_kwargs: Dict[str, Any] = {
                        "max_new_tokens": max_new,
                        "do_sample": T > 0,
                        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    }
                    if T > 0:
                        gen_kwargs["temperature"] = T
                        gen_kwargs["top_p"] = float(top_p)
                    with torch.inference_mode():
                        out_ids = model.generate(inp, **gen_kwargs)
                    gen = out_ids[0, len(prompt_ids) :].tolist()
                    text = tokenizer.decode(gen, skip_special_tokens=True)
                    route = _normalize_route(text)
                    rec = {
                        "backend": "hf_exact_prompt_ids",
                        "audit_path": "A",
                        "sample_id": sid,
                        "boundary": frozen.get(sid, {}).get("boundary") or labels.get(sid),
                        "canonical_prompt_sha256": crow.get("canonical_prompt_sha256")
                        or _sha_ids(prompt_ids),
                        "canonical_prompt_ids_ref": True,
                        "rollout_k": k,
                        "seed": seed,
                        "sampling_temperature": T,
                        "sampling_top_p": float(top_p),
                        "first_generate_token_ids": gen,
                        "first_generate_text": text,
                        "first_generate_len": len(gen),
                        "route_first": route,
                        "action": (
                            "search"
                            if route == "search"
                            else ("internal" if route == "internal" else "other")
                        ),
                    }
                    records.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                print(
                    f"[A top_p={top_p}] {qi+1}/{len(by)} {sid} "
                    f"routes={[r['action'] for r in records[-n:]]}",
                    flush=True,
                )
        by_top[tag] = records
        print(json.dumps({"top_p": top_p, "n_rows": len(records), "path": str(out_path)}, indent=2))

    def _rate(recs: List[dict[str, Any]], boundary: str) -> dict[str, Any]:
        return _route_rate(recs, boundary)

    table: Dict[str, Any] = {}
    for top_p in top_ps:
        tag = _top_p_tag(top_p)
        table[f"top_p_{tag}"] = {
            "top_p": top_p,
            "NoSearch": _rate(by_top[tag], "NoSearch"),
            "NeedSearch": _rate(by_top[tag], "NeedSearch"),
            "All": _rate(by_top[tag], None),
        }

    # Compare 0.95 vs 1.0 if both present
    verdict = "SAMPLER_ALIGN_INCONCLUSIVE"
    t095 = next((table[k] for k in table if abs(table[k]["top_p"] - 0.95) < 1e-9), None)
    t100 = next((table[k] for k in table if abs(table[k]["top_p"] - 1.0) < 1e-9), None)
    if t095 and t100:
        p095 = float(t095["NoSearch"].get("p_internal") or 0.0)
        p100 = float(t100["NoSearch"].get("p_internal") or 0.0)
        if p095 < 0.05 and p100 >= 0.15:
            verdict = "NUCLEUS_TRUNCATION_CAUSES_ROUTING_COLLAPSE"
        elif p095 >= 0.15:
            verdict = "HF_TOP_P095_HAS_INTERNAL_SGLANG_MISMATCH_LIKELY"
        elif p100 < 0.05 and p095 < 0.05:
            verdict = "HF_ALIGNED_ALSO_ALL_SEARCH"
        else:
            verdict = "SAMPLER_ALIGN_INCONCLUSIVE"

    summary = {
        "purpose": "hf_exact_id_aligned_sampling",
        "git_commit": git_commit(),
        "model_path": str(model_path),
        "source_dump": str(source),
        "temperature": T,
        "n_rollouts": n,
        "max_new_tokens": max_new,
        "top_p_list": top_ps,
        "rates": table,
        "layer_verdict": verdict,
        "note": (
            "Old HF routing smoke used react_loop without top_p → transformers default "
            "top_p=1.0; train SGLang uses top_p=0.95."
        ),
    }
    summary_path = out_dir / "hf_sampler_align_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"layer_verdict": verdict, "rates": table, "summary_path": str(summary_path)}, indent=2))
    if own_model:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary


def phase_sampler_align(args: argparse.Namespace) -> None:
    """One GPU load: branch-nucleus then Exact-ID HF top_p ablation."""
    import torch

    out_dir = resolve(args.output_dir)
    model_path = resolve(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_hf_model(model_path, device)
    nuc = phase_branch_nucleus(args, model=model, tokenizer=tokenizer)
    samp = phase_sample_hf(args, model=model, tokenizer=tokenizer)
    combined = {
        "purpose": "routing_sampler_alignment_audit",
        "git_commit": git_commit(),
        "branch_nucleus": {
            "layer_verdict": nuc.get("layer_verdict"),
            "NoSearch": nuc.get("NoSearch"),
            "NeedSearch": nuc.get("NeedSearch"),
        },
        "hf_aligned_sample": {
            "layer_verdict": samp.get("layer_verdict"),
            "rates": samp.get("rates"),
        },
        "combined_read": (
            "If both say NUCLEUS_TRUNCATION_*: set rollout top_p=1.0 and rerun SGLang parity. "
            "If nucleus keeps internal but sample@.95 has internal while SGLang=0: TIM/sampler. "
            "If HF@.95 and @1.0 both ~0 internal: reconsider true collapse (still no Branching "
            "until TIM greedy parity)."
        ),
        "deferred": ["B-none", "Branching", "REINFORCE", "Mixed-action train"],
    }
    path = out_dir / "sampler_align_combined_summary.json"
    path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"combined_path": str(path), **combined}, indent=2), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def phase_greedy_tim(args: argparse.Namespace) -> None:
    """HF greedy parity + teacher-force TIM on Path-B SGLang tokens (no SGLang rerun yet)."""
    import torch
    import torch.nn.functional as F

    out_dir = resolve(args.output_dir)
    source, by = _load_prompt_source(out_dir)
    dump_b_path = out_dir / "dump_pathB_stop_current.jsonl"
    if not dump_b_path.is_file():
        raise SystemExit("need dump_pathB_stop_current.jsonl for SGLang first-gen tokens")
    b_rows: List[dict[str, Any]] = []
    with dump_b_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                b_rows.append(json.loads(line))

    model_path = resolve(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_hf_model(model_path, device)
    open_ids = _route_open_ids(tokenizer)
    search_ids = open_ids["<search>"]
    internal_ids = open_ids["<internal>"]
    lcp = _lcp_len(search_ids, internal_ids)
    search_branch = int(search_ids[lcp])
    internal_branch = int(internal_ids[lcp])

    labels = load_boundary(resolve(args.boundary_table))
    sid_meta = json.loads((out_dir / "sample_ids.json").read_text(encoding="utf-8"))
    frozen = {s["sample_id"]: s for s in sid_meta["samples"]}
    max_new = min(int(args.max_new_tokens), 16)

    greedy_by_sid: Dict[str, dict[str, Any]] = {}
    greedy_rows: List[dict[str, Any]] = []
    for sid, crow in sorted(by.items()):
        prompt_ids = list(crow["canonical_prompt_ids"])
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            out_ids = model.generate(
                inp,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            logits0 = model(inp).logits[0, -1].float()
        probs0 = F.softmax(logits0, dim=-1)
        gen = out_ids[0, len(prompt_ids) :].tolist()
        text = tokenizer.decode(gen, skip_special_tokens=True)
        route = _normalize_route(text)
        argmax_id = int(torch.argmax(logits0).item())
        rec = {
            "sample_id": sid,
            "boundary": frozen.get(sid, {}).get("boundary") or labels.get(sid),
            "backend": "hf_greedy_exact_prompt_ids",
            "temperature": 0.0,
            "top_p": 1.0,
            "first_generate_token_ids": gen,
            "first_token_id": int(gen[0]) if gen else None,
            "first_generate_text": text,
            "route_first": route,
            "action": (
                "search" if route == "search" else ("internal" if route == "internal" else "other")
            ),
            "argmax_token_id": argmax_id,
            "argmax_decoded": tokenizer.decode([argmax_id], skip_special_tokens=False),
            "p_search_branch": float(probs0[search_branch].item()),
            "p_internal_branch": float(probs0[internal_branch].item()),
            "search_branch_token_id": search_branch,
            "internal_branch_token_id": internal_branch,
            "argmax_is_search_branch": argmax_id == search_branch,
            "argmax_is_internal_branch": argmax_id == internal_branch,
        }
        greedy_by_sid[sid] = rec
        greedy_rows.append(rec)

    greedy_path = out_dir / "dump_pathA_hf_greedy.jsonl"
    with greedy_path.open("w", encoding="utf-8") as f:
        for r in greedy_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Teacher-force Path-B SGLang tokens under HF (TIM-lite without rollout logprobs).
    tim_rows: List[dict[str, Any]] = []
    tok0_match_greedy = 0
    n_b = 0
    for br in b_rows:
        sid = str(br["sample_id"])
        prompt_ids = list(br["canonical_prompt_ids"])
        sgl_ids = list(br.get("first_generate_token_ids") or [])
        if not sgl_ids:
            continue
        n_b += 1
        g = greedy_by_sid.get(sid)
        sgl_tok0 = int(sgl_ids[0])
        hf_tok0 = int(g["first_token_id"]) if g and g.get("first_token_id") is not None else None
        if hf_tok0 is not None and sgl_tok0 == hf_tok0:
            tok0_match_greedy += 1

        # HF logprobs for SGLang continuation (first up to 8 tokens)
        cont = sgl_ids[:8]
        full = torch.tensor([prompt_ids + cont], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(full).logits[0]
        start = len(prompt_ids) - 1
        token_lps: List[float] = []
        for j, tid in enumerate(cont):
            lp = F.log_softmax(logits[start + j].float(), dim=-1)[int(tid)].item()
            token_lps.append(float(lp))
        # At branch state (=prompt, lcp=0): mass on search/internal vs SGLang tok0
        with torch.inference_mode():
            logits_b = model(torch.tensor([prompt_ids], dtype=torch.long, device=device)).logits[0, -1].float()
        logp_b = F.log_softmax(logits_b, dim=-1)
        tim_rows.append(
            {
                "sample_id": sid,
                "boundary": frozen.get(sid, {}).get("boundary") or labels.get(sid) or br.get("boundary"),
                "sglang_route_first": br.get("route_first"),
                "sglang_tok0": sgl_tok0,
                "sglang_tok0_decoded": tokenizer.decode([sgl_tok0], skip_special_tokens=False),
                "hf_greedy_tok0": hf_tok0,
                "hf_greedy_route": g.get("action") if g else None,
                "tok0_matches_hf_greedy": bool(hf_tok0 is not None and sgl_tok0 == hf_tok0),
                "hf_logp_sglang_tok0": float(logp_b[sgl_tok0].item()),
                "hf_logp_search_branch": float(logp_b[search_branch].item()),
                "hf_logp_internal_branch": float(logp_b[internal_branch].item()),
                "hf_delta_logp_search_minus_internal": float(
                    logp_b[search_branch].item() - logp_b[internal_branch].item()
                ),
                "hf_logp_sglang_prefix_sum": float(sum(token_lps)),
                "hf_logp_sglang_tokens": token_lps,
            }
        )

    tim_path = out_dir / "hf_tim_on_sglang_pathB.jsonl"
    with tim_path.open("w", encoding="utf-8") as f:
        for r in tim_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    g_rate = _route_rate(greedy_rows, "NoSearch")
    g_rate_all = _route_rate(greedy_rows, None)
    # If HF greedy all search but stochastic HF has internal → mode is search; SGLang
    # failing to sample internal is still sampler pathology if rates differ wildly.
    nos_g = [r for r in greedy_rows if r.get("boundary") == "NoSearch"]
    frac_argmax_search = (
        sum(1 for r in nos_g if r.get("argmax_is_search_branch")) / len(nos_g) if nos_g else 0.0
    )
    frac_greedy_internal = float(g_rate.get("p_internal") or 0.0)

    # Binomial shock: SGLang PathB 0/80 vs HF@.95 ~0.28
    p_hf = 0.284  # from sampler-align NoSearch @0.95
    n_sgl = max(n_b, 1)
    # P(X=0) = (1-p)^n under iid
    p_zero = (1.0 - p_hf) ** n_sgl

    if frac_greedy_internal >= 0.15:
        # HF mode itself often internal but SGLang never → severe
        layer = "BACKEND_LOGIT_OR_GREEDY_MISMATCH"
    elif frac_argmax_search >= 0.90 and frac_greedy_internal < 0.05:
        # HF mode is search; stochastic HF explores internal; SGLang does not
        layer = "SGLANG_STOCHASTIC_SAMPLER_MISMATCH"
    else:
        layer = "SGLANG_SAMPLER_MISMATCH"

    # tok0 agreement between HF greedy and SGLang samples
    tok0_agree = (tok0_match_greedy / n_b) if n_b else 0.0

    summary = {
        "purpose": "hf_greedy_and_tim_lite_on_pathB",
        "git_commit": git_commit(),
        "model_path": str(model_path),
        "source_dump": str(source),
        "pathB_dump": str(dump_b_path),
        "search_branch_token_id": search_branch,
        "internal_branch_token_id": internal_branch,
        "lcp_len": lcp,
        "hf_greedy_NoSearch": g_rate,
        "hf_greedy_All": g_rate_all,
        "frac_NoSearch_argmax_is_search_branch": frac_argmax_search,
        "pathB_n": n_b,
        "pathB_tok0_agree_with_hf_greedy": tok0_agree,
        "binomial_P_zero_internal_under_hf095": {
            "p_hf_ref": p_hf,
            "n_sglang": n_sgl,
            "P_Xeq0": p_zero,
            "note": "Path B had 0 internal; under HF@.95 rate this is astronomically unlikely if same dist",
        },
        "layer_verdict": layer,
        "next": {
            "BACKEND_LOGIT_OR_GREEDY_MISMATCH": (
                "STOP train; run SGLang T=0 greedy dump; check weight sync / dtype"
            ),
            "SGLANG_STOCHASTIC_SAMPLER_MISMATCH": (
                "dump actual SGLang sampling_params + logprobs; TIM δ_t with rollout logprobs"
            ),
            "SGLANG_SAMPLER_MISMATCH": "SGLang T=0 greedy parity + sampling_params dump",
        }[layer],
        "paths": {"greedy": str(greedy_path), "tim": str(tim_path)},
    }
    summary_path = out_dir / "hf_greedy_tim_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def phase_score(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = resolve(args.output_dir)
    # Prefer Path C dump for canonical prompts; fall back to B variants.
    dump_c = out_dir / "dump_pathC.jsonl"
    dump_b_candidates = sorted(out_dir.glob("dump_pathB*.jsonl"))
    source = dump_c if dump_c.is_file() else (dump_b_candidates[0] if dump_b_candidates else None)
    if source is None or not source.is_file():
        raise SystemExit("need dump_pathC.jsonl or dump_pathB*.jsonl from print-cmd runs")
    by = _load_canonical_prompts(source)
    if not by:
        raise SystemExit(f"empty dump: {source}")

    model_path = resolve(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()

    open_ids = {
        tag: list(tokenizer.encode(tag, add_special_tokens=False)) for tag in OPEN_TAGS
    }
    # Opening-tag tokenization diagnostic (Qwen often fuses `search>`).
    open_tok_diag = {
        tag: {
            "token_ids": open_ids[tag],
            "n_tokens": len(open_ids[tag]),
            "decoded_pieces": [tokenizer.decode([t], skip_special_tokens=False) for t in open_ids[tag]],
        }
        for tag in OPEN_TAGS
    }
    labels = load_boundary(resolve(args.boundary_table))
    sid_meta = json.loads((out_dir / "sample_ids.json").read_text(encoding="utf-8"))
    frozen = {s["sample_id"]: s for s in sid_meta["samples"]}

    rows: List[dict[str, Any]] = []
    for sid, crow in by.items():
        prompt_ids = list(crow["canonical_prompt_ids"])
        L_s_sum, L_s_mean = _sequence_logprob(model, prompt_ids, open_ids["<search>"], device)
        L_i_sum, L_i_mean = _sequence_logprob(model, prompt_ids, open_ids["<internal>"], device)
        # First-token only (diagnostic; not the primary gate).
        L_s0, _ = _sequence_logprob(model, prompt_ids, open_ids["<search>"][:1], device)
        L_i0, _ = _sequence_logprob(model, prompt_ids, open_ids["<internal>"][:1], device)
        m = max(L_s_sum, L_i_sum)
        es = math.exp(L_s_sum - m)
        ei = math.exp(L_i_sum - m)
        p_int = ei / (ei + es)
        rows.append(
            {
                "sample_id": sid,
                "boundary": frozen.get(sid, {}).get("boundary") or labels.get(sid),
                "canonical_prompt_sha256": crow.get("canonical_prompt_sha256") or _sha_ids(prompt_ids),
                "canonical_prompt_len": len(prompt_ids),
                "open_tag_token_ids": open_ids,
                "L_search_sum": L_s_sum,
                "L_internal_sum": L_i_sum,
                "L_search_mean": L_s_mean,
                "L_internal_mean": L_i_mean,
                "L_search_first_tok": L_s0,
                "L_internal_first_tok": L_i0,
                "M_route": L_s_sum - L_i_sum,
                "p_tilde_internal": p_int,
                "source_dump": str(source),
            }
        )

    score_path = out_dir / "hf_route_scores.jsonl"
    with score_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stop_rep = _stop_tokenization_report(tokenizer)
    (out_dir / "stop_tokenization.json").write_text(
        json.dumps(stop_rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    nos = _p_int_stats(rows, "NoSearch")
    need = _p_int_stats(rows, "NeedSearch")
    collapse = bool(nos.get("n", 0) > 0 and nos["median"] < 0.01 and nos.get("frac_lt_0.01", 0) >= 0.80)
    support = bool(nos.get("n", 0) > 0 and ((nos["median"] >= 0.10) or (nos.get("frac_ge_0.10", 0) >= 0.25)))
    if collapse:
        layer1 = "ROOT_POLICY_COLLAPSE_LIKELY"
    elif support:
        layer1 = "BACKEND_MISMATCH_LIKELY"
    else:
        layer1 = "POLICY_ROUTE_INCONCLUSIVE"

    # Path C length contract interim from existing dump (first_gen only — cheap).
    length_interim: dict[str, Any] = {"status": "no_pathC_dump"}
    if dump_c.is_file():
        first_lens: List[int] = []
        routes = Counter()
        with dump_c.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ids = r.get("first_generate_token_ids") or []
                first_lens.append(len(ids))
                routes[r.get("route_first") or "none"] += 1
        length_interim = {
            "status": "pathC_first_gen_only",
            "n": len(first_lens),
            "first_generate_len_mean": (sum(first_lens) / len(first_lens)) if first_lens else 0.0,
            "first_generate_len_max": max(first_lens) if first_lens else 0,
            "first_generate_len_min": min(first_lens) if first_lens else 0,
            "route_first_counts": dict(routes),
            "note": (
                "Train metric response_length=2048/clip=1 is a SEPARATE multi-turn "
                "budget signal; short first_gen does NOT clear LENGTH_CONTRACT gate."
            ),
            "LENGTH_CONTRACT_GATE": "FAIL_PENDING_FIX",
            "train_metric_observed": {
                "response_length_mean": 2048.0,
                "clip_ratio": 1.0,
                "source": "PathC step logs",
            },
        }

    summary = {
        "purpose": "hf_exact_prompt_root_score",
        "git_commit": git_commit(),
        "model_path": str(model_path),
        "source_dump": str(source),
        "n_scored": len(rows),
        "open_tag_tokenization": open_tok_diag,
        "NoSearch": nos,
        "NeedSearch": need,
        "layer1_verdict": layer1,
        "gates": {
            "collapse_rule": "NoSearch median p_tilde_internal < 0.01 AND frac_lt_0.01 >= 0.80",
            "support_rule": "NoSearch median >= 0.10 OR frac_ge_0.10 >= 0.25",
        },
        "stop_tokenization": stop_rep,
        "length_contract_interim": length_interim,
        "next": {
            "ROOT_POLICY_COLLAPSE_LIKELY": "still run Path B stop ablation for LENGTH_CONTRACT; then Branching only after length gate",
            "BACKEND_MISMATCH_LIKELY": "Path B stop ablation REQUIRED before any Branching",
            "POLICY_ROUTE_INCONCLUSIVE": "inspect scores; then Path B",
        }[layer1],
    }
    summary_path = out_dir / "hf_route_score_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "n_scored": len(rows),
            "layer1_verdict": layer1,
            "NoSearch": nos,
            "NeedSearch": need,
            "summary_path": str(summary_path),
            "score_path": str(score_path),
            "LENGTH_CONTRACT": length_interim.get("LENGTH_CONTRACT_GATE"),
            "first_gen_len_mean": length_interim.get("first_generate_len_mean"),
        },
        indent=2,
    ))


def _p_int_stats(scores: List[dict[str, Any]], boundary: str) -> dict[str, Any]:
    vals = [float(r["p_tilde_internal"]) for r in scores if r.get("boundary") == boundary]
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)

    def pct(p: float) -> float:
        i = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return vals_sorted[i]

    return {
        "n": n,
        "median": pct(50),
        "p25": pct(25),
        "p75": pct(75),
        "frac_ge_0.10": sum(1 for v in vals if v >= 0.10) / n,
        "frac_lt_0.01": sum(1 for v in vals if v < 0.01) / n,
        "mean": sum(vals) / n,
    }


def _route_rate(rows: List[dict[str, Any]], boundary: Optional[str] = None) -> dict[str, Any]:
    xs = [r for r in rows if boundary is None or r.get("boundary") == boundary]
    n = len(xs)
    c = Counter(r.get("action") or r.get("route_first") for r in xs)
    return {
        "n": n,
        "p_search": (c.get("search", 0) / n) if n else 0.0,
        "p_internal": (c.get("internal", 0) / n) if n else 0.0,
        "counts": dict(c),
    }


def phase_aggregate(args: argparse.Namespace) -> None:
    out_dir = resolve(args.output_dir)
    labels = load_boundary(resolve(args.boundary_table))
    sid_meta = json.loads((out_dir / "sample_ids.json").read_text(encoding="utf-8"))
    frozen = {s["sample_id"]: s for s in sid_meta["samples"]}

    def load_jsonl(p: Path) -> List[dict[str, Any]]:
        if not p.is_file():
            return []
        rows = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        for r in rows:
            if "boundary" not in r or not r["boundary"]:
                r["boundary"] = frozen.get(r["sample_id"], {}).get("boundary") or labels.get(
                    r["sample_id"]
                )
            if "action" not in r:
                rf = r.get("route_first")
                r["action"] = (
                    "search"
                    if rf == "search"
                    else ("internal" if rf == "internal" else "other")
                )
        return rows

    dump_a = load_jsonl(out_dir / "dump_pathA_hf.jsonl")
    # Prefer dual top_p dumps from sampler-align.
    dump_a_095 = load_jsonl(out_dir / "dump_pathA_hf_top_p0p95.jsonl")
    dump_a_100 = load_jsonl(out_dir / "dump_pathA_hf_top_p1.jsonl")
    if not dump_a_100:
        dump_a_100 = load_jsonl(out_dir / "dump_pathA_hf_top_p1p0.jsonl")
    if dump_a_095:
        dump_a = dump_a_095  # canonical A for gate3 = train-parity top_p
    # Prefer stop_current as canonical Path B; merge none for length diagnostics only.
    dump_b = load_jsonl(out_dir / "dump_pathB_stop_current.jsonl")
    if not dump_b:
        dump_b = load_jsonl(out_dir / "dump_pathB.jsonl")
    dump_b_none = load_jsonl(out_dir / "dump_pathB_stop_none.jsonl")
    dump_c = load_jsonl(out_dir / "dump_pathC.jsonl")
    scores = load_jsonl(out_dir / "hf_route_scores.jsonl")
    score_summary = {}
    sp = out_dir / "hf_route_score_summary.json"
    if sp.is_file():
        score_summary = json.loads(sp.read_text(encoding="utf-8"))
    nuc_summary = {}
    npth = out_dir / "hf_branch_nucleus_summary.json"
    if npth.is_file():
        nuc_summary = json.loads(npth.read_text(encoding="utf-8"))
    align_summary = {}
    ap = out_dir / "hf_sampler_align_summary.json"
    if ap.is_file():
        align_summary = json.loads(ap.read_text(encoding="utf-8"))

    # Gate 1: prompt parity across B/C dumps (and within sample)
    prompt_ok = True
    prompt_issues: List[str] = []
    can_by: Dict[str, List[int]] = {}
    for rows, name in ((dump_b, "B"), (dump_c, "C")):
        for r in rows:
            sid = r["sample_id"]
            ids = r.get("canonical_prompt_ids")
            if not ids:
                prompt_ok = False
                prompt_issues.append(f"{name}:{sid}:missing_prompt_ids")
                continue
            if sid not in can_by:
                can_by[sid] = list(ids)
            elif list(ids) != can_by[sid]:
                prompt_ok = False
                prompt_issues.append(f"{name}:{sid}:prompt_ids_differ")

    gate1 = {
        "gate": "ROOT_PROMPT_PARITY_OK" if prompt_ok and can_by else "ROOT_PROMPT_PARITY_FAIL",
        "n_samples_with_prompt": len(can_by),
        "issues": prompt_issues[:20],
    }

    # Gate 2: policy support
    nos = _p_int_stats(scores, "NoSearch")
    support = False
    collapse_likely = False
    if nos.get("n", 0) > 0:
        support = (nos["median"] >= 0.10) or (nos["frac_ge_0.10"] >= 0.25)
        collapse_likely = (nos["median"] < 0.01) and (nos["frac_lt_0.01"] >= 0.80)
    gate2 = {
        "gate": (
            "INTERNAL_HAS_POLICY_SUPPORT"
            if support
            else ("ROOT_POLICY_COLLAPSE_LIKELY" if collapse_likely else "POLICY_ROUTE_INCONCLUSIVE")
        ),
        "NoSearch": nos,
        "NeedSearch": _p_int_stats(scores, "NeedSearch"),
    }

    # Gate 3: sampler / loop attribution
    rate_a = _route_rate(dump_a, "NoSearch")
    rate_b = _route_rate(dump_b, "NoSearch")
    rate_c = _route_rate(dump_c, "NoSearch")
    a_int = rate_a.get("p_internal", 0.0) > 0
    b_int = rate_b.get("p_internal", 0.0) > 0
    c_int = rate_c.get("p_internal", 0.0) > 0

    if gate1["gate"] == "ROOT_PROMPT_PARITY_FAIL":
        verdict = "ROOT_PROMPT_PARITY_FAIL"
    elif support and a_int and (not b_int):
        verdict = "SGLANG_SAMPLER_MISMATCH"
    elif support and a_int and b_int and (not c_int):
        verdict = "ECA_LOOP_OR_PARSER_MISMATCH"
    elif support and a_int and b_int and c_int:
        verdict = "PARITY_REPAIRED"
    elif collapse_likely and (not a_int) and (not b_int) and (not c_int):
        verdict = "TRUE_ROOT_ACTION_COLLAPSE"
    elif (not support) and (not a_int) and (not b_int) and (not c_int):
        # weak scores + all-search → treat as collapse for action
        verdict = "TRUE_ROOT_ACTION_COLLAPSE"
    else:
        verdict = "POLICY_ROUTE_INCONCLUSIVE"

    stop_path = out_dir / "stop_tokenization.json"
    stop_rep = json.loads(stop_path.read_text(encoding="utf-8")) if stop_path.is_file() else {}
    stop_risk = bool(stop_rep.get("last_token_collision"))

    def _len_stats(rows: List[dict[str, Any]], key: str = "first_generate_len") -> dict[str, Any]:
        vals = [int(r[key]) for r in rows if r.get(key) is not None]
        if not vals and rows:
            vals = [len(r.get("first_generate_token_ids") or []) for r in rows]
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "max": max(vals),
            "min": min(vals),
            "frac_ge_256": sum(1 for v in vals if v >= 256) / len(vals),
        }

    length_gate = {
        "status": "LENGTH_CONTRACT_FAIL_PENDING",
        "note": (
            "Hard gate before next train: Agent rollouts must NOT be 100% "
            "response_length clip_ratio=1. Path C train metrics already FAIL this."
        ),
        "pathC_train_metric": {"response_length_mean": 2048.0, "clip_ratio": 1.0},
        "pathB_stop_current_first_gen": _len_stats(dump_b),
        "pathB_stop_none_first_gen": _len_stats(dump_b_none),
        "pathC_first_gen": _len_stats(dump_c),
    }

    summary = {
        "purpose": "routing_worker_mismatch_audit",
        "git_commit": git_commit(),
        "output_dir": str(out_dir),
        "gate1_root_prompt_parity": gate1,
        "gate2_policy_route_support": gate2,
        "layer1_from_score": score_summary.get("layer1_verdict"),
        "branch_nucleus": nuc_summary.get("layer_verdict"),
        "hf_aligned_sample": align_summary.get("layer_verdict"),
        "hf_aligned_rates": align_summary.get("rates"),
        "gate3_rates_NoSearch": {
            "A_hf_top_p0.95": rate_a,
            "A_hf_top_p1.0": _route_rate(dump_a_100, "NoSearch") if dump_a_100 else {},
            "B_bare": rate_b,
            "C_eca": rate_c,
        },
        "stop_handling": {
            "status": "STOP_HANDLING_RISK" if stop_risk else "STOP_HANDLING_OK",
            "note": "B-none deferred; first-gen stop OK on Path B-current",
            "report": stop_rep,
        },
        "length_contract": length_gate,
        "verdict": (
            align_summary.get("layer_verdict")
            or nuc_summary.get("layer_verdict")
            or verdict
        ),
        "verdict_legacy_abc": verdict,
        "n_dump": {
            "A_095": len(dump_a_095) if dump_a_095 else len(dump_a),
            "A_100": len(dump_a_100),
            "B_current": len(dump_b),
            "B_none": len(dump_b_none),
            "C": len(dump_c),
            "scores": len(scores),
        },
        "next": {
            "ROOT_PROMPT_PARITY_FAIL": "fix Eca/HF prompt path; do not claim collapse",
            "SGLANG_SAMPLER_MISMATCH": "greedy parity + TIM logprob; do not Branching",
            "ECA_LOOP_OR_PARSER_MISMATCH": "fix route parser / stop interaction",
            "TRUE_ROOT_ACTION_COLLAPSE": (
                "only after HF@.95 and @1.0 both ~0 AND TIM greedy OK"
            ),
            "PARITY_REPAIRED": "reconsider natural Mixed-action GRPO after length gate",
            "POLICY_ROUTE_INCONCLUSIVE": "inspect sampler-align summaries",
            "NUCLEUS_TRUNCATION_CAUSES_ROUTING_COLLAPSE": (
                "set rollout top_p=1.0; rerun SGLang parity 32×4; no Branching"
            ),
            "HF_TOP_P095_HAS_INTERNAL_SGLANG_MISMATCH_LIKELY": (
                "greedy parity + TIM δ_t on route tokens; no Branching"
            ),
            "HF_ALIGNED_ALSO_ALL_SEARCH": (
                "revisit true collapse only after TIM; still no Branching yet"
            ),
            "NUCLEUS_DOES_NOT_EXPLAIN_COLLAPSE": "run Exact-ID sample / TIM next",
            "SAMPLER_ALIGN_INCONCLUSIVE": "inspect dumps; maybe raise n_rollouts",
        }.get(
            align_summary.get("layer_verdict")
            or nuc_summary.get("layer_verdict")
            or verdict,
            "inspect",
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"verdict": verdict, "gate1": gate1["gate"], "gate2": gate2["gate"]}, indent=2))
    print(f"[mismatch] wrote {out_dir / 'summary.json'}", flush=True)


def main() -> None:
    args = parse_args()
    if args.config:
        cfg = json.loads(resolve(args.config).read_text(encoding="utf-8"))
        for k, v in cfg.items():
            if hasattr(args, k) and v is not None:
                setattr(args, k, v)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if args.phase == "prepare":
        phase_prepare(args)
    elif args.phase == "print-cmd":
        phase_print_cmd(args)
    elif args.phase == "score":
        phase_score(args)
    elif args.phase == "branch-nucleus":
        phase_branch_nucleus(args)
    elif args.phase == "sample-hf":
        phase_sample_hf(args)
    elif args.phase == "sampler-align":
        phase_sampler_align(args)
    elif args.phase == "greedy-tim":
        phase_greedy_tim(args)
    else:
        phase_aggregate(args)


if __name__ == "__main__":
    main()
