#!/usr/bin/env python3
"""Offline Evidence reward unit test (CPU). Hard gate before 3C GRPO.

Primary: smoke128 train samples + synthetic evidence variants (perfect/half/none).
Optional: Phase-3A traces if sample_id resolves in a pool.
PASS requires scorer distinguishability + nonzero evidence on injected gold SF.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.rl.rewards_3c import compute_score


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sf_min(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for sf in sample.get("supporting_facts") or []:
        title = sf.get("title")
        sid = sf.get("sentence_id", sf.get("sent_id"))
        if title is None or sid is None:
            continue
        out.append({"title": str(title), "sentence_id": int(sid)})
    return out


def _ev_block(pairs: Sequence[Dict[str, Any]]) -> str:
    lines = [
        f"[document_id=d | title={p['title']} | sentence_id={p['sentence_id']}]\n."
        for p in pairs
    ]
    return "<evidence>\n" + "\n\n".join(lines) + "\n</evidence>\n"


def _load_smoke_samples(parquet: Path, pool: Path, ids_txt: Path) -> List[Dict[str, Any]]:
    by_id = {str(r["sample_id"]): r for r in _load_jsonl(pool)}
    ids = [ln.strip() for ln in ids_txt.read_text().splitlines() if ln.strip()]
    # Prefer pool (has SF+contexts); fall back to parquet ground_truth only
    samples = []
    for sid in ids:
        if sid in by_id:
            samples.append(by_id[sid])
            continue
    if samples:
        return samples
    # parquet fallback
    import datasets

    ds = datasets.Dataset.from_parquet(str(parquet))
    out = []
    for row in ds:
        gt = row["reward_model"]["ground_truth"]
        out.append(
            {
                "sample_id": row["extra_info"]["sample_id"],
                "gold_answers": gt.get("target") or [],
                "supporting_facts": gt.get("supporting_facts") or [],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-pool",
        type=Path,
        default=REPO / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl",
    )
    ap.add_argument(
        "--train-parquet",
        type=Path,
        default=REPO / "data/rl/grpo_smoke_128/train.parquet",
    )
    ap.add_argument(
        "--train-ids",
        type=Path,
        default=REPO / "data/rl/grpo_smoke_128/train_ids.txt",
    )
    ap.add_argument("--max-samples", type=int, default=48)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "results/phase3c_offline_reward_replay/summary.json",
    )
    args = ap.parse_args()

    # --- A) Hand-crafted distinguishability ---
    gold_sf = [
        {"title": "DocA", "sentence_id": 1},
        {"title": "DocB", "sentence_id": 3},
    ]
    gt0 = {"target": ["Paris"], "supporting_facts": gold_sf}
    syn_cases = [
        ("perfect", _ev_block(gold_sf) + "<answer>London</answer>", 1.0, 0.6),
        ("half", _ev_block([gold_sf[0], {"title": "DocC", "sentence_id": 4}]) + "<answer>London</answer>", 0.5, 0.35),
        ("none", "<answer>Paris</answer>", 0.0, 1.1),
        ("malformed", "<evidence>oops</evidence>\n<answer>Paris</answer>", 0.0, 1.1),
    ]
    syn_rows = []
    syn_ok = True
    for name, text, exp_f1, exp_total in syn_cases:
        r = compute_score(solution_str=text, ground_truth=gt0, extra_info={})
        ok = abs(r["evidence_f1"] - exp_f1) < 1e-6 and abs(r["score"] - exp_total) < 1e-6
        syn_ok = syn_ok and ok
        syn_rows.append(
            {"name": name, "evidence_f1": r["evidence_f1"], "total": r["score"], "expect_f1": exp_f1, "ok": ok}
        )

    # --- B) Smoke128 gold-SF injection replay ---
    samples = _load_smoke_samples(args.train_parquet, args.train_pool, args.train_ids)[: args.max_samples]
    scored = []
    for sample in samples:
        sf = _sf_min(sample)
        if len(sf) < 1:
            continue
        gt = {"target": list(sample.get("gold_answers") or []), "supporting_facts": sf}
        sid = str(sample["sample_id"])
        variants = {
            "perfect_wrong_ans": _ev_block(sf) + "<answer>WRONG_ANSWER_XYZ</answer>",
            "none_correct_ans": f"<answer>{(sample.get('gold_answers') or ['x'])[0]}</answer>",
            "half": _ev_block(sf[: max(1, len(sf) // 2)]) + "<answer>WRONG_ANSWER_XYZ</answer>",
        }
        for vname, text in variants.items():
            r = compute_score(solution_str=text, ground_truth=gt, extra_info={"sample_id": sid})
            scored.append({"sample_id": sid, "variant": vname, **{k: r[k] for k in (
                "answer_reward", "evidence_reward", "format_reward", "total_reward", "evidence_f1"
            )}})

    perfect = [x for x in scored if x["variant"] == "perfect_wrong_ans"]
    none_v = [x for x in scored if x["variant"] == "none_correct_ans"]
    half_v = [x for x in scored if x["variant"] == "half"]

    perfect_mean = sum(x["evidence_reward"] for x in perfect) / max(1, len(perfect))
    none_mean = sum(x["evidence_reward"] for x in none_v) / max(1, len(none_v))
    half_mean = sum(x["evidence_reward"] for x in half_v) / max(1, len(half_v))
    # Discrimination: perfect >> none; half in between (or == perfect if |sf|==1)
    pass_disc = perfect_mean > 0.8 and none_mean < 0.05 and half_mean >= 0.2
    pass_n = len(perfect) >= 20
    # Group std simulation: same wrong answer, varying evidence → nonzero group std
    group_stds = []
    by_sid: Dict[str, List[float]] = {}
    for x in scored:
        if x["variant"] in ("perfect_wrong_ans", "half", "none_correct_ans"):
            # only wrong-ans variants for pure evidence discrimination on same answer=0
            if x["variant"] == "none_correct_ans":
                continue
            by_sid.setdefault(x["sample_id"], []).append(x["total_reward"])
    # rebuild groups with perfect + half + empty-ev wrong ans
    by_sid = {}
    for sample in samples[:40]:
        sf = _sf_min(sample)
        if not sf:
            continue
        gt = {"target": list(sample.get("gold_answers") or []), "supporting_facts": sf}
        sid = str(sample["sample_id"])
        texts = [
            _ev_block(sf) + "<answer>WRONG</answer>",
            _ev_block(sf[:1]) + "<answer>WRONG</answer>",
            "<answer>WRONG</answer>",
            _ev_block(sf)[:20] + "<answer>WRONG</answer>",  # malformed-ish
        ]
        totals = [
            compute_score(solution_str=t, ground_truth=gt, extra_info={"sample_id": sid})["score"]
            for t in texts
        ]
        if len(totals) >= 2:
            group_stds.append(st.pstdev(totals))
    zero_std_rate = sum(1 for s in group_stds if s <= 1e-6) / max(1, len(group_stds))
    pass_group = zero_std_rate < 0.5 and (sum(group_stds) / max(1, len(group_stds))) > 0.05

    passed = syn_ok and pass_n and pass_disc and pass_group

    summary = {
        "pass": passed,
        "gates": {
            "synthetic_distinguish": syn_ok,
            "n_smoke_scored": pass_n,
            "perfect_gt_none": pass_disc,
            "group_std_from_evidence": pass_group,
        },
        "synthetic": syn_rows,
        "n_smoke_samples": len(samples),
        "n_scored_rows": len(scored),
        "evidence_perfect_mean": perfect_mean,
        "evidence_half_mean": half_mean,
        "evidence_none_mean": none_mean,
        "sim_group_reward_std_mean": (sum(group_stds) / len(group_stds)) if group_stds else None,
        "sim_zero_std_group_rate": zero_std_rate,
        "lambda_e": 0.5,
        "formula": "R = answer + 0.5*evidence_f1 + 0.1*format",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out.parent / "scored.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in scored[:200]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit("OFFLINE_REWARD_REPLAY_FAIL")
    print("OFFLINE_REWARD_REPLAY_PASS")


if __name__ == "__main__":
    main()
