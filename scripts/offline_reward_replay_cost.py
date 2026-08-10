#!/usr/bin/env python3
"""Phase 3D0: offline Uniform Cost λ sweep (CPU). Does NOT train.

For each calib sample build paired trajectories:
  I-group: internal✓ (N=0) vs search✓+gold evidence (N=1)
  S-group: internal✗ (N=0) vs search✓+gold evidence (N=1)

Sweep λ_s and report prefer-internal on I / prefer-search on S.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.rl.rewards_evidence import compute_score

# Include 0.50: with perfect EvidF1=1, need λ_s>0.5 to beat unnecessary search.
LAMBDAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


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


def _gold(sample: Dict[str, Any]) -> str:
    g = sample.get("gold_answers") or sample.get("answers") or []
    if isinstance(g, str):
        return g
    if isinstance(g, list) and g:
        return str(g[0])
    return "unknown"


def _wrong_answer(gold: str) -> str:
    return f"NOT_{gold}" if gold else "WRONG_ANSWER"


def _typical_sf(sf: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep ~2/3 gold SF so EvidF1≈0.67 (matches 3C-GEN), not perfect farming."""
    if len(sf) <= 1:
        return list(sf)
    k = max(1, int(round(len(sf) * 2 / 3)))
    return list(sf[:k])


def make_pair(sample: Dict[str, Any], group: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    gold = _gold(sample)
    sf = _sf_min(sample)
    sf_pred = _typical_sf(sf)
    q = sample.get("question") or "query"
    sid = sample["sample_id"]
    gt = {"target": [gold], "supporting_facts": sf}

    if group == "I":
        internal = (
            "<internal>\nUse internal knowledge.\n</internal>\n"
            f"<answer>\n{gold}\n</answer>"
        )
    else:
        internal = (
            "<internal>\nUse internal knowledge.\n</internal>\n"
            f"<answer>\n{_wrong_answer(gold)}\n</answer>"
        )

    search = (
        f"<search>\n{q}\n</search>\n"
        f"{_ev_block(sf_pred)}"
        f"<answer>\n{gold}\n</answer>"
    )

    a = {
        "role": "internal",
        "group": group,
        "sample_id": sid,
        "solution_str": internal,
        "search_count": 0,
        "ground_truth": gt,
    }
    b = {
        "role": "search",
        "group": group,
        "sample_id": sid,
        "solution_str": search,
        "search_count": 1,
        "ground_truth": gt,
    }
    return a, b


def score_traj(traj: Dict[str, Any], lam: float) -> Dict[str, Any]:
    extra = {
        "sample_id": traj["sample_id"],
        "search_count": traj["search_count"],
        "supporting_facts": traj["ground_truth"].get("supporting_facts"),
        "reward_weights": {
            "answer_weight": 1.0,
            "evidence_weight": 0.5,
            "format_weight": 0.1,
            "search_cost_weight": lam,
            "duplicate_weight": 0.0,
        },
    }
    out = compute_score(
        solution_str=traj["solution_str"],
        ground_truth=traj["ground_truth"],
        extra_info=extra,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--calib-dir",
        type=Path,
        default=REPO / "data/rl/calib_cost_lambda_512",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "results/11_sweep_offline_cost_lambda",
    )
    args = ap.parse_args()

    samples = {r["sample_id"]: r for r in _load_jsonl(args.calib_dir / "calib_samples.jsonl")}
    labels = _load_jsonl(args.calib_dir / "labels.jsonl")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for lab in labels:
        sid = lab["sample_id"]
        sample = samples[sid]
        pairs.append(make_pair(sample, lab["group"]))

    by_lambda: Dict[str, Any] = {}
    scored_rows: List[Dict[str, Any]] = []

    for lam in LAMBDAS:
        n_I = n_S = 0
        pref_I = pref_S = 0
        deltas_I: List[float] = []
        deltas_S: List[float] = []
        for a, b in pairs:
            sa = score_traj(a, lam)
            sb = score_traj(b, lam)
            ra, rb = float(sa["score"]), float(sb["score"])
            row = {
                "lambda": lam,
                "sample_id": a["sample_id"],
                "group": a["group"],
                "R_internal": ra,
                "R_search": rb,
                "delta_search_minus_internal": rb - ra,
                "em_internal": sa["em"],
                "em_search": sb["em"],
                "ev_search": sb["evidence_f1"],
                "prefer_internal": ra > rb,
                "prefer_search": rb > ra,
            }
            scored_rows.append(row)
            if a["group"] == "I":
                n_I += 1
                if ra > rb:
                    pref_I += 1
                deltas_I.append(rb - ra)
            else:
                n_S += 1
                if rb > ra:
                    pref_S += 1
                deltas_S.append(rb - ra)

        key = f"{lam:.2f}"
        by_lambda[key] = {
            "lambda": lam,
            "n_I": n_I,
            "n_S": n_S,
            "prefer_internal_on_I": round(pref_I / max(n_I, 1), 4),
            "prefer_search_on_S": round(pref_S / max(n_S, 1), 4),
            "mean_delta_S_minus_I_on_I": round(sum(deltas_I) / max(n_I, 1), 4),
            "mean_delta_S_minus_I_on_S": round(sum(deltas_S) / max(n_S, 1), 4),
            # tie counts as not prefer
            "tie_rate_I": round(
                sum(1 for d in deltas_I if abs(d) < 1e-9) / max(n_I, 1), 4
            ),
            "tie_rate_S": round(
                sum(1 for d in deltas_S if abs(d) < 1e-9) / max(n_S, 1), 4
            ),
        }

    # Strict: prefer_I>=0.95 & prefer_S>=0.95
    strict = None
    for lam in LAMBDAS:
        m = by_lambda[f"{lam:.2f}"]
        if m["prefer_internal_on_I"] >= 0.95 and m["prefer_search_on_S"] >= 0.95:
            strict = lam
            break
    # Operational 3D1 start: first λ with majority I prefer-internal + meanΔ<0 + S intact
    operational = None
    for lam in LAMBDAS:
        m = by_lambda[f"{lam:.2f}"]
        if (
            m["prefer_internal_on_I"] >= 0.50
            and m["mean_delta_S_minus_I_on_I"] < 0
            and m["prefer_search_on_S"] >= 0.95
        ):
            operational = lam
            break
    chosen = operational if operational is not None else strict
    if chosen is None:
        chosen = 0.40

    recommendation = {
        "chosen_lambda_s": chosen,
        "operational_lambda_s": operational,
        "strict_lambda_s": strict,
        "rule": (
            "operational = smallest λ with prefer_I>=0.5, meanΔ_I<0, prefer_S>=0.95; "
            "strict = prefer_I>=0.95 & prefer_S>=0.95"
        ),
        "rationale": by_lambda[f"{chosen:.2f}"],
        "note": (
            "Break-even ≈ 0.5 * EvidF1. With typical EvidF1≈0.67, λ≈0.33 ties; "
            "λ=0.40 flips majority I pairs; λ=0.50 flips all. "
            "λ∈{0.05,0.10,0.20,0.30} do NOT stop evidence-farming on I."
        ),
    }

    summary = {
        "phase": "3D0",
        "formula": "R = EM + 0.5*EvidF1 + 0.1*Format - λ_s * N_search",
        "n_pairs": len(pairs),
        "lambdas": LAMBDAS,
        "by_lambda": by_lambda,
        "recommendation": recommendation,
        "next": "3D1 Uniform Cost GRPO fresh from SFT-v1, STEPS=400, λ_s=chosen",
    }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out_dir / "scored.jsonl").open("w", encoding="utf-8") as f:
        for r in scored_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (args.out_dir / "by_lambda.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(by_lambda[f"{LAMBDAS[0]:.2f}"].keys()))
        w.writeheader()
        for lam in LAMBDAS:
            w.writerow(by_lambda[f"{lam:.2f}"])

    print(json.dumps({"by_lambda": by_lambda, "recommendation": recommendation}, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
