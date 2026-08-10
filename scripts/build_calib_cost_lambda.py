#!/usr/bin/env python3
"""Build Phase 3D0 calibration slice (CPU): stratified I/S, disjoint from smoke128+val200."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set

REPO = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ids_txt(path: Path) -> Set[str]:
    if not path.is_file():
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def _direct_ok(lab: Dict[str, Any]) -> bool:
    return bool(lab.get("direct_correct")) or float(lab.get("exact_match") or 0) >= 1.0 - 1e-9


def _oracle_ok(row: Dict[str, Any]) -> bool:
    em = row.get("exact_match")
    if em is None and isinstance(row.get("metrics"), dict):
        em = row["metrics"].get("exact_match")
    return float(em or 0) >= 1.0 - 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-pool",
        type=Path,
        default=REPO / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl",
    )
    ap.add_argument(
        "--direct-labels",
        type=Path,
        default=REPO
        / "results/phase2e1_direct_label_n8000_20260807_202826_phase2e1/labels.jsonl",
    )
    ap.add_argument(
        "--oracle-metrics",
        type=Path,
        default=REPO
        / "results/phase2e1_base_oracle_n8000_20260807_205154/merged/metrics.json",
    )
    ap.add_argument(
        "--smoke-train-ids",
        type=Path,
        default=REPO / "data/rl/train_smoke_128/train_ids.txt",
    )
    ap.add_argument(
        "--smoke-manifest",
        type=Path,
        default=REPO / "data/rl/train_smoke_128/manifest.json",
    )
    ap.add_argument(
        "--val200-ids",
        type=Path,
        default=REPO / "data/eval/hotpotqa_200_ids.txt",
    )
    ap.add_argument("--n-internal", type=int, default=256)
    ap.add_argument("--n-search", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data/rl/calib_cost_lambda_512",
    )
    args = ap.parse_args()

    pool = {str(r["sample_id"]): r for r in _load_jsonl(args.train_pool)}
    direct = {str(r["sample_id"]): r for r in _load_jsonl(args.direct_labels)}
    oracle_rows = json.loads(args.oracle_metrics.read_text())
    oracle = {str(r["sample_id"]): r for r in oracle_rows}

    exclude = _ids_txt(args.smoke_train_ids) | _ids_txt(args.val200_ids)
    if args.smoke_manifest.is_file():
        man = json.loads(args.smoke_manifest.read_text())
        exclude |= set(man.get("train_ids") or [])
        exclude |= set(man.get("val_ids") or [])

    I, S = [], []
    for sid, sample in pool.items():
        if sid in exclude:
            continue
        if sid not in direct or sid not in oracle:
            continue
        # need SF for evidence synthesis
        if not sample.get("supporting_facts"):
            continue
        d_ok = _direct_ok(direct[sid])
        o_ok = _oracle_ok(oracle[sid])
        if d_ok:
            I.append(sid)
        elif o_ok:
            S.append(sid)

    rng = random.Random(args.seed)
    rng.shuffle(I)
    rng.shuffle(S)
    if len(I) < args.n_internal or len(S) < args.n_search:
        raise SystemExit(
            f"insufficient pool after exclude: I={len(I)} S={len(S)} "
            f"need {args.n_internal}/{args.n_search}"
        )
    pick_I = I[: args.n_internal]
    pick_S = S[: args.n_search]
    picked = pick_I + pick_S
    assert not (set(picked) & exclude), "leak into smoke/val200"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.out_dir / "labels.jsonl"
    samples_path = args.out_dir / "calib_samples.jsonl"
    ids_path = args.out_dir / "calib_ids.txt"

    with labels_path.open("w", encoding="utf-8") as lf, samples_path.open(
        "w", encoding="utf-8"
    ) as sf, ids_path.open("w", encoding="utf-8") as idf:
        for sid in pick_I + pick_S:
            group = "I" if sid in pick_I else "S"
            lab = {
                "sample_id": sid,
                "group": group,
                "direct_correct": _direct_ok(direct[sid]),
                "oracle_em": float(
                    (oracle[sid].get("metrics") or oracle[sid]).get("exact_match")
                    if isinstance(oracle[sid].get("metrics"), dict)
                    else oracle[sid].get("exact_match")
                    or 0
                ),
            }
            lf.write(json.dumps(lab, ensure_ascii=False) + "\n")
            sf.write(json.dumps(pool[sid], ensure_ascii=False) + "\n")
            idf.write(sid + "\n")

    manifest = {
        "phase": "3D0",
        "seed": args.seed,
        "n_internal": len(pick_I),
        "n_search": len(pick_S),
        "n_total": len(picked),
        "exclude_n": len(exclude),
        "pool_I_available": len(I),
        "pool_S_available": len(S),
        "disjoint_smoke128_val200": True,
        "note": "dev calibration only; not Phase-4 final test",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
