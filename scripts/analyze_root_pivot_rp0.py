#!/usr/bin/env python3
"""Hard-gate Root-Pivot counterfactuals; never infer cosine across different rollouts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

BASE = {"NeedSearch": 1.4722222023540072, "NoSearch": 0.8636363704096187}


def load(path: Path):
    return json.loads(path.read_text())


def trajectory_signature(path: Path) -> Counter:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return Counter((str(row["sample_id"]), str(row["trajectory_sha256"])) for row in rows)


def branch(root: Path, subset: str, mode: str):
    path = root / f"{subset}_{mode}"
    return load(path / "branch_summary.json"), trajectory_signature(path / "trajectories.jsonl")


def cosine(gt: float, gr: float, gj: float, beta: float) -> float:
    denom = 2.0 * beta * gt * gr
    if denom <= 0:
        raise ValueError("zero gradient norm makes cosine undefined")
    value = (gj * gj - gt * gt - (beta * gr) ** 2) / denom
    if not -1.05 <= value <= 1.05:
        raise ValueError(f"invalid norm-derived cosine={value}; runs are not comparable")
    return max(-1.0, min(1.0, value))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/21_root_pivot/rp0")
    p.add_argument("--beta-only", action="store_true")
    args = p.parse_args()
    root = Path(args.root)
    task, _ = branch(root, "all", "task_only")
    route, _ = branch(root, "all", "route_only")
    beta = float(task["gradient_norm"]) / float(route["gradient_norm"])
    if args.beta_only:
        print(f"{beta:.12g}")
        return

    groups = {}
    identical = True
    for subset in ("all", "need", "no"):
        t, ts = branch(root, subset, "task_only")
        r, rs = branch(root, subset, "route_only")
        j, js = branch(root, subset, "joint")
        same = ts == rs == js
        identical &= same
        groups[subset] = {
            "trajectory_identical": same,
            "n_trajectories": sum(ts.values()),
            "g_task": t["gradient_norm"],
            "g_route": r["gradient_norm"],
            "g_joint": j["gradient_norm"],
            "cosine_task_route": cosine(
                float(t["gradient_norm"]), float(r["gradient_norm"]),
                float(j["gradient_norm"]), beta,
            ) if same else None,
        }

    route_margin = route["route_margin"]
    joint, _ = branch(root, "all", "joint")
    joint_margin = joint["route_margin"]
    route_delta = {k: float(route_margin[k]) - BASE[k] for k in BASE}
    joint_delta = {k: float(joint_margin[k]) - BASE[k] for k in BASE}
    direction_pass = (
        route_delta["NoSearch"] < 0
        and route_delta["NeedSearch"] > 0
        and joint_delta["NoSearch"] < 0
        and joint_delta["NeedSearch"] >= 0
    )
    passed = identical and direction_pass and math.isfinite(beta) and beta > 0
    out = {
        "gate": "RP0_PASS" if passed else "RP0_FAIL",
        "beta": beta,
        "beta_definition": "||g_task||/||g_route|| on balanced all batch; fixed once",
        "baseline_route_margin": BASE,
        "route_only_delta": route_delta,
        "joint_delta": joint_delta,
        "trajectory_identical_all_groups": identical,
        "direction_pass": direction_pass,
        "groups": groups,
    }
    (root / "rp0_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
