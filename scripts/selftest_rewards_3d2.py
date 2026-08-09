#!/usr/bin/env python3
"""CPU self-test for Phase 3D2 reward gating (no GPU / no train)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    from src.rl import rewards_3d2 as R

    # Reset module cache between table swaps.
    R._PINT_CACHE_KEY = None
    R._PINT_CACHE = {}

    with tempfile.TemporaryDirectory() as td:
        table = Path(td) / "pint.json"
        table.write_text(
            json.dumps(
                {
                    "q_easy": 1.0,
                    "q_hard": 0.0,
                    "q_mid": 0.5,
                }
            )
        )
        os.environ["ECA_PINT_TABLE"] = str(table)
        os.environ["ECA_EVIDENCE_WEIGHT"] = "0.5"
        os.environ["ECA_SEARCH_COST_WEIGHT"] = "0.40"
        os.environ["ECA_PINT_DEFAULT"] = "0.0"
        os.environ["ECA_PINT_STRICT"] = "0"  # selftest may use default

        def score(sid: str, text: str, gold: str = "Paris"):
            return R.compute_score(
                solution_str=text,
                ground_truth=gold,
                extra_info={
                    "sample_id": sid,
                    "supporting_facts": [{"title": "France", "sentence_id": 0}],
                },
            )

        ev = (
            "<evidence>\n"
            "[document_id=d0 | title=France | sentence_id=0]\n"
            "Paris is the capital.\n"
            "</evidence>\n"
        )
        # Perfect answer + evidence + one search
        good = "<search>capital</search>\n" + ev + "<answer>Paris</answer>"
        # Easy: p_int=1 → evidence gated to 0, full search cost
        s_easy = score("q_easy", good)
        assert s_easy["p_int"] == 1.0
        assert abs(s_easy["eff_evidence_weight"] - 0.0) < 1e-9
        assert abs(s_easy["eff_search_cost_weight"] - 0.40) < 1e-9
        assert abs(s_easy["evidence_f1"] - 1.0) < 1e-9, s_easy
        # R ≈ 1.0 + 0 + 0.1 - 0.40 = 0.70
        assert abs(s_easy["score"] - 0.70) < 1e-6, s_easy["score"]

        # Hard: p_int=0 → full evidence, zero search cost
        s_hard = score("q_hard", good)
        assert s_hard["p_int"] == 0.0
        assert abs(s_hard["eff_evidence_weight"] - 0.5) < 1e-9
        assert abs(s_hard["eff_search_cost_weight"] - 0.0) < 1e-9
        assert abs(s_hard["evidence_f1"] - 1.0) < 1e-9, s_hard
        # R ≈ 1 + 0.5*1 + 0.1 - 0 = 1.6
        assert abs(s_hard["score"] - 1.6) < 1e-6, s_hard["score"]

        # Mid + no search → no cost even if p_int>0
        nos = ev + "<answer>Paris</answer>"
        s_mid = score("q_mid", nos)
        assert s_mid["search_indicator"] == 0.0
        assert abs(s_mid["eff_search_cost_weight"] - 0.20) < 1e-9  # 0.4*0.5
        # cost term 0 because not searched
        expected = 1.0 + 0.5 * 0.5 * 1.0 + 0.1 * 1.0  # 1.35
        assert abs(s_mid["score"] - expected) < 1e-6, (s_mid["score"], expected)

        # Missing sid uses default 0.0 when STRICT=0
        s_miss = score("q_unknown", good)
        assert s_miss["p_int"] == 0.0

        # STRICT=1 must raise on missing
        os.environ["ECA_PINT_STRICT"] = "1"
        R._PINT_CACHE_KEY = None
        R._PINT_CACHE = {}
        R._LOGGED_HASH = False
        try:
            score("q_unknown", good)
            raise AssertionError("STRICT should raise on missing sample_id")
        except KeyError:
            pass

    print("selftest_rewards_3d2: PASS")


if __name__ == "__main__":
    main()
