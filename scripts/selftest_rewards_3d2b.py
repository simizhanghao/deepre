#!/usr/bin/env python3
"""CPU self-test for Phase 3D2b boundary-aware reward (no GPU / no train)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    from src.rl import rewards_3d2b as R

    R._BOUNDARY_CACHE_KEY = None
    R._BOUNDARY_CACHE = {}
    R._LOGGED_HASH = False

    with tempfile.TemporaryDirectory() as td:
        table = Path(td) / "boundary.json"
        table.write_text(
            json.dumps(
                {
                    "boundary": {
                        "q_no": "NoSearch",
                        "q_need": "NeedSearch",
                        "q_und": "Undetermined",
                    }
                }
            )
        )
        os.environ["ECA_BOUNDARY_TABLE"] = str(table)
        os.environ["ECA_EVIDENCE_WEIGHT"] = "0.5"
        os.environ["ECA_SEARCH_COST_WEIGHT"] = "0.30"
        os.environ["ECA_BOUNDARY_DEFAULT"] = "Undetermined"
        os.environ["ECA_BOUNDARY_STRICT"] = "0"

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
        good = "<search>capital</search>\n" + ev + "<answer>Paris</answer>"
        nos = ev + "<answer>Paris</answer>"

        # NoSearch + search → Evidence OFF, pay α
        s_no = score("q_no", good)
        assert s_no["boundary"] == "NoSearch"
        assert abs(s_no["eff_evidence_weight"] - 0.0) < 1e-9
        assert abs(s_no["eff_search_cost_weight"] - 0.30) < 1e-9
        # R = 1 + 0 + 0.1 - 0.30 = 0.80
        assert abs(s_no["score"] - 0.80) < 1e-6, s_no["score"]

        # NoSearch + no search → no cost
        s_no2 = score("q_no", nos)
        assert abs(s_no2["score"] - 1.1) < 1e-6, s_no2["score"]

        # NeedSearch → full Evidence, no cost even with search
        s_need = score("q_need", good)
        assert s_need["boundary"] == "NeedSearch"
        assert abs(s_need["eff_evidence_weight"] - 0.5) < 1e-9
        assert abs(s_need["eff_search_cost_weight"] - 0.0) < 1e-9
        assert abs(s_need["score"] - 1.6) < 1e-6, s_need["score"]

        # Undetermined same as NeedSearch for v1
        s_und = score("q_und", good)
        assert s_und["boundary"] == "Undetermined"
        assert abs(s_und["score"] - 1.6) < 1e-6, s_und["score"]

        # STRICT missing
        os.environ["ECA_BOUNDARY_STRICT"] = "1"
        R._BOUNDARY_CACHE_KEY = None
        R._BOUNDARY_CACHE = {}
        R._LOGGED_HASH = False
        try:
            score("q_unknown", good)
            raise AssertionError("STRICT should raise")
        except KeyError:
            pass

    print("selftest_rewards_3d2b: PASS")


if __name__ == "__main__":
    main()
