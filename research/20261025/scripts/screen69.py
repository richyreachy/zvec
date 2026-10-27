from __future__ import annotations

import csv
import json
import logging

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")
assert "RUN69 COMPLETE" in (R / "logs/run69.log").read_text()
shared = {
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
    "BENCH_STOP_GRID": "1",
    "BENCH_MARGIN_DENOM": "1000",
}
run(
    "diag69_disabled",
    "fusion69_native",
    [20000],
    extra={**shared, "QG_FUSION_ENABLE": "0"},
    duration=-1,
    threads="1",
    nav="margin",
)
new = sorted(
    csv.reader((R / "experiments/diag69_disabled/search_k10_r0_ids.csv").open())
)
old = sorted(
    csv.reader((R / "experiments/time65_control_pre/search_k10_r0_ids.csv").open())
)
assert len(new) == 10000
assert new == old
(R / "results/fusion69_primary_equivalence.json").write_text(
    json.dumps(
        {
            "primary_queries": 1000,
            "id_score_rows": 10000,
            "disabled_matches_frozen_baseline": True,
            "primary_queries_already_tuned_not_blind": True,
            "lambda_frozen_no_refit": True,
        },
        indent=2,
    )
    + "\n"
)
run(
    "screen69_enabled",
    "fusion69_native",
    [16000, 20000, 24000, 28000, 32000, 40000],
    extra={**shared, "QG_FUSION_ENABLE": "1"},
    duration=1,
    threads="1",
    nav="margin",
)
logging.info("%s", "SCREEN69 COMPLETE")
