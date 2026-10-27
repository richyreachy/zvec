from __future__ import annotations

import csv
import json
import logging

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")
M = R / "models68/primary"
M.mkdir(exist_ok=False)
shared = {
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
    "BENCH_STOP_GRID": "1",
    "BENCH_MARGIN_DENOM": "1000",
    "QG_MULTI_TRACE_FILE": str(M / "events.bin"),
}
run(
    "diag68_primary",
    "trace68_native",
    [20000],
    extra=shared,
    duration=-1,
    threads="1",
    nav="margin",
)
new = sorted(
    csv.reader((R / "experiments/diag68_primary/search_k10_r0_ids.csv").open())
)
old = sorted(
    csv.reader((R / "experiments/time65_control_pre/search_k10_r0_ids.csv").open())
)
assert len(new) == 10000
assert new == old
(R / "results/trace68_primary_equivalence.json").write_text(
    json.dumps(
        {
            "primary_queries": 1000,
            "id_score_rows": 10000,
            "matches_frozen_baseline": True,
            "primary_queries_already_tuned_not_blind": True,
            "gt_only_used_by_existing_recall_driver_not_training": True,
            "no_model_fitted_from_primary_results": True,
        },
        indent=2,
    )
    + "\n"
)
logging.info("%s", "PRIMARY68 COMPLETE")
