from __future__ import annotations

import json
import logging

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")
assert "SCREEN65 COMPLETE" in (R / "logs/screen65.log").read_text()
shared = {
    "QG_CENTER_MEAN_FILE": str(R / "mean54/mean.f32"),
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
    "BENCH_STOP_GRID": "1",
    "BENCH_MARGIN_DENOM": "1000",
}
plan = {
    "nearest_labels": [14000, 16000, 20000],
    "angular_labels": [18000, 20000, 22000],
    "rounds": 2,
    "duration_seconds": 3,
    "threads": [1, 8],
    "selection_from_primary_queries": True,
    "frozen_controls_label": 20000,
}
(R / "results/graph65_timing_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
run(
    "time65_control_pre",
    "bucket90_w16",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
run(
    "time65_integer_pre",
    "int55_original",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
for mode in ("nearest", "angular"):
    for kind in ("native", "integer"):
        extra = {**shared, "QG_GRAPH_IMPORT_FILE": str(R / f"models65/full_{mode}.bin")}
        run(
            "time65_" + kind + "_" + mode,
            "graph65_" + kind,
            plan[mode + "_labels"],
            extra=extra,
            rounds=2,
            duration=3,
            threads="1,8",
            nav="margin",
        )
run(
    "time65_integer_post",
    "int55_original",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
run(
    "time65_control_post",
    "bucket90_w16",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
logging.info("%s", "TIME65 COMPLETE")
