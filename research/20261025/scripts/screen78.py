from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

from explore90 import run

logging.basicConfig(level=logging.INFO, format="%(message)s")

R = Path(__file__).resolve().parent.parent
assert json.loads((R / "results/independent78_disabled_equivalence.json").read_text())[
    "passed"
]
assert json.loads((R / "results/integration78_actual_v2.json").read_text())["passed"]
for key in list(os.environ):
    if key.startswith("QG_INDEPENDENT_"):
        del os.environ[key]
name = "screen78_independent_v1"
run(
    name,
    "independent78_native_v1",
    [16000, 20000, 24000, 28000, 32000, 40000],
    extra={
        "QG_CAP": "64",
        "QG_PATIENCE": "0",
        "BENCH_STOP_GRID": "1",
        "BENCH_MARGIN_DENOM": "1000",
        "QG_INDEPENDENT_CODE_FILE": str(R / "models77/full/blocks.bin"),
        "QG_INDEPENDENT_RAW_FILE": str(R / "models64/full_vectors.f32"),
        "QG_INDEPENDENT_ROTATED_FILE": str(R / "models77/rotated.f32"),
        "QG_INDEPENDENT_HEADER_FILE": str(R / "models77/header1000000.bin"),
        "QG_INDEPENDENT_MODEL_SHA256": "2ac03543d4a09ae655d38130419df41668e4d62bdd3231b49135d2603c39f138",
    },
    rounds=1,
    duration=1,
    threads="1",
    nav="margin",
)
out = R / "experiments" / name
log = (out / "run.log").read_text()
assert "INDEPENDENT78 ready nodes=1000000 copied=997722 native_short=2278" in log
with (out / "search_k10_r0.csv").open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 6
eligible = [row for row in rows if float(row["recall"]) >= 0.9]
best = max(eligible, key=lambda row: float(row["qps"])) if eligible else None
with (R / "results/independent78_screen.json").open("x") as stream:
    json.dump(
        {
            "passed_full_import": True,
            "primary_query_count": 1000,
            "screen_only": True,
            "not_blind_queries": True,
            "new_timing_rows": len(rows),
            "rows": rows,
            "selected_candidate": best,
            "repeat_confirmation_pending": bool(best),
            "source_experiment": str(out),
        },
        stream,
        indent=2,
    )
    stream.write("\n")
logging.info("SCREEN78 %s", json.dumps(best))
