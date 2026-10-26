from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import statistics
from pathlib import Path

from explore90 import run

logging.basicConfig(level=logging.INFO, format="%(message)s")

R = Path(__file__).resolve().parent.parent
screen = json.loads((R / "results/independent78_screen.json").read_text())
selected = screen["selected_candidate"]
assert selected is not None
assert int(selected["ef"]) == 24000
for key in list(os.environ):
    if key.startswith("QG_INDEPENDENT_"):
        del os.environ[key]
reference = (R / "experiments/time65_control_pre/search_k10_r0_ids.csv").read_bytes()
learned = b"".join(
    line
    for line in (R / "experiments/screen78_independent_v1/search_k10_r0_ids.csv")
    .read_bytes()
    .splitlines(keepends=True)
    if line.startswith(b"24000,")
)
assert len(reference.splitlines()) == len(learned.splitlines()) == 10000
variants = {
    "baseline": "bucket90_w16",
    "learned": "independent78_native_v1",
    "disabled": "independent78_native_v1",
}
orders = [
    ["baseline", "learned", "disabled"],
    ["disabled", "learned", "baseline"],
    ["learned", "baseline", "disabled"],
    ["disabled", "baseline", "learned"],
]
model = {
    "QG_INDEPENDENT_CODE_FILE": str(R / "models77/full/blocks.bin"),
    "QG_INDEPENDENT_RAW_FILE": str(R / "models64/full_vectors.f32"),
    "QG_INDEPENDENT_ROTATED_FILE": str(R / "models77/rotated.f32"),
    "QG_INDEPENDENT_HEADER_FILE": str(R / "models77/header1000000.bin"),
    "QG_INDEPENDENT_MODEL_SHA256": "2ac03543d4a09ae655d38130419df41668e4d62bdd3231b49135d2603c39f138",
}
rows = []
runs = []
for paired_round, order in enumerate(orders):
    for condition in order:
        name = f"confirm78_r{paired_round}_{condition}"
        extra = {
            "QG_CAP": "64",
            "QG_PATIENCE": "0",
            "BENCH_STOP_GRID": "1",
            "BENCH_MARGIN_DENOM": "1000",
        }
        if condition == "learned":
            extra.update(model)
        ef = 24000 if condition == "learned" else 20000
        run(
            name,
            variants[condition],
            [ef],
            extra=extra,
            rounds=1,
            duration=3,
            threads="1,8",
            nav="margin",
        )
        out = R / "experiments" / name
        actual = (out / "search_k10_r0_ids.csv").read_bytes()
        assert actual == (learned if condition == "learned" else reference)
        log = (out / "run.log").read_text()
        if condition == "learned":
            assert (
                "INDEPENDENT78 ready nodes=1000000 copied=997722 native_short=2278"
                in log
            )
        else:
            assert "INDEPENDENT78 ready" not in log
        with (out / "search_k10_r0.csv").open() as stream:
            measured = list(csv.DictReader(stream))
        assert len(measured) == 2
        assert {row["threads"] for row in measured} == {"1", "8"}
        for row in measured:
            assert float(row["seconds"]) >= 3
            assert float(row["recall"]) >= 0.9
            rows.append(dict(row, condition=condition, paired_round=paired_round))
        runs.append(
            {
                "condition": condition,
                "paired_round": paired_round,
                "experiment": name,
                "ids_hex_sha256": hashlib.sha256(actual).hexdigest(),
            }
        )
        progress = {
            "complete": False,
            "runs": runs,
            "rows": rows,
            "planned_orders": orders,
            "timing_rows_completed": len(rows),
        }
        (R / "results/independent78_confirmation_progress.json").write_text(
            json.dumps(progress, indent=2) + "\n"
        )
summary = {}
for threads in ["1", "8"]:
    by_condition = {
        condition: [
            float(row["qps"])
            for row in rows
            if row["threads"] == threads and row["condition"] == condition
        ]
        for condition in variants
    }
    medians = {
        condition: statistics.median(values)
        for condition, values in by_condition.items()
    }
    paired_gain = [
        by_condition["learned"][i] / by_condition["baseline"][i] - 1 for i in range(4)
    ]
    summary[threads] = {
        "qps_rounds": by_condition,
        "median_qps": medians,
        "learned_over_baseline_median_gain": medians["learned"] / medians["baseline"]
        - 1,
        "learned_over_baseline_paired_gains": paired_gain,
        "disabled_over_baseline_median_gain": medians["disabled"] / medians["baseline"]
        - 1,
    }
result = {
    "complete": True,
    "rounds": 4,
    "minimum_seconds": 3,
    "threads": [1, 8],
    "orders": orders,
    "runs": runs,
    "rows": rows,
    "summary": summary,
    "timing_rows": len(rows),
    "all_top10_ids_and_hex_scores_match_frozen_screen_or_baseline": True,
    "additional_queries_tested": False,
    "breakthrough_verified": False,
}
with (R / "results/independent78_confirmation.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
logging.info("CONFIRM78 COMPLETE %s", json.dumps(summary))
