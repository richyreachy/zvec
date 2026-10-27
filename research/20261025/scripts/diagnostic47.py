from __future__ import annotations

import csv
import json
import logging
import statistics
import subprocess
import time

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")


def emit(*items, **_kwargs):
    logging.info(" ".join(map(str, items)))


while "SCREEN46 COMPLETE" not in (R / "logs/screen46.log").read_text():
    time.sleep(3)
common = [
    "QG_BUCKET=1",
    "QG_ADAPTIVE=1",
    "QG_RUNTIME_ADAPTIVE=1",
    "QG_STOP_MIN=10",
    "QG_STOP_PATIENCE=0",
    "QG_STOP_MARGIN=.02f",
    "QG_ROTATION_768=1",
    "QG_AUGMENT=3",
    "QG_CENTER_STATS=1",
    "QG_DIAG=1",
]
variants = {
    "diag47_onebit16": ["QG_BUCKET_WIDTH=.016f"],
    "diag47_oracle16": ["QG_BUCKET_WIDTH=.016f", "QG_EXPERIMENT=1"],
    "diag47_oracle8": ["QG_BUCKET_WIDTH=.008f", "QG_EXPERIMENT=1"],
    "diag47_oracle2": ["QG_BUCKET_WIDTH=.002f", "QG_EXPERIMENT=1"],
}
labels = [-4000, -2000, 0, 1000, 2000, 4000, 6000]
summary = []
for name, flags in variants.items():
    if not (R / f"variants/{name}/commands.json").exists():
        with (R / f"logs/build_{name}.log").open("w") as f:
            subprocess.run(
                [
                    "python3",
                    str(R / "scripts/compile_variant.py"),
                    name,
                    *common,
                    *flags,
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
    if name == "diag47_onebit16" and (R / f"experiments/{name}").exists():
        assert (
            "DONE diag47_onebit16"
            in (R / "logs/diagnostic47.log.initial_parser_failure").read_text()
        )
    else:
        run(
            name,
            name,
            labels,
            extra={
                "BENCH_STOP_GRID": "1",
                "QG_CAP": "64",
                "QG_CENTER_STATS_FILE": str(R / f"results/{name}_stats.csv"),
                "QG_TRACE": str(R / f"results/{name}_diag.csv"),
            },
            duration=-1,
            threads="1",
            nav="margin",
        )
    rows = list(
        csv.DictReader((R / f"experiments/{name}/search_k10_r0_recall.csv").open())
    )
    stats = list(csv.reader((R / f"results/{name}_stats.csv").open()))
    assert len(stats) == len(labels) * 1000, (name, len(stats))
    for i, label in enumerate(labels):
        rr = [r for r in rows if int(r["ef"]) == label]
        assert len(rr) == 1000
        recall = sum(int(r["hits"]) for r in rr) / 10000
        expanded = [int(r[3]) for r in stats[i * 1000 : (i + 1) * 1000]]
        summary.append(
            {
                "variant": name,
                "margin": label / 100000,
                "recall": recall,
                "mean_expanded": statistics.mean(expanded),
                "median_expanded": statistics.median(expanded),
                "cap_fraction": sum(x == 64 for x in expanded) / 1000,
            }
        )
    (R / "results/diagnostic47.json").write_text(
        json.dumps(
            {
                "diagnostic_only": True,
                "oracle_is_not_deployable_qps": True,
                "rows": summary,
            },
            indent=2,
        )
    )
    emit(json.dumps(summary[-len(labels) :]), flush=True)
emit("DIAGNOSTIC47 COMPLETE", flush=True)
