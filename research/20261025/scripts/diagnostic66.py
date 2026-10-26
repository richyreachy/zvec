from __future__ import annotations

import json
import logging
import subprocess

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")
labels = [0, 10000, 20000, 30000, 40000, 50000, 60000]
plan = {
    "labels": labels,
    "margin_denominator": 1000000,
    "graphs": ["original", "nearest", "angular"],
    "scores": ["onebit", "exact"],
    "topk": 10,
    "degree": 32,
    "ef": 800,
    "cap": 64,
    "bucket_width": 0.016,
    "same_original_entry": True,
    "primary_not_blind": True,
    "no_qps_measurement": True,
    "oracle_not_deployable_or_lower_bound": True,
    "graph_models_fixed_in_phase65": True,
}
(R / "results/graph66_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
base = json.loads((R / "variants/bucket90_w16/metadata.json").read_text())[
    "defines"
] + ["QG_GRAPH_IMPORT=1", "QG_CENTER_STATS=1", "QG_DIAG=1"]
for score, extra in [("onebit", []), ("exact", ["QG_EXPERIMENT=1"])]:
    name = "graph66_" + score
    with (R / f"logs/build_{name}.log").open("w") as f:
        subprocess.run(
            ["python3", str(R / "scripts/compile_variant.py"), name, *base, *extra],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    logging.info("%s %s", "BUILT", name)
for graph in ["original", "nearest", "angular"]:
    for score in ["onebit", "exact"]:
        name = f"diag66_{graph}_{score}"
        env = {
            "QG_CAP": "64",
            "QG_PATIENCE": "0",
            "BENCH_STOP_GRID": "1",
            "BENCH_MARGIN_DENOM": "1000",
            "QG_CENTER_STATS_FILE": str(R / f"results/{name}_stats.csv"),
            "QG_TRACE": str(R / f"results/{name}_diag.csv"),
        }
        if graph != "original":
            env["QG_GRAPH_IMPORT_FILE"] = str(R / f"models65/full_{graph}.bin")
        run(
            name,
            "graph66_" + score,
            labels,
            extra=env,
            duration=-1,
            threads="1",
            nav="margin",
        )
logging.info("%s", "DIAGNOSTIC66 COMPLETE")
