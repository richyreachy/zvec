from __future__ import annotations

import csv
import json
import logging
import statistics
import subprocess

from explore90 import R, run

logging.basicConfig(level=logging.INFO)
assert "SCREEN59 COMPLETE" in (R / "logs/screen59.log").read_text()
rows = []
for d in sorted((R / "experiments").glob("screen59_cohere_b*")):
    for p in d.glob("search_k10_r*.csv"):
        if p.stem.endswith(("_ids", "_recall")):
            continue
        for r in csv.DictReader(p.open()):
            if r["threads"] == "1" and float(r["recall"]) >= 0.9:
                rows.append({**r, "experiment": d.name})
assert rows
best = max(rows, key=lambda r: float(r["qps"]))
(R / "results/router59_selected_primary.json").write_text(json.dumps(best, indent=2))
config = json.loads(
    (R / "experiments" / best["experiment"] / "config.json").read_text()
)
env = config["env"]
label = int(best["ef"])
for variant in ["hier59_audit", "hier59_full"]:
    extra = {
        k: v
        for k, v in env.items()
        if k.startswith("QG_") or k in ["BENCH_STOP_GRID", "BENCH_MARGIN_DENOM"]
    }
    extra["QG_CENTER_STATS_FILE"] = str(R / f"results/{variant}_stats.csv")
    run(
        "diag59_" + variant,
        variant,
        [label],
        extra=extra,
        duration=-1,
        threads="1",
        nav="margin",
    )
reference = sorted(
    r[1:]
    for r in csv.reader(
        (R / "experiments" / best["experiment"] / "search_k10_r0_ids.csv").open()
    )
    if r[0] == str(label)
)
assert len(reference) == 10000
for variant in ["hier59_audit", "hier59_full"]:
    actual = sorted(
        r[1:]
        for r in csv.reader(
            (R / f"experiments/diag59_{variant}/search_k10_r0_ids.csv").open()
        )
    )
    assert actual == reference
stats = list(csv.reader((R / "results/hier59_audit_stats.csv").open()))
assert len(stats) == 1000
result = {
    "selected_primary": best,
    "audit_full_performance_queries": 1000,
    "id_score_rows_per_mode": 10000,
    "mean_expanded": statistics.mean(int(r[3]) for r in stats),
    "mean_graph_refined": statistics.mean(int(r[4]) for r in stats),
    "router_fp32_exact_calls": 0,
    "model": str(R / "models59/router.bin"),
    "extra_queries_used": False,
}
(R / "results/router59_actual_validation.json").write_text(json.dumps(result, indent=2))
s = (
    (R / "scripts/validate_stage58_default.py")
    .read_text()
    .replace("stage58", "stage59")
    .replace("STAGE58", "STAGE59")
)
(R / "scripts/validate_stage59_default.py").write_text(s)
with (R / "logs/validate_stage59_default.log").open("w") as f:
    subprocess.run(
        ["python3", str(R / "scripts/validate_stage59_default.py")],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
logging.info("%s", "VALIDATE59 COMPLETE")
