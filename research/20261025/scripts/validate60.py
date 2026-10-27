from __future__ import annotations

import csv
import json
import logging
import statistics
import subprocess

from explore90 import R, run

logging.basicConfig(level=logging.INFO)
assert "SCREEN60 COMPLETE" in (R / "logs/screen60.log").read_text()
selected = {}
for dim in [768, 384, 256]:
    name = f"screen60_cohere_proj{dim}"
    rows = []
    for p in (R / "experiments" / name).glob("search_k10_r*.csv"):
        if not p.stem.endswith(("_ids", "_recall")):
            rows.extend(csv.DictReader(p.open()))
    passed = [r for r in rows if r["threads"] == "1" and float(r["recall"]) >= 0.9]
    assert passed
    best_label = max(
        {r["ef"] for r in passed},
        key=lambda e: statistics.median(
            float(r["qps"]) for r in passed if r["ef"] == e
        ),
    )
    selected[dim] = {
        "experiment": name,
        "label": int(best_label),
        "qps_median": statistics.median(
            float(r["qps"]) for r in passed if r["ef"] == best_label
        ),
    }
(R / "results/router60_selected_primary.json").write_text(
    json.dumps(selected, indent=2)
)
flags = json.loads((R / "variants/hier59_integer/metadata.json").read_text())["defines"]
for dim in [768, 384, 256]:
    for mode, extra in [
        ("audit", ["QG_CENTER_AUDIT=1"]),
        ("full", ["QG_DEFERRED_FULL=1"]),
    ]:
        variant = f"proj60_{dim}_{mode}"
        with (R / f"logs/build_{variant}.log").open("w") as f:
            subprocess.run(
                [
                    "python3",
                    str(R / "scripts/compile_variant.py"),
                    variant,
                    *flags,
                    f"QG_ROUTER_PROJECTED={dim}",
                    "QG_CENTER_STATS=1",
                    *extra,
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
results = []
for dim, chosen in selected.items():
    env = json.loads(
        (R / "experiments" / chosen["experiment"] / "config.json").read_text()
    )["env"]
    extra = {
        k: v
        for k, v in env.items()
        if k.startswith("QG_") or k in ["BENCH_STOP_GRID", "BENCH_MARGIN_DENOM"]
    }
    ref = sorted(
        r[1:]
        for r in csv.reader(
            (R / "experiments" / chosen["experiment"] / "search_k10_r0_ids.csv").open()
        )
        if r[0] == str(chosen["label"])
    )
    assert len(ref) == 10000
    for mode in ["audit", "full"]:
        variant = f"proj60_{dim}_{mode}"
        extra["QG_CENTER_STATS_FILE"] = str(R / f"results/{variant}_stats.csv")
        run(
            "diag60_" + variant,
            variant,
            [chosen["label"]],
            extra=extra,
            duration=-1,
            threads="1",
            nav="margin",
        )
        actual = sorted(
            r[1:]
            for r in csv.reader(
                (R / f"experiments/diag60_{variant}/search_k10_r0_ids.csv").open()
            )
        )
        assert actual == ref
    stats = list(csv.reader((R / f"results/proj60_{dim}_audit_stats.csv").open()))
    assert len(stats) == 1000
    results.append(
        {
            "dimension": dim,
            "queries": 1000,
            "id_score_rows_per_mode": 10000,
            "audit_full_performance_equal": True,
            "mean_expanded": statistics.mean(int(r[3]) for r in stats),
            "mean_refined": statistics.mean(int(r[4]) for r in stats),
        }
    )
(R / "results/router60_actual_validation.json").write_text(
    json.dumps(results, indent=2)
)
s = (
    (R / "scripts/validate_stage58_default.py")
    .read_text()
    .replace("stage58", "stage60")
    .replace("STAGE58", "STAGE60")
)
(R / "scripts/validate_stage60_default.py").write_text(s)
with (R / "logs/validate_stage60_default.log").open("w") as f:
    subprocess.run(
        ["python3", str(R / "scripts/validate_stage60_default.py")],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
logging.info("%s", "VALIDATE60 COMPLETE")
