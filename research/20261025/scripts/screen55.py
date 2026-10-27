from __future__ import annotations

import csv
import json
import logging
import subprocess

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")

common = [
    "QG_BUCKET=1",
    "QG_BUCKET_WIDTH=.016f",
    "QG_ADAPTIVE=1",
    "QG_RUNTIME_ADAPTIVE=1",
    "QG_STOP_MIN=10",
    "QG_STOP_PATIENCE=0",
    "QG_STOP_MARGIN=.02f",
    "QG_ROTATION_768=1",
    "QG_AUGMENT=3",
    "QG_CENTER_SQ8=1",
    "QG_DEFERRED_COSINE=1",
    "QG_DEFERRED_ADAPTIVE=1",
    "QG_REANCHOR=1",
    "QG_CENTER_SHIFT=1",
    "QG_REFINE_BATCH=4",
    "QG_DEFERRED_ESTIMATE_STOP=1",
    "QG_INTEGER_CENTER=1",
]
variants = {"int55_original": ["QG_INTEGER_ORIGINAL_ANCHOR=1"], "int55_reanchor": []}
env = {
    "QG_CENTER_MEAN_FILE": str(R / "mean54/mean.f32"),
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
}
command = [
    "g++",
    "-O3",
    "-std=c++17",
    "-UNDEBUG",
    "-I/home/xingrui/workspace/quantized_graph/symphony_research_20261025/src/core",
    str(R / "scripts/integer_center55_test.cc"),
    "-o",
    str(R / "integer_center55_test"),
]
subprocess.run(command, check=True)
(R / "results/integer_center55_test_build_command.json").write_text(
    json.dumps(command, indent=2)
)
with (R / "logs/integer_center55_test.log").open("w") as f:
    subprocess.run(
        [str(R / "integer_center55_test")],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
logging.info("INTEGER TEST PASSED")
for name, flags in variants.items():
    with (R / f"logs/build_{name}.log").open("w") as f:
        subprocess.run(
            ["python3", str(R / "scripts/compile_variant.py"), name, *common, *flags],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
run(
    "screen55_cohere_control_pre",
    "bucket90_w16",
    [800],
    extra={"QG_MARGIN": ".02", "QG_PATIENCE": "0", "QG_CAP": "64"},
    rounds=2,
    duration=2,
    threads="1,8",
)
for name in variants:
    run(
        "screen55_cohere_" + name,
        name,
        [10000, 15000, 20000, 25000, 30000, 35000, 40000],
        extra={**env, "BENCH_STOP_GRID": "1", "BENCH_MARGIN_DENOM": "1000"},
        rounds=1,
        duration=2,
        threads="1,8",
        nav="margin",
    )
run(
    "screen55_cohere_control_post",
    "bucket90_w16",
    [800],
    extra={"QG_MARGIN": ".02", "QG_PATIENCE": "0", "QG_CAP": "64"},
    rounds=2,
    duration=2,
    threads="1,8",
)
for base, flags in variants.items():
    rows = list(
        csv.DictReader(
            (R / f"experiments/screen55_cohere_{base}/search_k10_r0.csv").open()
        )
    )
    passing = [r for r in rows if r["threads"] == "1" and float(r["recall"]) >= 0.9]
    label = int(max(passing, key=lambda r: float(r["qps"]))["ef"]) if passing else 20000
    margin = label // 1000 / 1000
    patience = label % 1000
    for mode, flag in [("audit", "QG_CENTER_AUDIT=1"), ("full", "QG_DEFERRED_FULL=1")]:
        name = base + "_" + mode
        with (R / f"logs/build_{name}.log").open("w") as f:
            subprocess.run(
                [
                    "python3",
                    str(R / "scripts/compile_variant.py"),
                    name,
                    *common,
                    *flags,
                    "QG_CENTER_STATS=1",
                    flag,
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        run(
            "screen55_" + name,
            name,
            [800],
            extra={
                **env,
                "QG_MARGIN": str(margin),
                "QG_PATIENCE": str(patience),
                "QG_CENTER_STATS_FILE": str(R / f"results/{name}_stats.csv"),
            },
            duration=-1,
            threads="1",
        )
    a = sorted(
        csv.reader(
            (R / f"experiments/screen55_{base}_audit/search_k10_r0_ids.csv").open()
        )
    )
    b = sorted(
        csv.reader(
            (R / f"experiments/screen55_{base}_full/search_k10_r0_ids.csv").open()
        )
    )
    assert len(a) == len(b) == 10000
    assert a == b
    perf = sorted(
        r[1:]
        for r in csv.reader(
            (R / f"experiments/screen55_cohere_{base}/search_k10_r0_ids.csv").open()
        )
        if int(r[0]) == label
    )
    assert perf == sorted(r[1:] for r in a)
    (R / f"results/{base}_equivalence.json").write_text(
        json.dumps(
            {
                "queries": 1000,
                "id_score_rows": 10000,
                "audit_vs_full_exact_match": True,
                "performance_vs_audit_exact_match": True,
                "margin": margin,
                "patience": patience,
                "ef_label": label,
            },
            indent=2,
        )
    )
logging.info("SCREEN55 COMPLETE")
