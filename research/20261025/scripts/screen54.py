from __future__ import annotations

import csv
import json
import logging
import os
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
]
variants = {
    "mean54_plain": [],
    "mean54_tight": ["QG_COSINE_BOUND_TIGHT=1"],
    "mean54_cv": ["QG_COSINE_BOUND_TIGHT=1", "QG_CENTER_CONTROL_VARIATE=1"],
}
env = {
    "QG_CENTER_MEAN_FILE": str(R / "mean54/mean.f32"),
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
}
commands = json.loads((R / "results/build_commands.json").read_text())
oldobj = str(R / "objects/benchmark.o")
executed = []
for tight in [False, True]:
    name = "mean_center54_test" + ("_tight" if tight else "")
    obj = str(R / f"objects/{name}.o")
    c = next(x["argv"] for x in commands if "-c" in x["argv"] and oldobj in x["argv"])
    c = [
        obj
        if x == oldobj
        else str(R / "scripts/mean_center54_test.cc")
        if x == str(R / "scripts/benchmark.cc")
        else x
        for x in c
    ]
    c += ["-DQG_ROTATION_768=1", "-UNDEBUG"] + (
        ["-DQG_COSINE_BOUND_TIGHT=1"] if tight else []
    )
    executed.append(c)
    subprocess.run(c, check=True, cwd=R)
    c = next(
        x["argv"] for x in commands if "-c" not in x["argv"] and oldobj in x["argv"]
    )
    c = [
        obj if x == oldobj else str(R / name) if x == str(R / "benchmark") else x
        for x in c
    ]
    executed.append(c)
    subprocess.run(c, check=True, cwd=R)
    with (R / f"logs/{name}.log").open("w") as f:
        subprocess.run(
            [str(R / name)],
            env={**os.environ, "LD_LIBRARY_PATH": str(R / "lib")},
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    logging.info("TEST PASSED %s", name)
(R / "results/mean_center54_test_build_commands.json").write_text(
    json.dumps(executed, indent=2)
)
for name, flags in variants.items():
    with (R / f"logs/build_{name}.log").open("w") as f:
        subprocess.run(
            ["python3", str(R / "scripts/compile_variant.py"), name, *common, *flags],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
run(
    "screen54_cohere_control_pre",
    "bucket90_w16",
    [800],
    extra={"QG_MARGIN": ".02", "QG_PATIENCE": "0", "QG_CAP": "64"},
    rounds=2,
    duration=2,
    threads="1,8",
)
for name in variants:
    run(
        "screen54_cohere_" + name,
        name,
        [10000, 15000, 20000, 25000, 30000, 35000, 40000],
        extra={**env, "BENCH_STOP_GRID": "1", "BENCH_MARGIN_DENOM": "1000"},
        rounds=1,
        duration=2,
        threads="1,8",
        nav="margin",
    )
run(
    "screen54_cohere_control_post",
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
            (R / f"experiments/screen54_cohere_{base}/search_k10_r0.csv").open()
        )
    )
    passing = [r for r in rows if r["threads"] == "1" and float(r["recall"]) >= 0.9]
    label = int(max(passing, key=lambda r: float(r["qps"]))["ef"]) if passing else 20000
    # STOP_GRID with denominator 1000 provides 0.005 margin increments.
    margin = (label // 1000) / 1000
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
            "screen54_" + name,
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
            (R / f"experiments/screen54_{base}_audit/search_k10_r0_ids.csv").open()
        )
    )
    b = sorted(
        csv.reader(
            (R / f"experiments/screen54_{base}_full/search_k10_r0_ids.csv").open()
        )
    )
    assert len(a) == len(b) == 10000
    assert a == b
    (R / f"results/{base}_equivalence.json").write_text(
        json.dumps(
            {
                "queries": 1000,
                "id_score_rows": 10000,
                "audit_vs_full_exact_match": True,
                "margin": margin,
                "patience": patience,
            },
            indent=2,
        )
    )
logging.info("SCREEN54 COMPLETE")
