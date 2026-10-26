from __future__ import annotations

import csv
import json
import logging
import os
import subprocess

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")

commands = json.loads((R / "results/build_commands.json").read_text())
oldobj = str(R / "objects/benchmark.o")
newobj = str(R / "objects/radial57_test.o")
executed = []
c = next(x["argv"] for x in commands if "-c" in x["argv"] and oldobj in x["argv"])
c = [
    newobj
    if x == oldobj
    else str(R / "scripts/radial57_test.cc")
    if x == str(R / "scripts/benchmark.cc")
    else x
    for x in c
]
c += ["-DQG_ROTATION_768=1", "-DQG_RADIAL_CALIBRATION=1", "-UNDEBUG"]
executed.append(c)
subprocess.run(c, check=True, cwd=R)
c = next(x["argv"] for x in commands if "-c" not in x["argv"] and oldobj in x["argv"])
c = [
    newobj
    if x == oldobj
    else str(R / "radial57_test")
    if x == str(R / "benchmark")
    else x
    for x in c
]
executed.append(c)
subprocess.run(c, check=True, cwd=R)
(R / "results/radial57_test_build_commands.json").write_text(
    json.dumps(executed, indent=2)
)
with (R / "logs/radial57_test.log").open("w") as f:
    subprocess.run(
        [str(R / "radial57_test")],
        env={**os.environ, "LD_LIBRARY_PATH": str(R / "lib")},
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
logging.info("RADIAL TEST PASSED")
base = [
    "QG_BUCKET=1",
    "QG_BUCKET_WIDTH=.016f",
    "QG_ADAPTIVE=1",
    "QG_RUNTIME_ADAPTIVE=1",
    "QG_STOP_MIN=10",
    "QG_STOP_PATIENCE=0",
    "QG_STOP_MARGIN=.02f",
    "QG_ROTATION_768=1",
    "QG_AUGMENT=3",
]
integer = json.loads((R / "variants/int55_original/metadata.json").read_text())[
    "defines"
]
variants = {
    "rad57_native": [*base, "QG_RADIAL_CALIBRATION=1"],
    "rad57_integer": [*integer, "QG_RADIAL_CALIBRATION=1"],
}
env = {
    "QG_CENTER_MEAN_FILE": str(R / "mean54/mean.f32"),
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
    "QG_MARGIN": ".02",
}
for name, flags in variants.items():
    with (R / f"logs/build_{name}.log").open("w") as f:
        subprocess.run(
            ["python3", str(R / "scripts/compile_variant.py"), name, *flags],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
run(
    "screen57_cohere_control_pre",
    "bucket90_w16",
    [800],
    extra=env,
    rounds=2,
    duration=2,
    threads="1,8",
)
run(
    "screen57_cohere_integer_control",
    "int55_original",
    [800],
    extra=env,
    rounds=2,
    duration=2,
    threads="1,8",
)
for name in variants:
    run(
        "screen57_cohere_" + name,
        name,
        [0, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 60000],
        extra={**env, "BENCH_STOP_GRID": "1", "BENCH_MARGIN_DENOM": "1000"},
        rounds=1,
        duration=2,
        threads="1,8",
        nav="margin",
    )
run(
    "screen57_cohere_control_post",
    "bucket90_w16",
    [800],
    extra=env,
    rounds=2,
    duration=2,
    threads="1,8",
)
for base_name, flags in variants.items():
    rows = list(
        csv.DictReader(
            (R / f"experiments/screen57_cohere_{base_name}/search_k10_r0.csv").open()
        )
    )
    passing = [r for r in rows if r["threads"] == "1" and float(r["recall"]) >= 0.9]
    label = int(max(passing, key=lambda r: float(r["qps"]))["ef"]) if passing else 20000
    margin = label // 1000 / 1000
    patience = label % 1000
    modes = [("audit", "QG_CENTER_AUDIT=1")] + (
        [("full", "QG_DEFERRED_FULL=1")] if base_name == "rad57_integer" else []
    )
    for mode, flag in modes:
        name = base_name + "_" + mode
        with (R / f"logs/build_{name}.log").open("w") as f:
            subprocess.run(
                [
                    "python3",
                    str(R / "scripts/compile_variant.py"),
                    name,
                    *flags,
                    "QG_CENTER_STATS=1",
                    "QG_RADIAL_STATS=1",
                    flag,
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        run(
            "screen57_" + name,
            name,
            [800],
            extra={
                **env,
                "QG_MARGIN": str(margin),
                "QG_PATIENCE": str(patience),
                "QG_CENTER_STATS_FILE": str(R / f"results/{name}_stats.csv"),
                "QG_RADIAL_STATS_FILE": str(R / f"results/{name}_geometry.json"),
            },
            duration=-1,
            threads="1",
        )
    a = sorted(
        r[1:]
        for r in csv.reader(
            (R / f"experiments/screen57_{base_name}_audit/search_k10_r0_ids.csv").open()
        )
    )
    perf = sorted(
        r[1:]
        for r in csv.reader(
            (
                R / f"experiments/screen57_cohere_{base_name}/search_k10_r0_ids.csv"
            ).open()
        )
        if int(r[0]) == label
    )
    assert len(a) == len(perf) == 10000
    assert a == perf
    if base_name == "rad57_integer":
        b = sorted(
            r[1:]
            for r in csv.reader(
                (
                    R / f"experiments/screen57_{base_name}_full/search_k10_r0_ids.csv"
                ).open()
            )
        )
        assert a == b
    geometry = json.loads((R / f"results/{base_name}_audit_geometry.json").read_text())
    (R / f"results/{base_name}_equivalence.json").write_text(
        json.dumps(
            {
                "queries": 1000,
                "id_score_rows": 10000,
                "performance_vs_audit_exact_match": True,
                "audit_vs_full_exact_match": True
                if base_name == "rad57_integer"
                else None,
                "margin": margin,
                "ef_label": label,
                "geometry": geometry,
            },
            indent=2,
        )
    )
logging.info("SCREEN57 COMPLETE")
