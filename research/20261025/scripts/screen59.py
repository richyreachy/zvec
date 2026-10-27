from __future__ import annotations

import csv
import json
import logging
import subprocess

from explore90 import R, run

logging.basicConfig(level=logging.INFO)
flags = json.loads((R / "variants/int55_original/metadata.json").read_text())[
    "defines"
] + ["QG_INTEGER_ROUTER=1"]
for name, extra in [
    ("hier59_integer", []),
    ("hier59_audit", ["QG_CENTER_STATS=1", "QG_CENTER_AUDIT=1"]),
    ("hier59_full", ["QG_CENTER_STATS=1", "QG_DEFERRED_FULL=1"]),
]:
    with (R / f"logs/build_{name}.log").open("w") as f:
        subprocess.run(
            ["python3", str(R / "scripts/compile_variant.py"), name, *flags, *extra],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
subprocess.run(
    [
        "g++",
        "-O3",
        "-std=c++17",
        "-UNDEBUG",
        "-I/home/xingrui/workspace/quantized_graph/symphony_research_20261025/src/core",
        str(R / "scripts/router59_micro.cc"),
        "-o",
        str(R / "router59_micro"),
    ],
    check=True,
)
with (R / "results/router59_micro.csv").open("w") as f:
    subprocess.run(
        [
            "taskset",
            "-c",
            "0",
            str(R / "router59_micro"),
            str(R / "models59/router.bin"),
            "/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1m.1000.new.txt",
        ],
        stdout=f,
        check=True,
    )
shared = {
    "QG_CENTER_MEAN_FILE": str(R / "mean54/mean.f32"),
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
    "BENCH_STOP_GRID": "1",
    "BENCH_MARGIN_DENOM": "1000",
}
run(
    "screen59_cohere_control_pre",
    "bucket90_w16",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
run(
    "screen59_cohere_integer_control",
    "int55_original",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
run(
    "screen59_cohere_disabled",
    "hier59_integer",
    [20000],
    extra=shared,
    duration=-1,
    threads="1",
    nav="margin",
)
old = sorted(
    r[1:]
    for r in csv.reader(
        (R / "experiments/screen55_cohere_int55_original/search_k10_r0_ids.csv").open()
    )
    if r[0] == "20000"
)
current = sorted(
    r[1:]
    for r in csv.reader(
        (R / "experiments/screen59_cohere_disabled/search_k10_r0_ids.csv").open()
    )
)
assert len(current) == len(old) == 10000
assert current == old
(R / "results/router59_disabled_equivalence.json").write_text(
    json.dumps({"queries": 1000, "id_score_rows": 10000, "exact_match": True}, indent=2)
)
for beam in [1, 2, 4]:
    for keep in [0, 1]:
        env = {
            **shared,
            "QG_INTEGER_ROUTER_FILE": str(R / "models59/router.bin"),
            "QG_INTEGER_ROUTER_BEAM": str(beam),
            "QG_ROUTE_KEEP_SCALE": str(keep),
        }
        run(
            f"screen59_cohere_b{beam}_keep{keep}",
            "hier59_integer",
            [0, 10000, 20000, 40000],
            extra=env,
            duration=2,
            threads="1,8",
            nav="margin",
        )
run(
    "screen59_cohere_control_post",
    "bucket90_w16",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
logging.info("%s", "SCREEN59 COMPLETE")
