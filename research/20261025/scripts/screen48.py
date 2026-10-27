from __future__ import annotations

import json
import logging
import os
import subprocess
import time

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")


def emit(*items, **_kwargs):
    logging.info(" ".join(map(str, items)))


while "DIAGNOSTIC47 COMPLETE" not in (R / "logs/diagnostic47.log").read_text():
    time.sleep(3)
commands = json.loads((R / "results/build_commands.json").read_text())
oldobj = str(R / "objects/benchmark.o")
newobj = str(R / "objects/lowrank48_test.o")
executed = []
c = next(x["argv"] for x in commands if "-c" in x["argv"] and oldobj in x["argv"])
c = [
    newobj
    if x == oldobj
    else str(R / "scripts/lowrank48_test.cc")
    if x == str(R / "scripts/benchmark.cc")
    else x
    for x in c
]
c += ["-DQG_LOWRANK=4", "-DQG_ROTATION_768=1", "-UNDEBUG"]
executed.append(c)
subprocess.run(c, check=True, cwd=R)
c = next(x["argv"] for x in commands if "-c" not in x["argv"] and oldobj in x["argv"])
c = [
    newobj
    if x == oldobj
    else str(R / "lowrank48_test")
    if x == str(R / "benchmark")
    else x
    for x in c
]
executed.append(c)
subprocess.run(c, check=True, cwd=R)
(R / "results/lowrank48_test_build_commands.json").write_text(
    json.dumps(executed, indent=2)
)
with (R / "logs/lowrank48_test.log").open("w") as f:
    subprocess.run(
        [str(R / "lowrank48_test")],
        env={
            **os.environ,
            "LD_LIBRARY_PATH": str(R / "lib"),
            "QG_LOWRANK_BASIS": str(R / "results/lowrank48_test_basis.f32"),
        },
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
subprocess.run(
    [
        "/home/xingrui/bench/symphony_qg/.venv/bin/python",
        str(R / "scripts/prepare_lowrank48.py"),
    ],
    env={**os.environ, "OPENBLAS_NUM_THREADS": "8"},
    check=True,
)
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
]
for rank in [4, 8, 16]:
    name = f"lowrank48_{rank}"
    with (R / f"logs/build_{name}.log").open("w") as f:
        subprocess.run(
            [
                "python3",
                str(R / "scripts/compile_variant.py"),
                name,
                *common,
                f"QG_LOWRANK={rank}",
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
run(
    "screen48_cohere_control_pre",
    "bucket90_w16",
    [800],
    extra={"QG_MARGIN": ".02", "QG_PATIENCE": "0", "QG_CAP": "64"},
    rounds=2,
    duration=2,
    threads="1,8",
)
for rank in [4, 8, 16]:
    name = f"lowrank48_{rank}"
    run(
        "screen48_cohere_" + name,
        name,
        [-2000, -1000, 0, 1000, 2000, 3000, 4000, 6000],
        extra={
            "BENCH_STOP_GRID": "1",
            "QG_CAP": "64",
            "QG_LOWRANK_BASIS": str(R / "lowrank48/basis.f32"),
        },
        rounds=1,
        duration=2,
        threads="1,8",
        nav="margin",
    )
run(
    "screen48_cohere_control_post",
    "bucket90_w16",
    [800],
    extra={"QG_MARGIN": ".02", "QG_PATIENCE": "0", "QG_CAP": "64"},
    rounds=2,
    duration=2,
    threads="1,8",
)
emit("SCREEN48 COMPLETE", flush=True)
