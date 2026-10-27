from __future__ import annotations

import csv
import json
import logging
import subprocess

from explore90 import R, run

logging.basicConfig(level=logging.INFO)
S = "/home/xingrui/workspace/quantized_graph/symphony_research_20261025"
assert "PREPARE60 COMPLETE" in (R / "logs/prepare60.log").read_text()
flags = json.loads((R / "variants/hier59_integer/metadata.json").read_text())["defines"]
commands = []
for dim in [768, 384, 256]:
    variant = f"proj60_{dim}"
    with (R / f"logs/build_{variant}.log").open("w") as f:
        subprocess.run(
            [
                "python3",
                str(R / "scripts/compile_variant.py"),
                variant,
                *flags,
                f"QG_ROUTER_PROJECTED={dim}",
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        "-UNDEBUG",
        "-DQG_ROTATION_768=1",
        f"-DQG_ROUTER_PROJECTED={dim}",
        "-I" + S + "/src/core",
        "-I" + S + "/src",
        str(R / "scripts/router60_micro.cc"),
        str(R / "lib/libzvec_turbo.a"),
        str(R / "lib/libzvec_ailego.a"),
        "-o",
        str(R / f"router60_micro{dim}"),
    ]
    commands.append(cmd)
    subprocess.run(cmd, check=True)
for dim in [768, 384, 256]:
    cmd = [
        "taskset",
        "-c",
        "0",
        str(R / f"router60_micro{dim}"),
        str(R / f"models60/router{dim}.bin"),
        "/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1m.1000.new.txt",
        str(R / f"results/router60_trace{dim}.csv"),
    ]
    commands.append(cmd)
    with (R / f"results/router60_micro{dim}.csv").open("w") as f:
        subprocess.run(cmd, stdout=f, check=True)
(R / "results/router60_micro_commands.json").write_text(json.dumps(commands, indent=2))
shared = {
    "QG_CENTER_MEAN_FILE": str(R / "mean54/mean.f32"),
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
    "BENCH_STOP_GRID": "1",
    "BENCH_MARGIN_DENOM": "1000",
}
run(
    "screen60_cohere_control_pre",
    "bucket90_w16",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
run(
    "screen60_cohere_integer_control",
    "int55_original",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
run(
    "screen60_cohere_original_router",
    "hier59_integer",
    [20000],
    extra={
        **shared,
        "QG_INTEGER_ROUTER_FILE": str(R / "models59/router.bin"),
        "QG_INTEGER_ROUTER_BEAM": "1",
        "QG_ROUTE_KEEP_SCALE": "0",
    },
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
run(
    "screen60_cohere_disabled",
    "proj60_256",
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
new = sorted(
    r[1:]
    for r in csv.reader(
        (R / "experiments/screen60_cohere_disabled/search_k10_r0_ids.csv").open()
    )
)
assert len(new) == len(old) == 10000
assert new == old
(R / "results/router60_disabled_equivalence.json").write_text(
    json.dumps({"queries": 1000, "id_score_rows": 10000, "exact_match": True}, indent=2)
)
for dim in [768, 384, 256]:
    env = {
        **shared,
        "QG_INTEGER_ROUTER_FILE": str(R / f"models60/router{dim}.bin"),
        "QG_INTEGER_ROUTER_BEAM": "1",
        "QG_ROUTE_KEEP_SCALE": "0",
    }
    run(
        f"screen60_cohere_proj{dim}",
        f"proj60_{dim}",
        [0, 10000, 20000, 40000],
        extra=env,
        rounds=2,
        duration=3,
        threads="1,8",
        nav="margin",
    )
run(
    "screen60_cohere_control_post",
    "bucket90_w16",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
logging.info("%s", "SCREEN60 COMPLETE")
