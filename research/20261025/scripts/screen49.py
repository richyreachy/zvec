from __future__ import annotations

import json
import logging
import subprocess
import time

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")


def emit(*items, **_kwargs):
    logging.info(" ".join(map(str, items)))


while "AFTER48 CHECKS COMPLETE" not in (R / "logs/after48_checks.log").read_text():
    time.sleep(3)
command = [
    "g++",
    "-O2",
    "-std=c++17",
    "-UNDEBUG",
    "-I/home/xingrui/workspace/quantized_graph/symphony_research_20261025/src/core",
    str(R / "scripts/cover49_test.cc"),
    "-o",
    str(R / "cover49_test"),
]
(R / "results/cover49_test_build_commands.json").write_text(
    json.dumps([command], indent=2)
)
subprocess.run(command, check=True)
with (R / "logs/cover49_test.log").open("w") as f:
    subprocess.run(
        [str(R / "cover49_test")], stdout=f, stderr=subprocess.STDOUT, check=True
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
    "QG_AUGMENT=2",
]
variants = {
    "cover49_aug2_reference": [],
    "cover49_true": ["QG_COVER_PRUNE=1"],
    "cover49_quantized": ["QG_COVER_PRUNE=2"],
}
for name, flags in variants.items():
    with (R / f"logs/build_{name}.log").open("w") as f:
        subprocess.run(
            ["python3", str(R / "scripts/compile_variant.py"), name, *common, *flags],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
run(
    "screen49_cohere_control_pre",
    "bucket90_w16",
    [800],
    extra={"QG_MARGIN": ".02", "QG_PATIENCE": "0", "QG_CAP": "64"},
    rounds=2,
    duration=2,
    threads="1,8",
)
for name in variants:
    run(
        "screen49_cohere_" + name,
        name,
        [-2000, 0, 2000, 4000, 6000, 8000, 10000, 12000],
        extra={"BENCH_STOP_GRID": "1", "QG_CAP": "64"},
        rounds=1,
        duration=2,
        threads="1,8",
        nav="margin",
    )
run(
    "screen49_cohere_control_post",
    "bucket90_w16",
    [800],
    extra={"QG_MARGIN": ".02", "QG_PATIENCE": "0", "QG_CAP": "64"},
    rounds=2,
    duration=2,
    threads="1,8",
)
emit("SCREEN49 COMPLETE", flush=True)
