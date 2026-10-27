from __future__ import annotations

import json
import logging
import subprocess

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")
assert "TEST62 COMPLETE" in (R / "logs/test62.log").read_text()
native = json.loads((R / "variants/bucket90_w16/metadata.json").read_text())["defines"]
integer = json.loads((R / "variants/int55_original/metadata.json").read_text())[
    "defines"
]
variants = {
    f"dir62_{kind}_{mode}": [*base, flag]
    for kind, base in [("native", native), ("integer", integer)]
    for mode, flag in [
        ("trained", "QG_DIRECTION_BETA=.9359792556459329"),
        ("mse", "QG_DIRECTION_MSE=1"),
    ]
}
(R / "results/directional62_screen_plan.json").write_text(
    json.dumps(
        {
            "variants": variants,
            "labels": [0, 10000, 20000, 40000, 60000],
            "rounds": 2,
            "duration_seconds": 3,
            "threads": [1, 8],
            "trained_beta_frozen_before_query_evaluation": 0.9359792556459329,
            "primary_queries_not_blind": True,
        },
        indent=2,
    )
    + "\n"
)
for name, flags in variants.items():
    with (R / f"logs/build_{name}.log").open("w") as f:
        subprocess.run(
            ["python3", str(R / "scripts/compile_variant.py"), name, *flags],
            stdout=f,
            stderr=subprocess.STDOUT,
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
    "screen62_cohere_control_pre",
    "bucket90_w16",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
run(
    "screen62_cohere_integer_pre",
    "int55_original",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
for name in variants:
    run(
        "screen62_cohere_" + name,
        name,
        [0, 10000, 20000, 40000, 60000],
        extra=shared,
        rounds=2,
        duration=3,
        threads="1,8",
        nav="margin",
    )
run(
    "screen62_cohere_integer_post",
    "int55_original",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
run(
    "screen62_cohere_control_post",
    "bucket90_w16",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
logging.info("%s", "SCREEN62 COMPLETE")
