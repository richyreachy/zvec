from __future__ import annotations

import json
import logging

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")
assert "SCREEN62 COMPLETE" in (R / "logs/screen62.log").read_text()
plan = json.loads((R / "results/directional62_screen_plan.json").read_text())
shared = {
    "QG_CENTER_MEAN_FILE": str(R / "mean54/mean.f32"),
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
    "BENCH_STOP_GRID": "1",
    "BENCH_MARGIN_DENOM": "1000",
}
labels = {
    name: [65000, 70000, 80000]
    if name.endswith("trained")
    else [200000, 400000, 800000]
    for name in plan["variants"]
}
(R / "results/directional62_recovery_plan.json").write_text(
    json.dumps(
        {
            "reason": "Initial trained-native margin0.06 reaches88.98%; directional shrink changes absolute neighbor scores and triggers earlier stopping. Extend stopping margin to compare at passing recall. This is explicit primary-query development, not held-out evaluation.",
            "labels": labels,
            "rounds": 2,
            "duration_seconds": 3,
            "threads": [1, 8],
            "model_beta_unchanged": True,
        },
        indent=2,
    )
    + "\n"
)
run(
    "recover62_cohere_control_pre",
    "bucket90_w16",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
for name, grid in labels.items():
    run(
        "recover62_cohere_" + name,
        name,
        grid,
        extra=shared,
        rounds=2,
        duration=3,
        threads="1,8",
        nav="margin",
    )
run(
    "recover62_cohere_integer_control",
    "int55_original",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
run(
    "recover62_cohere_control_post",
    "bucket90_w16",
    [20000],
    extra=shared,
    rounds=2,
    duration=3,
    threads="1,8",
    nav="margin",
)
logging.info("%s", "RECOVER62 COMPLETE")
