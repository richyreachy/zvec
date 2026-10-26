from __future__ import annotations

import csv
import json
import logging

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")
assert "SCREEN69 COMPLETE" in (R / "logs/screen69.log").read_text()
rows = list(
    csv.DictReader((R / "experiments/screen69_enabled/search_k10_r0.csv").open())
)
valid = [x for x in rows if float(x["recall"]) >= 0.9]
assert valid
best = max(valid, key=lambda x: float(x["qps"]))
margin = int(best["ef"])
(R / "results/fusion69_selected.json").write_text(
    json.dumps(
        {
            "screen_best": best,
            "chosen_margin": margin / 1000000,
            "fixed_lambda": 0.5903141882326695,
            "primary_not_blind": True,
            "selection_only_uses_normal_margin_grid": True,
        },
        indent=2,
    )
    + "\n"
)
shared = {
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
    "BENCH_STOP_GRID": "1",
    "BENCH_MARGIN_DENOM": "1000",
}
for name, variant, enable, ef in [
    ("control_pre", "bucket90_w16", "0", 20000),
    ("disabled", "fusion69_native", "0", 20000),
    ("enabled", "fusion69_native", "1", margin),
    ("control_post", "bucket90_w16", "0", 20000),
]:
    run(
        "time69_" + name,
        variant,
        [ef],
        extra={**shared, "QG_FUSION_ENABLE": enable},
        rounds=2,
        duration=3,
        threads="1,8",
        nav="margin",
    )
run(
    "diag69_enabled",
    "fusion69_audit",
    [margin],
    extra={
        **shared,
        "QG_FUSION_ENABLE": "1",
        "QG_TRACE": str(R / "models69/primary_diag.csv"),
    },
    duration=-1,
    threads="1",
    nav="margin",
)
logging.info("%s", "TIME69 COMPLETE")
