from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from explore90 import run

logging.basicConfig(level=logging.INFO, format="%(message)s")

R = Path(__file__).resolve().parent.parent
for key in list(os.environ):
    if key.startswith("QG_INDEPENDENT_"):
        del os.environ[key]
variants = ["independent78_off_v1", "independent78_native_v1"]
baseline = json.loads((R / "variants/bucket90_w16/metadata.json").read_text())[
    "defines"
]
assert (
    json.loads((R / "variants" / variants[0] / "metadata.json").read_text())["defines"]
    == baseline
)
assert json.loads((R / "variants" / variants[1] / "metadata.json").read_text())[
    "defines"
] == [*baseline, "QG_INDEPENDENT_CODE_IMPORT=1"]
reference = R / "experiments/time65_control_pre/search_k10_r0_ids.csv"
expected = reference.read_bytes()
assert len(expected.splitlines()) == 10000
reports = []
for variant in variants:
    name = "check78_disabled_" + variant
    run(
        name,
        variant,
        [20000],
        extra={
            "QG_CAP": "64",
            "QG_PATIENCE": "0",
            "BENCH_STOP_GRID": "1",
            "BENCH_MARGIN_DENOM": "1000",
        },
        rounds=1,
        duration=-1,
        threads="1",
        nav="margin",
    )
    actual_file = R / "experiments" / name / "search_k10_r0_ids.csv"
    actual = actual_file.read_bytes()
    reports.append(
        {
            "variant": variant,
            "file": str(actual_file),
            "top10_rows": len(actual.splitlines()),
            "ids_and_hex_scores_byte_identical": actual == expected,
            "sha256": hashlib.sha256(actual).hexdigest(),
        }
    )
    assert actual == expected, reports[-1]
out = R / "results/independent78_disabled_equivalence.json"
with out.open("x") as stream:
    json.dump(
        {
            "passed": True,
            "reference": str(reference),
            "reference_sha256": hashlib.sha256(expected).hexdigest(),
            "query_count": 1000,
            "topk": 10,
            "timed": False,
            "variants": reports,
        },
        stream,
        indent=2,
    )
    stream.write("\n")
logging.info("DISABLED78 PASS")
