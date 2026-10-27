from __future__ import annotations

import csv
import json
import logging
import subprocess

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")

common = json.loads((R / "variants/int55_original/metadata.json").read_text())[
    "defines"
]
variants = {
    "int56_pref3072": ["QG_PREFETCH_BATCH=1", "QG_PREFETCH_BYTES=3072"],
    "int56_pref4224": ["QG_PREFETCH_BATCH=1", "QG_PREFETCH_BYTES=4224"],
    "int56_lean3072": [
        "QG_PREFETCH_BATCH=1",
        "QG_PREFETCH_BYTES=3072",
        "QG_VISIT_EXPECT=64",
        "QG_RESULT_TOPK=1",
        "QG_RESULT_GATE=1",
    ],
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
            ["python3", str(R / "scripts/compile_variant.py"), name, *common, *flags],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
run(
    "screen56_cohere_control_pre",
    "bucket90_w16",
    [800],
    extra=env,
    rounds=3,
    duration=3,
    threads="1,8",
)
run(
    "screen56_cohere_integer_control",
    "int55_original",
    [800],
    extra=env,
    rounds=3,
    duration=3,
    threads="1,8",
)
for name in variants:
    run(
        "screen56_cohere_" + name,
        name,
        [800],
        extra=env,
        rounds=3,
        duration=3,
        threads="1,8",
    )
run(
    "screen56_cohere_control_post",
    "bucket90_w16",
    [800],
    extra=env,
    rounds=3,
    duration=3,
    threads="1,8",
)
reference = sorted(
    r[1:]
    for r in csv.reader(
        (R / "experiments/screen55_cohere_int55_original/search_k10_r0_ids.csv").open()
    )
    if r[0] == "20000"
)
assert len(reference) == 10000
checks = []
for name in ["integer_control", *variants]:
    directory = R / ("experiments/screen56_cohere_" + name)
    for f in sorted(directory.glob("search_k10_r*_ids.csv")):
        rows = sorted(r[1:] for r in csv.reader(f.open()))
        assert rows == reference
        checks.append(
            {
                "file": str(f.relative_to(R)),
                "queries": 1000,
                "id_score_rows": 10000,
                "exact_match": True,
            }
        )
(R / "results/integer_prefetch56_equivalence.json").write_text(
    json.dumps(checks, indent=2)
)
logging.info("SCREEN56 COMPLETE")
