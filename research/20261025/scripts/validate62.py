from __future__ import annotations

import csv
import hashlib
import json
import logging
import statistics
import subprocess
import xml.etree.ElementTree as ET

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")
S = __import__("pathlib").Path(
    "/home/xingrui/workspace/quantized_graph/symphony_research_20261025"
)
assert "SCREEN62 COMPLETE" in (R / "logs/screen62.log").read_text()
assert "RECOVER62 COMPLETE" in (R / "logs/recover62.log").read_text()
plan = json.loads((R / "results/directional62_screen_plan.json").read_text())
selected = {}
builds = {}
for variant, flags in plan["variants"].items():
    rows = []
    for prefix in ["screen62", "recover62"]:
        experiment = prefix + "_cohere_" + variant
        for p in (R / "experiments" / experiment).glob("search_k10_r*.csv"):
            if not p.stem.endswith(("_ids", "_recall")):
                rows.extend(
                    {**row, "experiment": experiment}
                    for row in csv.DictReader(p.open())
                )
    assert len(rows) == 32
    singles = [r for r in rows if r["threads"] == "1"]
    passed = [r for r in singles if float(r["recall"]) >= 0.9]
    if passed:
        label = max(
            {r["ef"] for r in passed},
            key=lambda e: statistics.median(
                float(r["qps"]) for r in passed if r["ef"] == e
            ),
        )
    else:
        label = max(singles, key=lambda r: float(r["recall"]))["ef"]
    chosen = [r for r in rows if r["ef"] == label]
    selected[variant] = {
        "experiment": chosen[0]["experiment"],
        "label": int(label),
        "passing": bool(passed),
        "recall": float(chosen[0]["recall"]),
        "qps_single_median": statistics.median(
            float(r["qps"]) for r in chosen if r["threads"] == "1"
        ),
        "qps_eight_median": statistics.median(
            float(r["qps"]) for r in chosen if r["threads"] == "8"
        ),
    }
    for mode, extras in [("audit", ["QG_CENTER_AUDIT=1"])] + (
        [("full", ["QG_DEFERRED_FULL=1"])] if "integer" in variant else []
    ):
        builds[variant + "_" + mode] = [*flags, "QG_CENTER_STATS=1", *extras]
(R / "results/directional62_selected_primary.json").write_text(
    json.dumps(selected, indent=2) + "\n"
)
integer = json.loads((R / "variants/int55_original/metadata.json").read_text())[
    "defines"
]
builds["dir62_integer_identity"] = [*integer, "QG_DIRECTION_BETA=1."]
for name, flags in builds.items():
    with (R / f"logs/build_{name}.log").open("w") as f:
        subprocess.run(
            ["python3", str(R / "scripts/compile_variant.py"), name, *flags],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
results = []
for variant, chosen in selected.items():
    env = json.loads(
        (R / "experiments" / chosen["experiment"] / "config.json").read_text()
    )["env"]
    extra = {
        k: v
        for k, v in env.items()
        if k.startswith("QG_") or k in ["BENCH_STOP_GRID", "BENCH_MARGIN_DENOM"]
    }
    refs = []
    for roundnum in [0, 1]:
        refs.append(
            sorted(
                r[1:]
                for r in csv.reader(
                    (
                        R
                        / "experiments"
                        / chosen["experiment"]
                        / f"search_k10_r{roundnum}_ids.csv"
                    ).open()
                )
                if r[0] == str(chosen["label"])
            )
        )
    assert len(refs[0]) == 10000
    assert refs[0] == refs[1]
    modes = ["audit", "full"] if "integer" in variant else ["audit"]
    for mode in modes:
        name = variant + "_" + mode
        extra["QG_CENTER_STATS_FILE"] = str(R / f"results/{name}_stats.csv")
        run(
            "diag62_" + name,
            name,
            [chosen["label"]],
            extra=extra,
            duration=-1,
            threads="1",
            nav="margin",
        )
        actual = sorted(
            r[1:]
            for r in csv.reader(
                (R / f"experiments/diag62_{name}/search_k10_r0_ids.csv").open()
            )
        )
        assert actual == refs[0]
    stats = list(csv.reader((R / f"results/{variant}_audit_stats.csv").open()))
    assert len(stats) == 1000
    results.append(
        {
            "variant": variant,
            "queries": 1000,
            "id_score_rows_per_mode": 10000,
            "matching_modes": modes,
            "rounds_identical": True,
            "mean_expanded": statistics.mean(int(r[3]) for r in stats),
            "mean_refined": statistics.mean(int(r[4]) for r in stats),
        }
    )
shared = {
    "QG_CENTER_MEAN_FILE": str(R / "mean54/mean.f32"),
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
    "BENCH_STOP_GRID": "1",
    "BENCH_MARGIN_DENOM": "1000",
}
run(
    "diag62_identity",
    "dir62_integer_identity",
    [20000],
    extra=shared,
    duration=-1,
    threads="1",
    nav="margin",
)
ref = sorted(
    r[1:]
    for r in csv.reader(
        (R / "experiments/screen62_cohere_integer_pre/search_k10_r0_ids.csv").open()
    )
)
actual = sorted(
    r[1:]
    for r in csv.reader(
        (R / "experiments/diag62_identity/search_k10_r0_ids.csv").open()
    )
)
assert len(actual) == 10000
assert ref == actual
s = (
    (R / "scripts/validate_stage58_default.py")
    .read_text()
    .replace("stage58", "stage62")
    .replace("STAGE58", "STAGE62")
)
(R / "scripts/validate_stage62_default.py").write_text(s)
with (R / "logs/validate_stage62_default.log").open("w") as f:
    subprocess.run(
        ["python3", str(R / "scripts/validate_stage62_default.py")],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
root = ET.parse(R / "results/stage62_default_tests.xml").getroot()
assert int(root.attrib["tests"]) == 28
assert int(root.attrib["failures"]) == 0
assert int(root.attrib["errors"]) == 0
verified = {}
for name in [*plan["variants"], *builds, "stage62_default"]:
    hashes = json.loads((R / f"variants/{name}/source_files_sha256.json").read_text())
    for path, digest in hashes.items():
        assert hashlib.sha256((S / path).read_bytes()).hexdigest() == digest
    verified[name] = len(hashes)
summary = {
    "results": results,
    "identity_queries": 1000,
    "identity_id_score_rows": 10000,
    "identity_equal": True,
    "default_tests": 28,
    "source_manifests_verified": verified,
    "breakthrough_verified": False,
    "extra_queries_not_evaluated": True,
}
(R / "results/directional62_actual_validation.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
logging.info("%s", "VALIDATE62 COMPLETE")
