from __future__ import annotations

import csv
import hashlib
import json
import logging
import statistics
import subprocess
from pathlib import Path

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
assert "TIME65 COMPLETE" in (R / "logs/time65.log").read_text()
selected = {}
allrows = []
for kind in ("native", "integer"):
    for mode in ("nearest", "angular"):
        name = "time65_" + kind + "_" + mode
        rows = []
        for p in (R / "experiments" / name).glob("search_k10_r*.csv"):
            if not p.stem.endswith(("_ids", "_recall")):
                rows.extend(dict(r, experiment=name) for r in csv.DictReader(p.open()))
        assert len(rows) == 12
        allrows.extend(rows)
        singles = [r for r in rows if r["threads"] == "1"]
        passing = [r for r in singles if float(r["recall"]) >= 0.9]
        assert passing, "No passing timing; revise explicitly"
        label = max(
            {r["ef"] for r in passing},
            key=lambda e: statistics.median(
                float(r["qps"]) for r in passing if r["ef"] == e
            ),
        )
        rr = [r for r in rows if r["ef"] == label]
        selected[kind + "_" + mode] = {
            "experiment": name,
            "label": int(label),
            "recall": float(rr[0]["recall"]),
            "qps_single_median": statistics.median(
                float(r["qps"]) for r in rr if r["threads"] == "1"
            ),
            "qps_eight_median": statistics.median(
                float(r["qps"]) for r in rr if r["threads"] == "8"
            ),
            "rss_gib": max(int(r["rss_kib"]) for r in rr) / 2**20,
        }
controls = {}
for kind in ("control", "integer"):
    rows = []
    for side in ("pre", "post"):
        name = f"time65_{kind}_{side}"
        for p in (R / "experiments" / name).glob("search_k10_r*.csv"):
            if not p.stem.endswith(("_ids", "_recall")):
                rows.extend(dict(r, experiment=name) for r in csv.DictReader(p.open()))
    assert len(rows) == 8
    allrows.extend(rows)
    controls[kind] = {
        "recall": float(rows[0]["recall"]),
        "qps_single_median": statistics.median(
            float(r["qps"]) for r in rows if r["threads"] == "1"
        ),
        "qps_eight_median": statistics.median(
            float(r["qps"]) for r in rows if r["threads"] == "8"
        ),
    }
summary = {
    "selected": selected,
    "controls": controls,
    "timing_rows": len(allrows),
    "primary_tuned": True,
    "rounds_per_candidate": 2,
    "breakthrough_verified": False,
}
(R / "results/graph65_selected.json").write_text(json.dumps(summary, indent=2) + "\n")
with (R / "results/graph65_timings.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(allrows[0]))
    writer.writeheader()
    writer.writerows(allrows)
builds = {}
for kind in ("native", "integer"):
    base = json.loads((R / f"variants/graph65_{kind}/metadata.json").read_text())[
        "defines"
    ]
    builds["graph65_" + kind + "_audit"] = (
        base
        + ["QG_CENTER_STATS=1"]
        + (["QG_CENTER_AUDIT=1"] if kind == "integer" else [])
    )
    if kind == "integer":
        builds["graph65_integer_full"] = [
            *base,
            "QG_CENTER_STATS=1",
            "QG_DEFERRED_FULL=1",
        ]
for name, flags in builds.items():
    with (R / f"logs/build_{name}.log").open("w") as f:
        subprocess.run(
            ["python3", str(R / "scripts/compile_variant.py"), name, *flags],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )


def ids(name, roundnum, label):
    return sorted(
        r[1:]
        for r in csv.reader(
            (R / "experiments" / name / f"search_k10_r{roundnum}_ids.csv").open()
        )
        if r[0] == str(label)
    )


checks = []
for key, choice in selected.items():
    kind, mode = key.split("_")
    label = choice["label"]
    reference = ids(choice["experiment"], 0, label)
    assert len(reference) == 10000
    assert reference == ids(choice["experiment"], 1, label)
    all_labels = [14000, 16000, 20000] if mode == "nearest" else [18000, 20000, 22000]
    for each_label in all_labels:
        left = ids(choice["experiment"], 0, each_label)
        right = ids(choice["experiment"], 1, each_label)
        assert len(left) == 10000
        assert left == right
    for margin in (20000,):
        assert ids(choice["experiment"], 0, margin) == ids("diag65_" + key, 0, margin)
    modes = ("audit", "full") if kind == "integer" else ("audit",)
    stats_result = {}
    for checkmode in modes:
        name = "diag65_" + key + "_" + checkmode
        statsfile = R / f"results/{name}_stats.csv"
        extra = {
            "QG_CENTER_MEAN_FILE": str(R / "mean54/mean.f32"),
            "QG_CAP": "64",
            "QG_PATIENCE": "0",
            "BENCH_STOP_GRID": "1",
            "BENCH_MARGIN_DENOM": "1000",
            "QG_GRAPH_IMPORT_FILE": str(R / f"models65/full_{mode}.bin"),
            "QG_CENTER_STATS_FILE": str(statsfile),
        }
        run(
            name,
            "graph65_" + kind + "_" + checkmode,
            [label],
            extra=extra,
            duration=-1,
            threads="1",
            nav="margin",
        )
        assert reference == ids(name, 0, label)
        rows = list(csv.reader(statsfile.open()))
        assert len(rows) == 1000
        stats_result[checkmode] = {
            "mean_expanded": statistics.mean(int(r[3]) for r in rows),
            "mean_refined": statistics.mean(int(r[4]) for r in rows),
            "stride": int(rows[0][6]),
        }
    checks.append(
        {
            "candidate": key,
            "label": label,
            "queries_per_mode": 1000,
            "id_score_rows_per_mode": 10000,
            "rounds_identical": True,
            "matches_diagnostic_at_margin_002": True,
            "stats": stats_result,
        }
    )
verified = {}
for name in [
    "graph65_native",
    "graph65_integer",
    "graph65_exportcheck",
    "stage65_default",
    *builds,
]:
    manifest = json.loads((R / f"variants/{name}/source_files_sha256.json").read_text())
    for path, digest in manifest.items():
        assert hashlib.sha256((S / path).read_bytes()).hexdigest() == digest
    verified[name] = len(manifest)
for control_kind, reference_name in [
    ("control", "screen62_cohere_control_pre"),
    ("integer", "screen62_cohere_integer_pre"),
]:
    for side in ("pre", "post"):
        for roundnum in (0, 1):
            assert ids(f"time65_{control_kind}_{side}", roundnum, 20000) == ids(
                reference_name, 0, 20000
            )
report = {
    "candidate_round_query_checks": 12000,
    "control_query_checks": 8000,
    "query_checks": checks,
    "default_tests": 28,
    "source_manifests": verified,
    "new_graph_export_byte_identity": True,
    "disabled_2000_queries_identical": True,
    "extra_queries_not_evaluated": True,
    "breakthrough_verified": False,
}
(R / "results/graph65_search_validation.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
logging.info("%s", "VALIDATE65 COMPLETE")
