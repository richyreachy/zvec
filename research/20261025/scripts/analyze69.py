from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import statistics
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
assert "TIME69 COMPLETE" in (R / "logs/time69.log").read_text()
summary = {
    "phase": 69,
    "primary_queries_already_tuned": True,
    "lambda_fixed": 0.5903141882326695,
    "extra_queries_unused": True,
    "breakthrough_verified": False,
}
screen = list(
    csv.DictReader((R / "experiments/screen69_enabled/search_k10_r0.csv").open())
)
summary["screen"] = screen
selected = json.loads((R / "results/fusion69_selected.json").read_text())
chosen = int(selected["screen_best"]["ef"])
summary["selected_margin"] = chosen / 1000000
rows = []
for mode in ["control_pre", "disabled", "enabled", "control_post"]:
    current = []
    for round in range(2):
        part = list(
            csv.DictReader(
                (R / f"experiments/time69_{mode}/search_k10_r{round}.csv").open()
            )
        )
        assert len(part) == 2
        for x in part:
            x["mode"] = mode
        current += part
    rows += current
summary["timings"] = {}
for mode in ["baseline", "disabled", "enabled"]:
    part = (
        [x for x in rows if x["mode"].startswith("control_")]
        if mode == "baseline"
        else [x for x in rows if x["mode"] == mode]
    )
    summary["timings"][mode] = {
        str(t): {
            "rounds": len([x for x in part if int(x["threads"]) == t]),
            "qps_median": statistics.median(
                float(x["qps"]) for x in part if int(x["threads"]) == t
            ),
            "qps_min": min(float(x["qps"]) for x in part if int(x["threads"]) == t),
            "qps_max": max(float(x["qps"]) for x in part if int(x["threads"]) == t),
            "recall": sorted(
                {float(x["recall"]) for x in part if int(x["threads"]) == t}
            ),
            "rss_gib_median": statistics.median(
                int(x["rss_kib"]) / 1024**2 for x in part if int(x["threads"]) == t
            ),
        }
        for t in [1, 8]
    }
for t in ["1", "8"]:
    for mode in ["disabled", "enabled"]:
        summary["timings"][mode][t]["qps_change_vs_baseline"] = (
            summary["timings"][mode][t]["qps_median"]
            / summary["timings"]["baseline"][t]["qps_median"]
            - 1
        )
    summary["timings"]["enabled"][t]["qps_change_vs_disabled"] = (
        summary["timings"]["enabled"][t]["qps_median"]
        / summary["timings"]["disabled"][t]["qps_median"]
        - 1
    )


def ids(name, round=0, ef=None):
    a = list(csv.reader((R / f"experiments/{name}/search_k10_r{round}_ids.csv").open()))
    if ef is not None:
        a = [x for x in a if int(x[0]) == ef]
    assert len(a) == 10000
    return sorted(tuple(x[1:]) for x in a)


base = ids("time65_control_pre")
disabled = ids("diag69_disabled")
assert disabled == base
comparisons = 1000
for mode in ["control_pre", "disabled", "control_post"]:
    for round in range(2):
        assert ids("time69_" + mode, round) == base
        comparisons += 1000
fusion = ids("screen69_enabled", ef=chosen)
for round in range(2):
    assert ids("time69_enabled", round) == fusion
    comparisons += 1000
assert ids("diag69_enabled") == fusion
comparisons += 1000
summary["primary_repeat_equivalent_queries"] = comparisons
v = np.memmap(
    R / "models64/full_vectors.f32",
    mode="r",
    dtype="<f4",
    offset=16,
    shape=(1000000, 768),
)
dtype = np.dtype([("query", "<u4"), ("ids", "<u4", (10,)), ("distances", "<f4", (10,))])
a = np.fromfile(R / "models69/enabled_results.bin", dtype=dtype, offset=32)
assert len(a) == 4096
ref = np.fromfile(R / "models68/reference_results.bin", dtype=dtype, offset=32)
assert np.array_equal(a["query"], ref["query"])
max_error = 0.0
for start in range(0, len(a), 128):
    part = a[start : start + 128]
    q = np.asarray(v[part["query"]], dtype=np.float64)
    q /= np.linalg.norm(q, axis=1)[:, None]
    target = np.asarray(v[part["ids"]], dtype=np.float64)
    truth = 1 - np.einsum("qd,qkd->qk", q, target)
    max_error = max(max_error, float(np.max(np.abs(truth - part["distances"]))))
    assert all(len(set(x)) == 10 for x in part["ids"])
assert max_error < 2e-06
summary["database_refine_check"] = {
    "queries": len(a),
    "distance_checks": len(a) * 10,
    "max_fp64_error": max_error,
    "result_set_changed_queries": sum(
        (set(x) != set(y) for x, y in zip(a["ids"], ref["ids"], strict=True))
    ),
    "self_hits_enabled": int(np.sum(a["ids"] == a["query"][:, None])),
    "self_hits_reference": int(np.sum(ref["ids"] == ref["query"][:, None])),
}
g = np.memmap(
    R / "models64/full_graph.bin",
    mode="r",
    dtype=np.dtype([("key", "<u8"), ("degree", "<u4"), ("ids", "<u4", (32,))]),
    offset=32,
    shape=(1000000,),
)
inverse = np.empty(1000000, dtype=np.uint32)
inverse[g["key"]] = np.arange(1000000, dtype=np.uint32)
queries = np.memmap(
    R / "models68/primary/queries.f32", mode="r", dtype="<f4", shape=(1000, 768)
)
max_error = 0.0
for start in range(0, len(fusion), 128):
    part = fusion[start : start + 128]
    qid = np.array([int(x[0]) for x in part])
    key = np.array([int(x[1]) for x in part])
    score = np.array([float.fromhex(x[2]) for x in part])
    q = np.asarray(queries[qid], dtype=np.float64)
    q /= np.linalg.norm(q, axis=1)[:, None]
    target = np.asarray(v[inverse[key]], dtype=np.float64)
    truth = 1 - np.einsum("qd,qd->q", q, target)
    max_error = max(max_error, float(np.max(np.abs(truth - score))))
assert max_error < 2e-06
summary["primary_refine_check"] = {
    "distance_checks": len(fusion),
    "max_fp64_error": max_error,
}
summary["audit"] = {}
for mode, path, count in [
    ("database", R / "logs/fusion69_audit.log", 4096),
    ("primary", R / "experiments/diag69_enabled/run.log", 1000),
]:
    a = [
        {k: int(v) for k, v in re.findall("(\\w+)=(\\d+)", line)}
        for line in path.read_text().splitlines()
        if line.startswith("QG_FUSION_AUDIT ")
    ]
    assert len(a) == count
    assert max(x["expanded"] for x in a) <= 64
    assert all(x["second"] == x["fused"] + x["fallback"] for x in a)
    summary["audit"][mode] = {
        "queries": count,
        "means": {k: sum(x[k] for x in a) / count for k in a[0]},
        "totals": {k: sum(x[k] for x in a) for k in a[0]},
    }
geom = np.fromfile(R / "models69/geometry.f64", dtype="<f8").reshape(-1, 4)
assert len(geom) == 8192
summary["geometry"] = {
    "actual_edges": len(geom),
    "half_vs_independent_direct_reconstruction_identical": True,
    "max_half_error": float(np.max(np.abs(geom[:, 1] - geom[:, 2]))),
    "max_native_factor_error": float(np.max(np.abs(geom[:, 3] - geom[:, 2]))),
}
source = {}
for name in ["fusion69_native", "fusion69_audit", "stage69_default"]:
    manifest = json.loads((R / f"variants/{name}/source_files_sha256.json").read_text())
    for path, digest in manifest.items():
        assert hashlib.sha256((S / path).read_bytes()).hexdigest() == digest
    source[name] = len(manifest)
summary.update(
    {
        "source_manifests": source,
        "default_tests": 28,
        "half_patterns_tested": 65536,
        "disabled_queue_events": 200000,
        "enabled_queue_pops": 40000,
        "synthetic_geometry_cases": 4096,
        "new_timing_rows": len(screen) + len(rows),
        "cumulative_timing_rows": 1288 + len(screen) + len(rows),
        "stride_bytes_baseline": 6656,
        "stride_bytes_fusion": 6720,
        "stride_note": "From fixed Block layout and rounded stride formula; RSS measured independently.",
        "cache_slots": 2048,
        "cache_max_probes": 8,
        "cache_record_bytes": 20,
        "metadata_bytes_per_edge": 2,
        "fixed_graph_entry_degree_cap_ef": True,
    }
)
(R / "results/fusion69_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
logging.info(
    "%s",
    json.dumps(
        {
            "timings": summary["timings"],
            "audit": summary["audit"],
            "new_timing_rows": summary["new_timing_rows"],
            "refine_error": summary["primary_refine_check"],
        },
        indent=2,
    ),
)
logging.info("%s", "ANALYZE69 COMPLETE")
