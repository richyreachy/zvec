from __future__ import annotations

import csv
import hashlib
import json
import logging
import xml.etree.ElementTree as ET

from explore90 import R, run

logging.basicConfig(level=logging.INFO, format="%(message)s")
assert "BUILD65 AND UNIT VALIDATION COMPLETE" in (R / "logs/build65.log").read_text()
tree = ET.parse(R / "results/stage65_default_tests.xml").getroot()
assert int(tree.attrib["tests"]) == 28
assert int(tree.attrib["failures"]) == 0
assert int(tree.attrib["errors"]) == 0
labels = [0, 10000, 20000, 40000, 60000, 80000, 120000]
shared = {
    "QG_CENTER_MEAN_FILE": str(R / "mean54/mean.f32"),
    "QG_CAP": "64",
    "QG_PATIENCE": "0",
    "BENCH_STOP_GRID": "1",
    "BENCH_MARGIN_DENOM": "1000",
}
plan = {
    "labels": labels,
    "actual_ef": 800,
    "cap": 64,
    "topk": 10,
    "primary_queries_not_blind": True,
    "diagnostic_duration": -1,
    "candidate_pool_fixed_before_query_results": True,
    "new_models": json.loads((R / "results/graph65_full_validation.json").read_text()),
    "no_retrieval_timing_in_this_script": True,
}
(R / "results/graph65_screen_plan.json").write_text(json.dumps(plan, indent=2) + "\n")


def ids(name, label=20000):
    return sorted(
        r[1:]
        for r in csv.reader((R / "experiments" / name / "search_k10_r0_ids.csv").open())
        if r[0] == str(label)
    )


validation = {}
for kind, reference in [
    ("native", "screen62_cohere_control_pre"),
    ("integer", "screen62_cohere_integer_pre"),
]:
    name = "diag65_disabled_" + kind
    run(
        name,
        "graph65_" + kind,
        [20000],
        extra=shared,
        duration=-1,
        threads="1",
        nav="margin",
    )
    assert ids(name) == ids(reference)
    assert len(ids(name)) == 10000
    validation[kind + "_disabled_matches_historical_queries"] = 1000
for mode in ("nearest", "angular"):
    extra = {
        **shared,
        "QG_GRAPH_IMPORT_FILE": str(R / f"models65/full_{mode}.bin"),
        "QG_GRAPH_EXPORT_FILE": str(R / f"models65/imported_{mode}.bin"),
        "QG_GRAPH_VECTOR_FILE": str(R / f"models65/imported_{mode}_vectors.f32"),
    }
    run(
        "diag65_native_" + mode,
        "graph65_exportcheck",
        labels,
        extra=extra,
        duration=-1,
        threads="1",
        nav="margin",
    )
    digest = hashlib.sha256(
        (R / f"models65/imported_{mode}.bin").read_bytes()
    ).hexdigest()
    assert digest == plan["new_models"][mode + "_sha256"]
    validation[mode + "_imported_adjacency_sha256"] = digest
    vsha = hashlib.sha256(
        (R / f"models65/imported_{mode}_vectors.f32").read_bytes()
    ).hexdigest()
    assert (
        vsha
        == json.loads((R / "results/graph64_full_validation.json").read_text())[
            "sha256"
        ]["models64/full_vectors.f32"]
    )
    validation[mode + "_exported_vectors_match_original"] = True
    extra = {**shared, "QG_GRAPH_IMPORT_FILE": str(R / f"models65/full_{mode}.bin")}
    run(
        "diag65_integer_" + mode,
        "graph65_integer",
        labels,
        extra=extra,
        duration=-1,
        threads="1",
        nav="margin",
    )
recalls = {}
for kind in ("native", "integer"):
    for mode in ("nearest", "angular"):
        name = f"diag65_{kind}_{mode}"
        rows = list(
            csv.DictReader(
                (R / "experiments" / name / "search_k10_r0_recall.csv").open()
            )
        )
        recalls[name] = {
            str(label): sum(int(r["hits"]) for r in rows if r["ef"] == str(label))
            / 10000
            for label in labels
        }
(R / "results/graph65_diagnostic_recalls.json").write_text(
    json.dumps(recalls, indent=2) + "\n"
)
(R / "results/graph65_import_validation.json").write_text(
    json.dumps(validation, indent=2) + "\n"
)
logging.info("%s", "SCREEN65 COMPLETE")
