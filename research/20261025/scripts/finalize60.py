from __future__ import annotations

import csv
import hashlib
import json
import logging
import statistics as st
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO)
R = Path(__file__).resolve().parent.parent
S = Path("/home/xingrui/workspace/quantized_graph/symphony_research_20261025")
assert "SCREEN60 COMPLETE" in (R / "logs/screen60.log").read_text()
assert "VALIDATE60 COMPLETE" in (R / "logs/validate60.log").read_text()
rows = []
groups = []
manifest = {}
qcount = 0
for directory in sorted((R / "experiments").glob("screen60_*")):
    current = []
    for path in sorted(directory.glob("search_k10_r*.csv")):
        manifest[str(path.relative_to(R))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        if path.stem.endswith("_recall"):
            records = list(csv.DictReader(path.open()))
            qcount += len(records)
            for label in {r["ef"] for r in records}:
                selected = [r for r in records if r["ef"] == label]
                assert len(selected) == 1000
                assert {int(r["query_id"]) for r in selected} == set(range(1000))
        elif not path.stem.endswith("_ids"):
            current.extend(
                {**r, "experiment": directory.name} for r in csv.DictReader(path.open())
            )
    rows.extend(current)
    for threads in ["1", "8"]:
        passing = [
            r for r in current if r["threads"] == threads and float(r["recall"]) >= 0.9
        ]
        if not passing:
            continue
        label = max(
            {r["ef"] for r in passing},
            key=lambda e: st.median(float(r["qps"]) for r in passing if r["ef"] == e),
        )
        chosen = [r for r in passing if r["ef"] == label]
        groups.append(
            {
                "experiment": directory.name,
                "threads": int(threads),
                "label": int(label),
                "recall": float(chosen[0]["recall"]),
                "qps_median": st.median(float(r["qps"]) for r in chosen),
                "rss_gib": st.median(float(r["rss_kib"]) / 1048576 for r in chosen),
                "rounds": len(chosen),
            }
        )
assert len(rows) == 64
assert qcount == 33000
root = ET.parse(R / "results/stage60_default_tests.xml").getroot()
assert int(root.attrib["tests"]) == 28
assert int(root.attrib["failures"]) == 0
assert int(root.attrib["errors"]) == 0
change = json.loads((R / "results/router60_prebuild_guard_change.json").read_text())
header = S / change["file"]
current = header.read_text()
assert hashlib.sha256(header.read_bytes()).hexdigest() == change["after_sha256"]
assert current.count(change["added_code"]) == 1
assert (
    hashlib.sha256(current.replace(change["added_code"], "").encode()).hexdigest()
    == change["before_sha256"]
)
verified = {}
exceptions = {}
for dim in [768, 384, 256]:
    for suffix in ["", "_audit", "_full"]:
        name = f"proj60_{dim}{suffix}"
        h = json.loads((R / f"variants/{name}/source_files_sha256.json").read_text())
        exception = []
        for path, digest in h.items():
            actual = hashlib.sha256((S / path).read_bytes()).hexdigest()
            if actual != digest:
                assert not suffix
                assert path == change["file"]
                assert digest == change["before_sha256"]
                exception.append(path)
        verified[name] = len(h)
        exceptions[name] = exception
name = "stage60_default"
h = json.loads((R / f"variants/{name}/source_files_sha256.json").read_text())
assert all(
    (hashlib.sha256((S / p).read_bytes()).hexdigest() == v for p, v in h.items())
)
verified[name] = len(h)
summary = {
    "timing_rows": len(rows),
    "screen_query_evaluations": qcount,
    "breakthrough_verified": False,
    "extra_queries_used": False,
    "passing_groups": groups,
    "rows": rows,
    "default_tests": 28,
    "prebuild_guard_checks": 7,
    "actual_validation": json.loads(
        (R / "results/router60_actual_validation.json").read_text()
    ),
    "source_manifests_checked": verified,
    "timed_source_exceptions": exceptions,
    "exception_reason": "Only prebuild compatibility guard added after timed snapshots;exact patch removed reproduces original header hash;all online query code unchanged;post-guard audit/full outputs equal timed outputs",
    "model_preparation": json.loads(
        (R / "results/router60_preparation.json").read_text()
    ),
}
(R / "results/router60_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(R / "results/router60_artifact_sha256.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
logging.info("%s %s %s %s", "FINALIZE60 COMPLETE", len(rows), qcount, verified)
