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
assert "SCREEN59 COMPLETE" in (R / "logs/screen59.log").read_text()
assert "VALIDATE59 COMPLETE" in (R / "logs/validate59.log").read_text()
rows = []
groups = []
manifest = {}
qcount = 0
for d in sorted((R / "experiments").glob("screen59_*")):
    rr = []
    for p in sorted(d.glob("search_k10_r*.csv")):
        manifest[str(p.relative_to(R))] = hashlib.sha256(p.read_bytes()).hexdigest()
        if p.stem.endswith("_recall"):
            records = list(csv.DictReader(p.open()))
            assert len(records) % 1000 == 0
            qcount += len(records)
            for label in {r["ef"] for r in records}:
                selected = [r for r in records if r["ef"] == label]
                assert len(selected) == 1000
                assert {int(r["query_id"]) for r in selected} == set(range(1000))
        elif not p.stem.endswith("_ids"):
            for row in csv.DictReader(p.open()):
                rr.append({**row, "experiment": d.name})
    rows.extend(rr)
    for t in ["1", "8"]:
        passing = [r for r in rr if r["threads"] == t and float(r["recall"]) >= 0.9]
        if not passing:
            continue
        label = max(
            {r["ef"] for r in passing},
            key=lambda e: st.median(float(r["qps"]) for r in passing if r["ef"] == e),
        )
        chosen = [r for r in passing if r["ef"] == label]
        groups.append(
            {
                "experiment": d.name,
                "threads": int(t),
                "label": int(label),
                "recall": float(chosen[0]["recall"]),
                "qps_median": st.median(float(r["qps"]) for r in chosen),
                "rss_gib": st.median(float(r["rss_kib"]) / 1048576 for r in chosen),
                "rounds": len(chosen),
            }
        )
assert len(rows) == 60
assert qcount == 31000
root = ET.parse(R / "results/stage59_default_tests.xml").getroot()
assert int(root.attrib["tests"]) == 28
assert int(root.attrib["failures"]) == 0
assert int(root.attrib["errors"]) == 0
verified = {}
for name in ["hier59_integer", "hier59_audit", "hier59_full", "stage59_default"]:
    h = json.loads((R / f"variants/{name}/source_files_sha256.json").read_text())
    for p, v in h.items():
        assert hashlib.sha256((S / p).read_bytes()).hexdigest() == v, (name, p)
    verified[name] = len(h)
summary = {
    "timing_rows": len(rows),
    "query_evaluations_in_screen": qcount,
    "extra_queries_used": False,
    "breakthrough_verified": False,
    "passing_groups": groups,
    "rows": rows,
    "source_manifests_verified": verified,
    "default_tests": 28,
    "unit_log": (R / "logs/integer_router59_test.log").read_text().strip(),
    "actual_validation": json.loads(
        (R / "results/router59_actual_validation.json").read_text()
    ),
    "disabled_equivalence": json.loads(
        (R / "results/router59_disabled_equivalence.json").read_text()
    ),
}
(R / "results/router59_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(R / "results/router59_artifact_sha256.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
logging.info("%s %s %s %s", "FINALIZE59 COMPLETE", len(rows), qcount, verified)
