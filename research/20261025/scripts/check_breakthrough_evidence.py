from __future__ import annotations

import csv
import hashlib
import json
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")


def emit(*items, **_kwargs):
    logging.info(" ".join(map(str, items)))


R = Path(__file__).resolve().parent.parent


def read_ids(path, label=None):
    a = defaultdict(list)
    for r in csv.reader(path.open()):
        if label is not None and int(r[0]) != label:
            continue
        a[int(r[1]) if label is not None else (int(r[0]), int(r[1]))].append(
            (int(r[2]), float.fromhex(r[3]))
        )
    return {q: sorted(items) for q, items in a.items()}


def compare(a, b, label_a=None, label_b=None):
    aa = read_ids(a, label_a)
    bb = read_ids(b, label_b)
    keys = set(aa) | set(bb)
    mismatch = [q for q in keys if aa.get(q) != bb.get(q)]
    score_mismatch = [
        q
        for q in keys
        if sorted(x[1] for x in aa.get(q, [])) != sorted(x[1] for x in bb.get(q, []))
    ]
    return {
        "reference": str(a.relative_to(R)),
        "candidate": str(b.relative_to(R)),
        "queries_reference": len(aa),
        "queries_candidate": len(bb),
        "id_score_mismatch_count": len(mismatch),
        "score_multiset_mismatch_count": len(score_mismatch),
        "first_mismatch_keys": mismatch[:10],
    }


checks = []
base = R / "experiments/round3_bucket90_w16/search_k10_r0_ids.csv"
for phase in [36, 42]:
    for d in sorted((R / "experiments").glob(f"screen{phase}_*")):
        for f in sorted(d.glob("*_ids.csv")):
            checks.append(compare(base, f, 800, 800))
p = R / "experiments/screen40_cohere_expand_batch1/search_k10_r0_ids.csv"
if p.exists():
    checks.append(compare(base, p, 800, 2000))
a = R / "experiments/screen43_cohere_tail_bucket_fast/search_k10_r0_ids.csv"
b = R / "experiments/screen43_cohere_tail_bucket_batch/search_k10_r0_ids.csv"
if a.exists() and b.exists():
    checks.append(compare(a, b))
for kind in ["half", "sq8"]:
    a = R / f"experiments/breakthrough_{kind}_full/search_k10_r0_ids.csv"
    b = R / f"experiments/breakthrough_{kind}_audit/search_k10_r0_ids.csv"
    if a.exists() and b.exists():
        checks.append(compare(a, b))
for kind in ["half", "sq8"]:
    a = R / f"experiments/screen45_cohere_{kind}_estimated_stop/search_k10_r0_ids.csv"
    b = R / f"experiments/screen45_cohere_{kind}_estimated_batch/search_k10_r0_ids.csv"
    if a.exists() and b.exists():
        checks.append(compare(a, b))
result = {
    "query_comparisons": checks,
    "total_query_comparisons": sum(x["queries_reference"] for x in checks),
    "id_score_mismatches": sum(x["id_score_mismatch_count"] for x in checks),
    "score_multiset_mismatches": sum(
        x["score_multiset_mismatch_count"] for x in checks
    ),
}
a = R / "results/breakthrough_base_paths.csv"
b = R / "results/breakthrough_batch1_paths.csv"
if a.exists() and b.exists():
    aa = list(csv.reader(a.open()))
    bb = list(csv.reader(b.open()))
    result["batch1_path_equivalence"] = {
        "rows_base": len(aa),
        "rows_batch1": len(bb),
        "full_row_match": aa == bb,
    }
x = R / "results/breakthrough_default_tests.xml"
if x.exists():
    result["default_unit_tests"] = ET.parse(x).getroot().attrib
stats = {}
for kind in ["half", "sq8"]:
    for mode in ["audit", "full"]:
        p = R / f"results/breakthrough_{kind}_{mode}_stats.csv"
        if p.exists():
            rows = [[int(x) for x in r] for r in csv.reader(p.open())]
            stats[kind + "_" + mode] = {
                "queries": len(rows),
                "mean_expanded": sum(r[3] for r in rows) / len(rows),
                "mean_refined": sum(r[4] for r in rows) / len(rows),
                "max_expanded": max(r[3] for r in rows),
                "center_bytes": rows[0][5],
                "stride": rows[0][6],
            }
result["center_audit_stats"] = stats
old = json.loads((R / "artifact90/manifest.json").read_text())
result["historical_libraries_unchanged"] = {
    k: hashlib.sha256(Path(v["library"]).read_bytes()).hexdigest() == v["sha256"]
    for k, v in old.items()
}
(R / "results/breakthrough_correctness.json").write_text(json.dumps(result, indent=2))
emit(
    json.dumps({k: v for k, v in result.items() if k != "query_comparisons"}, indent=2)
)
assert result["score_multiset_mismatches"] == 0
if "batch1_path_equivalence" in result:
    assert result["batch1_path_equivalence"]["full_row_match"]
assert all(result["historical_libraries_unchanged"].values())
