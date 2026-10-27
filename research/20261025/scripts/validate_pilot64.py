from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models64"
started = time.monotonic()
assert "PILOT64 BOUNDED COMPLETE" in (R / "logs/pilot64_bounded.log").read_text()
gh = np.fromfile(M / "bounded_graph.bin", dtype="<u4", count=8)
vh = np.fromfile(M / "bounded_vectors.f32", dtype="<u4", count=4)
n = int(gh[2])
d = int(gh[3])
assert gh.tolist()[:5] == [1363625828, 1, 1000000, 768, 32]
assert vh.tolist() == [1363629668, 1, n, d]
dtype = np.dtype([("key", "<u8"), ("degree", "<u4"), ("neighbors", "<u4", (32,))])
assert dtype.itemsize == 140
assert (M / "bounded_graph.bin").stat().st_size == 32 + n * 140
assert (M / "bounded_vectors.f32").stat().st_size == 16 + n * d * 4
graph = np.memmap(M / "bounded_graph.bin", dtype=dtype, mode="r", offset=32, shape=(n,))
v = np.memmap(M / "bounded_vectors.f32", dtype="<f4", mode="r", offset=16, shape=(n, d))
keys = graph["key"].copy()
assert len(np.unique(keys)) == n
assert keys.max() < n
assert gh[5] < n
base = np.memmap(
    "/home/xingrui/bench/symphony_qg/cohere/original/data/base.f32",
    dtype="<f4",
    mode="r",
    shape=(n, d),
)
norms = np.empty(n)
maxdiff = 0.0
edges = 0
for start in range(0, n, 1024):
    a = v[start : start + 1024]
    assert np.isfinite(a).all()
    norms[start : start + len(a)] = np.linalg.norm(a.astype(np.float64), axis=1)
    maxdiff = max(
        maxdiff, float(np.max(np.abs(a - base[keys[start : start + len(a)]])))
    )
    rows = graph[start : start + len(a)]
    degree = rows["degree"]
    assert np.all(degree <= 32)
    adj = rows["neighbors"]
    mask = np.arange(32)[None, :] < degree[:, None]
    assert np.all(adj[~mask] == np.iinfo(np.uint32).max)
    assert np.all(adj[mask] < n)
    assert not np.any((adj == np.arange(start, start + len(a))[:, None]) & mask)
    ordered = np.sort(adj, axis=1)
    assert not np.any((np.diff(ordered, axis=1) == 0) & (ordered[:, 1:] < n))
    edges += int(np.sum(degree))
assert maxdiff < 2e-06
normerror = float(np.max(np.abs(norms**2 - 1)))
assert normerror < 0.0001
oldtype = np.dtype(
    [
        ("ids", "<u4", (4,)),
        ("keys", "<u8", (4,)),
        ("degree", "<u4"),
        ("offsets", "<u4", (3,)),
        ("vectors", "<f4", (4, 768)),
    ]
)
old = np.memmap(
    R / "models61/edges.bin", dtype=oldtype, mode="r", offset=24, shape=(65536,)
)
assert np.array_equal(graph["key"][old["ids"]], old["keys"])
assert np.array_equal(graph["degree"][old["ids"][:, 0]], old["degree"])
actual = graph["neighbors"][old["ids"][:, 0]]
assert np.array_equal(
    np.take_along_axis(actual, old["offsets"], axis=1), old["ids"][:, 1:]
)
ch = np.fromfile(M / "bounded_candidates.bin", dtype="<u4", count=8)
assert ch.tolist() == [1363624804, 1, n, d, 4096, 200, 400, 20261026]
ct = np.dtype([("query", "<u4"), ("ids", "<u4", (200,)), ("distances", "<f4", (200,))])
assert ct.itemsize == 1604
assert (M / "bounded_candidates.bin").stat().st_size == 32 + 4096 * 1604
rows = np.memmap(
    M / "bounded_candidates.bin", dtype=ct, mode="r", offset=32, shape=(4096,)
)
qids = rows["query"].copy()
assert np.array_equal(qids, (np.arange(4096, dtype=np.uint64) * 2654435761 + 7193) % n)
assert len(np.unique(qids)) == 4096
assert np.all(rows["ids"] < n)
assert not np.any(rows["ids"] == qids[:, None])
assert np.all(np.diff(np.sort(rows["ids"], axis=1), axis=1) > 0)
assert np.isfinite(rows["distances"]).all()
maxscore = 0.0
pairs = 0
for start in range(0, 4096, 8):
    batch = rows[start : start + 8]
    a = v[batch["query"]].astype(np.float64) / norms[batch["query"], None]
    ids = batch["ids"]
    b = v[ids].astype(np.float64) / norms[ids, None]
    truth = 1 - np.einsum("id,ijd->ij", a, b)
    maxscore = max(maxscore, float(np.max(np.abs(truth - batch["distances"]))))
    pairs += truth.size
assert maxscore < 3e-05
line = next(
    line
    for line in (R / "logs/pilot64_bounded.log").read_text().splitlines()
    if line.startswith("GRAPH64_RESULT ")
)
metrics = {key: float(value) for key, value in re.findall("(\\w+)=([0-9.e+-]+)", line)}
manifest = {}
for p in [
    M / "bounded_graph.bin",
    M / "bounded_vectors.f32",
    M / "bounded_candidates.bin",
]:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    manifest[str(p.relative_to(R))] = h.hexdigest()
assert (
    hashlib.sha256((M / "graph.bin").read_bytes()).hexdigest()
    == manifest["models64/bounded_graph.bin"]
)
h = hashlib.sha256()
with (M / "vectors.f32").open("rb") as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
        h.update(block)
assert h.hexdigest() == manifest["models64/bounded_vectors.f32"]
summary = {
    "initial_and_bounded_exports_identical": True,
    "nodes": n,
    "dimensions": d,
    "edges": edges,
    "full_coordinate_mapping_max_abs_error": maxdiff,
    "norm2_max_error": normerror,
    "previous_edge61_centers_checked": 65536,
    "previous_edge61_edges_checked": 196608,
    "pilot_queries": 4096,
    "candidates_per_query": 200,
    "self_excluded": True,
    "fp64_cosine_pairs_checked": pairs,
    "max_api_score_error": maxscore,
    "metrics": metrics,
    "linear_full_search_seconds_estimate_not_measurement": metrics["search_seconds"]
    * n
    / 4096,
    "sha256": manifest,
    "validation_seconds": time.monotonic() - started,
    "query_or_gt_used": False,
    "no_retrieval_qps_or_recall_claim": True,
}
(M / "candidate_dtype.json").write_text(
    json.dumps({"graph_record": dtype.descr, "candidate_record": ct.descr}, indent=2)
    + "\n"
)
(R / "results/graph64_pilot_validation.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
logging.info("%s", json.dumps(summary, indent=2))
logging.info("%s", "VALIDATE PILOT64 COMPLETE")
