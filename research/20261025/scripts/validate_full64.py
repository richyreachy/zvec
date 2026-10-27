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
assert "FULL64 COMPLETE" in (R / "logs/full64.log").read_text()
pilot = json.loads((R / "results/graph64_pilot_validation.json").read_text())
n = 1000000
d = 768
h = np.fromfile(M / "full_candidates.bin", dtype="<u4", count=8)
assert h.tolist() == [1363624804, 1, n, d, n, 200, 400, 20261026]
dtype = np.dtype(
    [("query", "<u4"), ("ids", "<u4", (200,)), ("distances", "<f4", (200,))]
)
assert (M / "full_candidates.bin").stat().st_size == 32 + n * dtype.itemsize
rows = np.memmap(
    M / "full_candidates.bin", dtype=dtype, mode="r", offset=32, shape=(n,)
)
small = np.memmap(
    M / "bounded_candidates.bin", dtype=dtype, mode="r", offset=32, shape=(4096,)
)
assert rows[:4096].tobytes() == small.tobytes()
for start in range(0, n, 1024):
    batch = rows[start : start + 1024]
    expected = (
        np.arange(start, start + len(batch), dtype=np.uint64) * 2654435761 + 7193
    ) % n
    assert np.array_equal(batch["query"], expected)
    ids = batch["ids"]
    assert np.all(ids < n)
    assert not np.any(ids == batch["query"][:, None])
    assert np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0)
    assert np.isfinite(batch["distances"]).all()
v = np.memmap(M / "full_vectors.f32", dtype="<f4", mode="r", offset=16, shape=(n, d))
selected = (np.arange(1024, dtype=np.int64) * 977 + 31) % n
maxerr = 0.0
for begin in range(0, 1024, 8):
    batch = rows[selected[begin : begin + 8]]
    q = v[batch["query"]].astype(np.float64)
    q /= np.linalg.norm(q, axis=1)[:, None]
    x = v[batch["ids"]].astype(np.float64)
    x /= np.linalg.norm(x, axis=2)[:, :, None]
    truth = 1 - np.einsum("id,ijd->ij", q, x)
    maxerr = max(maxerr, float(np.max(np.abs(truth - batch["distances"]))))
assert maxerr < 3e-05
manifest = {}
for name in ["full_graph.bin", "full_vectors.f32", "full_candidates.bin"]:
    h = hashlib.sha256()
    with (M / name).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    manifest["models64/" + name] = h.hexdigest()
assert (
    manifest["models64/full_graph.bin"] == pilot["sha256"]["models64/bounded_graph.bin"]
)
assert (
    manifest["models64/full_vectors.f32"]
    == pilot["sha256"]["models64/bounded_vectors.f32"]
)
line = next(
    x
    for x in (R / "logs/full64.log").read_text().splitlines()
    if x.startswith("GRAPH64_RESULT ")
)
metrics = {k: float(v) for k, v in re.findall("(\\w+)=([0-9.e+-]+)", line)}
summary = {
    "all_queries": n,
    "candidate_pairs": n * 200,
    "unique_nonself_candidates_per_query": 200,
    "pilot_prefix_bit_identical": 4096,
    "additional_fp64_score_pairs": 204800,
    "additional_fp64_score_max_error": maxerr,
    "full_graph_and_vectors_match_pilot": True,
    "metrics": metrics,
    "sha256": manifest,
    "validation_seconds": time.monotonic() - started,
    "query_or_gt_used": False,
    "no_retrieval_qps_or_recall_claim": True,
    "candidate_quality_not_ground_truth_certified": True,
}
(R / "results/graph64_full_validation.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
logging.info("%s", json.dumps(summary, indent=2))
logging.info("%s", "VALIDATE FULL64 COMPLETE")
