from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models75"
ids = np.fromfile(M / "ids.u32", dtype="<u4").reshape(2048, 65)
pairs = np.fromfile(M / "pairs.u32", dtype="<u4").reshape(1024, 2)
queries = np.fromfile(M / "cross_ids.u32", dtype="<u4").reshape(1024, 16)
assert np.array_equal(np.sort(pairs.ravel()), np.arange(2048))
v = np.memmap(
    R / "models64/full_vectors.f32",
    mode="r",
    dtype="<f4",
    offset=16,
    shape=(1000000, 768),
)
x = np.memmap(M / "rotated.f32", mode="r", dtype="<f4", shape=(2048, 65, 768))
qr = np.memmap(M / "cross_rotated.f32", mode="r", dtype="<f4", shape=(1024, 16, 768))
truth = np.empty((1024, 2, 16, 32))
max_score = 0.0
max_norm = 0.0
min_union = 64
duplicates = 0
for p, rows in enumerate(pairs):
    assert len(set(queries[p])) == 16
    assert not set(queries[p]) & set(ids[rows].ravel())
    q = np.asarray(v[queries[p]], dtype=np.float64)
    rot = np.asarray(qr[p], dtype=np.float64)
    max_norm = max(
        max_norm,
        float(np.max(np.abs(np.sum(q * q, axis=1) - np.sum(rot * rot, axis=1)))),
    )
    unique = len(set(ids[rows, 1:33].ravel()))
    min_union = min(min_union, unique)
    duplicates += 64 - unique
    for b, row in enumerate(rows):
        n = np.asarray(v[ids[row, 1:33]], dtype=np.float64)
        truth[p, b] = 1 - q @ n.T
        rot_truth = 1 - rot @ np.asarray(x[row, 1:33], dtype=np.float64).T
        max_score = max(max_score, float(np.max(np.abs(truth[p, b] - rot_truth))))
assert max_norm < 2e-06
assert max_score < 2e-06
truth.astype("<f8").tofile(M / "cross_truth.f64")
a = {
    "pairs": 1024,
    "cross_queries": 16384,
    "all_pairs_disjoint": True,
    "all_cross_queries_excluded_from_corresponding_centers_neighbors_build_and_local_validation": True,
    "raw_fp64_scores": 1048576,
    "max_raw_rotated_score_difference": max_score,
    "max_query_norm_difference": max_norm,
    "minimum_candidate_union_size": min_union,
    "total_duplicate_candidates_across_pairs": duplicates,
    "sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            M / "pairs.u32",
            M / "cross_ids.u32",
            M / "cross_rotated.f32",
            M / "cross_truth.f64",
        ]
    },
    "not_full_search_equivalence": True,
}
(R / "results/coupled75_cross_data.json").write_text(json.dumps(a, indent=2) + "\n")
logging.info("%s", json.dumps(a, indent=2))
logging.info("%s", "CROSSDATA75 COMPLETE")
