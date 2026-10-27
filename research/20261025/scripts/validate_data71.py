from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models71"
assert "ROTATE71 COMPLETE" in (R / "logs/rotate71.log").read_text()
ids = np.fromfile(M / "ids.u32", dtype="<u4").reshape(2048, 65)
x = np.memmap(M / "rotated.f32", mode="r", dtype="<f4", shape=(2048, 65, 768))
v = np.memmap(
    R / "models64/full_vectors.f32",
    mode="r",
    dtype="<f4",
    offset=16,
    shape=(1000000, 768),
)
g = np.memmap(
    R / "models64/full_graph.bin",
    mode="r",
    dtype=np.dtype([("key", "<u8"), ("degree", "<u4"), ("ids", "<u4", (32,))]),
    offset=32,
    shape=(1000000,),
)
assert np.array_equal(ids[:, 1:33], g["ids"][ids[:, 0]])
assert np.all(g["degree"][ids[:, 0]] == 32)
truth = np.empty((2048, 32, 32))
offsets = np.empty((2048, 32))
max_rank_error = 0.0
max_center_dot_error = 0.0
max_norm_error = 0.0
zero = 0
ambiguous = 0
for start in range(0, 2048, 64):
    a = np.asarray(x[start : start + 64], dtype=np.float64)
    raw = np.asarray(v[ids[start : start + 64]], dtype=np.float64)
    r = a[:, 1:33] - a[:, 0, None]
    q = a[:, 33:] - a[:, 0, None]
    nr = np.sum(r * r, axis=2)
    offset = 0.5 * (nr - np.sum(a[:, 1:33] ** 2, axis=2))
    exact = 1 - np.einsum("bqd,bkd->bqk", raw[:, 33:], raw[:, 1:33])
    rank = offset[:, None, :] - np.einsum("bqd,bkd->bqk", q, r)
    truth[start : start + len(a)] = exact
    offsets[start : start + len(a)] = offset
    max_rank_error = max(
        max_rank_error,
        float(np.max(np.abs(rank - rank[:, :, :1] - (exact - exact[:, :, :1])))),
    )
    max_center_dot_error = max(
        max_center_dot_error,
        float(
            np.max(
                np.abs(
                    np.einsum("bd,bkd->bk", a[:, 0], a[:, 1:])
                    - np.einsum("bd,bkd->bk", raw[:, 0], raw[:, 1:])
                )
            )
        ),
    )
    max_norm_error = max(
        max_norm_error,
        float(np.max(np.abs(np.sum(a * a, axis=2) - np.sum(raw * raw, axis=2)))),
    )
    zero += int(np.sum(nr == 0))
    ambiguous += int(
        np.sum(np.sum(exact <= exact.min(axis=2, keepdims=True) + 1e-07, axis=2) > 1)
    )
assert max_rank_error < 2e-06
assert max_center_dot_error < 2e-06
assert max_norm_error < 2e-06
truth.astype("<f8").tofile(M / "truth.f64")
offsets.astype("<f8").tofile(M / "offsets.f64")
a = {
    "actual_graph_centers": 2048,
    "all_edges_match_original_adjacency": True,
    "cosine_pair_scores_checked": 2048 * 32 * 32,
    "center_inner_products_checked": 2048 * 64,
    "norms_checked": 2048 * 65,
    "max_pair_gap_rotation_error": max_rank_error,
    "max_center_inner_product_rotation_error": max_center_dot_error,
    "max_norm2_rotation_error": max_norm_error,
    "zero_residual_edges_retained": zero,
    "proxy_queries_with_tied_minima_within_1e_7": ambiguous,
    "selection_validation_query_ids_disjoint_per_center": all(
        len(set(row[33:])) == 32 for row in ids
    ),
    "source_hashes": {
        str(p.relative_to(R)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            M / "config.json",
            M / "ids.u32",
            M / "rotated.f32",
            M / "truth.f64",
            M / "offsets.f64",
            R / "scripts/rotate71.cc",
            R / "scripts/prepare71.py",
        ]
    },
}
(R / "results/local71_data_validation.json").write_text(json.dumps(a, indent=2) + "\n")
logging.info("%s", json.dumps(a, indent=2))
logging.info("%s", "DATA71 COMPLETE")
