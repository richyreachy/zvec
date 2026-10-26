from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models67"
M.mkdir(exist_ok=False)
n = 1000000
g = np.memmap(
    R / "models64/full_graph.bin",
    dtype=np.dtype([("key", "<u8"), ("degree", "<u4"), ("ids", "<u4", (32,))]),
    mode="r",
    offset=32,
    shape=(n,),
)
rng = np.random.default_rng(202610260067)
pool = rng.choice(n, 16384, replace=False)
lookup = np.full(n, -1, dtype=np.int32)
lookup[pool] = np.arange(len(pool))
sources = []
destinations = []
for lo in range(0, n, 8192):
    a = g[lo : lo + 8192]
    valid = np.arange(32)[None, :] < a["degree"][:, None]
    dst = a["ids"][valid]
    src = np.broadcast_to(
        np.arange(lo, lo + len(a), dtype=np.uint32)[:, None], (len(a), 32)
    )[valid]
    keep = lookup[dst] >= 0
    sources.append(src[keep])
    destinations.append(dst[keep])
sources = np.concatenate(sources)
destinations = np.concatenate(destinations)
order = np.lexsort((sources, destinations))
sources = sources[order]
destinations = destinations[order]
unique, starts, counts = np.unique(destinations, return_index=True, return_counts=True)
positions = {
    int(x): (int(start), int(count))
    for x, start, count in zip(unique, starts, counts, strict=True)
}
targets = [int(x) for x in pool if int(x) in positions and positions[int(x)][1] >= 2][
    :8192
]
assert len(targets) == 8192
c = np.memmap(
    R / "models64/full_candidates.bin",
    dtype=np.dtype([("q", "<u4"), ("ids", "<u4", (200,)), ("dist", "<f4", (200,))]),
    mode="r",
    offset=32,
    shape=(n,),
)
inverse = np.empty(n, dtype=np.int64)
inverse[c["q"]] = np.arange(n)
rows = []
for target in targets:
    start, count = positions[target]
    centers = [int(sources[start]), int(sources[start + count // 2])]
    ids = [int(x) for x in c[inverse[target]]["ids"] if x not in centers]
    queries = [ids[i] for i in [0, 16, 63, len(ids) - 1]]
    row = [target, *centers, *queries]
    assert len(set(row)) == 7
    for center in centers:
        assert target in g[center]["ids"][: g[center]["degree"]]
    rows.append(row)
rows = np.asarray(rows, dtype="<u4")
rows.tofile(M / "ids.u32")
split = np.random.default_rng(6701).permutation(len(rows))
np.savez(M / "split.npz", train=split[:6144], valid=split[6144:])
config = {
    "seed": 202610260067,
    "sampled_targets": 8192,
    "centers_per_target": 2,
    "center_selection": "first and middle source ID among original directed incoming neighbors",
    "proxy_queries_per_target": 4,
    "query_positions_after_center_exclusion": [0, 16, 63, "last"],
    "train_targets": 6144,
    "validation_targets": 2048,
    "split_seed": 6701,
    "query_or_gt_inputs": [],
    "database_proxy_queries_only": True,
    "actual_online_cooccurrence_not_measured": True,
    "ideal_estimator_without_query_LUT_rounding": True,
    "isotropic_formula_is_model_not_cohere_assumption": True,
    "endpoint_overlap_between_target_groups_possible": True,
    "ids_sha256": hashlib.sha256((M / "ids.u32").read_bytes()).hexdigest(),
    "input_hashes": json.loads(
        (R / "results/graph64_full_validation.json").read_text()
    )["sha256"],
}
(M / "config.json").write_text(json.dumps(config, indent=2) + "\n")
logging.info("%s", json.dumps(config, indent=2))
