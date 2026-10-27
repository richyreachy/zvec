from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models70"
M.mkdir(exist_ok=False)
g = np.memmap(
    R / "models64/full_graph.bin",
    mode="r",
    dtype=np.dtype([("key", "<u8"), ("degree", "<u4"), ("ids", "<u4", (32,))]),
    offset=32,
    shape=(1000000,),
)
c = np.memmap(
    R / "models64/full_candidates.bin",
    mode="r",
    dtype=np.dtype(
        [("query", "<u4"), ("ids", "<u4", (200,)), ("distances", "<f4", (200,))]
    ),
    offset=32,
    shape=(1000000,),
)
assert np.array_equal(np.sort(c["query"]), np.arange(1000000))
inverse = np.empty(1000000, dtype=np.uint32)
inverse[c["query"]] = np.arange(1000000, dtype=np.uint32)
eligible = np.flatnonzero(g["degree"] == 32)
centers = np.sort(np.random.default_rng(7001).choice(eligible, 8192, replace=False))
ids = np.empty((8192, 37), dtype="<u4")
ids[:, 0] = centers
ids[:, 1:33] = g["ids"][centers]
positions = [16, 63, 127, 199]
fallback = 0
for i, center in enumerate(centers):
    excluded = set(ids[i, :33].tolist())
    pool = c["ids"][inverse[center]]
    for j, start in enumerate(positions):
        for offset in range(200):
            pos = (start + offset) % 200
            candidate = int(pool[pos])
            if candidate not in excluded:
                ids[i, 33 + j] = candidate
                excluded.add(candidate)
                fallback += offset > 0
                break
        else:
            raise AssertionError("insufficient distinct nonedge query proxies")
    assert len(set(ids[i].tolist())) == 37
ids.tofile(M / "ids.u32")
order = np.random.default_rng(7002).permutation(8192)
np.save(M / "train_rows.npy", order[:6144])
np.save(M / "validation_rows.npy", order[6144:])
tr = set(ids[order[:6144], 1:].ravel().tolist())
va = set(ids[order[6144:], 1:].ravel().tolist())
config = {
    "centers": 8192,
    "eligible_full_degree_centers": len(eligible),
    "neighbors": 32,
    "queries_per_center": 4,
    "query_candidate_positions": positions,
    "nonedge_distinct_query_selection_forward_wrap": True,
    "positions_replaced": fallback,
    "center_seed": 7001,
    "split_seed": 7002,
    "train_centers": 6144,
    "validation_centers": 2048,
    "shared_noncenter_ids_between_folds": len(tr & va),
    "primary_or_extra_query_files_read": False,
    "gt_files_read": False,
    "internal_database_proxies_not_unseen_vectors": True,
    "seeds": [70, 71, 72],
    "layers": 3,
    "pair_strides": [1, 16, 384],
    "steps": 600,
    "batch_centers": 32,
    "temperature": 0.01,
    "tie_tolerance": 1e-07,
    "initial_lr": 0.003,
    "final_lr_fraction": 0.1,
    "backtrack_factors": [1.0, 0.5, 0.25, 0.125, 0.0625],
    "update_acceptance": "actual same-batch crossentropy must not increase;otherwise keep angles,record rejected update",
    "final_step_evaluated": 600,
    "validation_not_used_for_checkpoint_selection": True,
    "gate": {
        "each_seed_actual_top1_relative_error_gain": 0.05,
        "paired_center_bootstrap_gain_lower95_positive": True,
        "all_seeds_required": True,
    },
    "no_qps_claim": True,
}
(M / "config.json").write_text(json.dumps(config, indent=2) + "\n")
logging.info("%s", json.dumps(config, indent=2))
logging.info("%s", "PREPARE70 COMPLETE")
