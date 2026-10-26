from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
M = R / "models74"
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
old = np.fromfile(R / "models70/ids.u32", dtype="<u4").reshape(8192, 37)
old71 = np.fromfile(R / "models71/ids.u32", dtype="<u4").reshape(2048, 65)
old72 = np.fromfile(R / "models72/ids.u32", dtype="<u4").reshape(2048, 65)
excluded_centers = np.concatenate([old[:, 0], old71[:, 0], old72[:, 0]])
eligible = np.setdiff1d(np.flatnonzero(g["degree"] == 32), excluded_centers)
centers = np.sort(np.random.default_rng(7401).choice(eligible, 2048, replace=False))
ids = np.empty((2048, 65), dtype="<u4")
ids[:, 0] = centers
ids[:, 1:33] = g["ids"][centers]
positions = np.empty((2048, 32), dtype="<u4")
rng = np.random.default_rng(7402)
minimum_pool = 200
for i, center in enumerate(centers):
    excluded = set(ids[i, :33].tolist())
    pool = c["ids"][inverse[center]]
    for layer in range(4):
        legal = [
            p
            for p in range(layer * 50, (layer + 1) * 50)
            if int(pool[p]) not in excluded
        ]
        assert len(set(pool[legal].tolist())) == len(legal)
        minimum_pool = min(minimum_pool, len(legal))
        assert len(legal) >= 8
        chosen = rng.choice(legal, 8, replace=False)
        positions[i, layer * 8 : layer * 8 + 8] = chosen
        ids[i, 33 + layer * 8 : 33 + layer * 8 + 8] = pool[chosen]
        excluded.update(pool[chosen].tolist())
    assert len(set(ids[i].tolist())) == 65
sel = np.array([8 * b + j for b in range(4) for j in range(4)], dtype="<u4")
val = sel + 4
assert not set(sel) & set(val)
assert len(set(sel) | set(val)) == 32
assert not set(centers) & set(excluded_centers)
selection_ids = ids[:, 33:][:, sel]
validation_ids = ids[:, 33:][:, val]
a = set(selection_ids.ravel().tolist())
b = set(validation_ids.ravel().tolist())
old_ids = set(old.ravel().tolist())
new_ids = set(ids.ravel().tolist())
ids.tofile(M / "ids.u32")
positions.tofile(M / "positions.u32")
sel.tofile(M / "selection_positions.u32")
val.tofile(M / "validation_positions.u32")
np.arange(2048, dtype="<u4").tofile(M / "rows.u32")
config = {
    "centers": 2048,
    "neighbors": 32,
    "queries_per_center": 32,
    "build_queries_per_center": 16,
    "validation_queries_per_center": 16,
    "center_seed": 7401,
    "proxy_seed": 7402,
    "score_modes": ["native", "precision_control", "learned"],
    "truth_tie_tolerance": 1e-07,
    "candidate_position_bins": [[0, 49], [50, 99], [100, 149], [150, 199]],
    "queries_per_bin": 8,
    "build_first_four_per_bin": True,
    "minimum_eligible_queries_in_bin": minimum_pool,
    "old_centers_excluded": 12288,
    "new_centers_disjoint_from_phase70_71_72": True,
    "build_validation_query_ids_disjoint_within_center": True,
    "unique_build_query_ids": len(a),
    "unique_validation_query_ids": len(b),
    "shared_query_ids_across_center_folds": len(a & b),
    "shared_any_vector_ids_with_phase70": len(old_ids & new_ids),
    "shared_any_vector_ids_with_phase71": len(new_ids & set(old71.ravel().tolist())),
    "shared_any_vector_ids_with_phase72": len(new_ids & set(old72.ravel().tolist())),
    "primary_or_extra_query_files_read": False,
    "gt_files_read": False,
    "local_queries_not_iid": True,
    "proxy_pool_is_approximate_not_exact_top200": True,
    "original_symphony_rotation_only": True,
    "optimizer": {
        "objective": "sum_i(e_i-mean_e)^T Q^T Q(e_i-mean_e)+lambda sum_i norm(e_i)^2",
        "Q": "16 build residual queries divided by sqrt16",
        "lambda": "trace(Q^T Q)/768;zero keeps native signs",
        "sweeps": 2,
        "max_accepted_flips_per_edge_per_sweep": 4,
        "max_alpha_ratio": 1.02,
        "minimum_delta": "-1e-13*max(1,current_objective)",
        "lane_order": "0..31",
        "coordinate_ties": "lowest coordinate",
        "actual_objective_must_strictly_decrease": True,
    },
    "gate": {
        "relative_heldout_actual_top1_error_reduction_against_native": 0.1,
        "paired_center_bootstrap95_lower_positive": True,
        "bootstrap_seed": 7474,
        "bootstrap_samples": 4000,
    },
    "no_retrieval_qps_or_recall_claim": True,
}
(M / "config.json").write_text(json.dumps(config, indent=2) + "\n")
logging.info("%s", json.dumps(config, indent=2))
logging.info("%s", "PREPARE74 COMPLETE")
