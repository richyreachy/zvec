"""Freeze mechanism-A states and held-out proxy IDs before any scoring."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")

R = Path(__file__).resolve().parent.parent
name = sys.argv[1] if len(sys.argv) > 1 else "path80_matched_v1"
if not name.startswith("path80_matched_") or Path(name).name != name:
    raise RuntimeError("Expected new direct path80_matched_* output name")
out = R / name
out.mkdir(exist_ok=False)
started = time.time()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


expected = {
    "models64/full_graph.bin": "c04308b5445a666026bd460100ef0a649fd3dfd7e75a1b57561ab40a1e7dc53b",
    "models64/full_candidates.bin": "6c55553e87dae59c6c67babe4a5c65b1f92fedef4c83b0702a00f2ac4f61c830",
    "models77/proxies.u32": "460f6f8d88f0360b47e34dc9e2061fbe64f4af9e5763f096ae258bf3883a12a2",
    "models68/primary/events.bin": "4787aa0f822bb2ddf35f4af28c04579283fac5d557fbea74c9f1802047d3d04c",
}
for relative, value in expected.items():
    if sha(R / relative) != value:
        raise RuntimeError(f"Frozen input differs: {relative}")
n = 1000000
gd = np.dtype([("key", "<u8"), ("degree", "<u4"), ("ids", "<u4", (32,))])
cd = np.dtype([("query", "<u4"), ("ids", "<u4", (200,)), ("distance", "<f4", (200,))])
ed = np.dtype(
    [
        ("query", "<u4"),
        ("kind", "<u4"),
        ("step", "<u4"),
        ("source", "<u4"),
        ("target", "<u4"),
        ("estimate", "<f4"),
        ("exact", "<f4"),
        ("reserved", "<u4"),
    ]
)
graph = np.memmap(
    R / "models64/full_graph.bin", dtype=gd, mode="r", offset=32, shape=(n,)
)
candidates = np.memmap(
    R / "models64/full_candidates.bin", dtype=cd, mode="r", offset=32, shape=(n,)
)
training = np.memmap(R / "models77/proxies.u32", dtype="<u4", mode="r", shape=(n, 16))
events = np.memmap(R / "models68/primary/events.bin", dtype=ed, mode="r", offset=16)
inverse = np.full(n, np.iinfo(np.uint32).max, dtype=np.uint32)
assert np.array_equal(np.sort(candidates["query"]), np.arange(n))
inverse[candidates["query"]] = np.arange(n, dtype=np.uint32)
steps = [10, 14, 18, 22]
selected = {}
for event in np.flatnonzero((events["kind"] == 1) & np.isin(events["step"], steps)):
    r = events[event]
    pair = (int(r["query"]), int(r["step"]))
    assert pair not in selected
    assert 0 <= pair[0] < 1000
    selected[pair] = (int(r["source"]), int(event))
states, missing = [], []
for query in range(1000):
    for step in steps:
        pair = (query, step)
        if pair not in selected:
            missing.append({"query": query, "step": step, "reason": "step_not_reached"})
            continue
        center, event = selected[pair]
        if graph[center]["degree"] != 32:
            missing.append(
                {
                    "query": query,
                    "step": step,
                    "center": center,
                    "reason": "degree_not_32",
                }
            )
            continue
        states.append([query, step, center, event])
centers = sorted({state[2] for state in states})
center_index = {center: i for i, center in enumerate(centers)}
rng = np.random.default_rng(8001)
rows, positions, layer_counts, unavailable = [], [], [], []
sentinel = int(np.iinfo(np.uint32).max)
for center in centers:
    excluded = {center, *map(int, graph[center]["ids"]), *map(int, training[center])}
    assert len(set(map(int, training[center]))) == 16
    assert not (
        {center, *map(int, graph[center]["ids"])} & set(map(int, training[center]))
    )
    pool = candidates[inverse[center]]["ids"]
    assert len(set(map(int, pool))) == 200
    assert np.all(pool < n)
    row, loc, counts = [center], [], []
    for layer in range(4):
        legal = [
            p
            for p in range(layer * 50, (layer + 1) * 50)
            if int(pool[p]) not in excluded
        ]
        counts.append(len(legal))
        if len(legal) < 4:
            unavailable.append(
                {"center": center, "layer": layer, "eligible_positions": len(legal)}
            )
            chosen = [sentinel] * 4
        else:
            chosen = list(map(int, rng.choice(legal, 4, replace=False)))
        loc.extend(chosen)
        row.extend(int(pool[p]) if p != sentinel else sentinel for p in chosen)
    if sentinel not in row:
        assert len(set(row)) == 17
        assert not (set(row[1:]) & excluded)
    rows.append(row)
    positions.append(loc)
    layer_counts.append(counts)
states = [
    [q, step, center, center_index[center], event] for q, step, center, event in states
]
eligible_centers = {row[0] for row in rows if sentinel not in row}
eligible_states = [row for row in states if row[2] in eligible_centers]
eligible_queries = sorted({row[0] for row in eligible_states})
arrays = {
    "states.u32": np.asarray(states, dtype="<u4").reshape(-1, 5),
    "centers_proxies.u32": np.asarray(rows, dtype="<u4").reshape(-1, 17),
    "proxy_positions.u32": np.asarray(positions, dtype="<u4").reshape(-1, 16),
    "layer_eligible_counts.u32": np.asarray(layer_counts, dtype="<u4").reshape(-1, 4),
}
for filename, array in arrays.items():
    with (out / filename).open("xb") as stream:
        array.tofile(stream)
write_json(out / "missing_states.json", missing)
write_json(out / "unavailable_proxy_layers.json", unavailable)
manifest = {
    "frozen_before_any_scoring": True,
    "created_unix": time.time(),
    "sampling_script_sha256": sha(Path(__file__)),
    "protocol_note_sha256": sha(R / "notes/path80_diagnostic_review.md"),
    "query_ids": [0, 999],
    "query_ids_interval_inclusive": True,
    "steps": steps,
    "primary_queries_already_tuned_not_blind": True,
    "no_training_or_parameter_fit": True,
    "rng": "numpy.default_rng(8001),PCG64",
    "numpy_version": np.__version__,
    "center_order": "ascending internal ID, repeated centers share one held-out proxy row",
    "sampling": "four consecutive 50-position candidate strata; each draws 4 eligible positions without replacement; preserve draw order",
    "exclusions": ["center", "32 final graph neighbors", "16 phase77 training proxies"],
    "insufficient_layer_policy": "record missing; no replacement center or seed; draw other feasible layers in fixed order; any missing layer excludes that center from matched scoring",
    "missing_query_policy": "1000 per-query rows retained; no-state queries marked ineligible and excluded from equal-query averages/bootstrap; no substitution",
    "input_sha256": expected,
    "table_sha256": {name: sha(out / name) for name in arrays},
    "state_layout_u32": [
        "query",
        "step",
        "center_id",
        "center_row",
        "center_event_index",
    ],
    "center_layout_u32": ["center_id", "16 heldout_proxy_internal_ids"],
    "missing_state_count": len(missing),
    "selected_state_count": len(states),
    "unique_center_count": len(centers),
    "unavailable_proxy_layers": len(unavailable),
    "eligible_state_count": len(eligible_states),
    "eligible_center_count": len(eligible_centers),
    "eligible_query_count": len(eligible_queries),
    "ineligible_query_ids": sorted(set(range(1000)) - set(eligible_queries)),
    "minimum_eligible_positions_per_layer": min(min(x) for x in layer_counts),
    "score_scale": {
        "path_observations": len(eligible_states),
        "proxy_observations_reused": len(eligible_centers) * 16,
        "total_32_candidate_observations": len(eligible_states)
        + len(eligible_centers) * 16,
        "candidate_lanes_per_score_bank": 32
        * (len(eligible_states) + len(eligible_centers) * 16),
    },
    "analysis": {
        "top1_exact_tie_tolerance": 1e-7,
        "selection_ties": "first minimum lane",
        "state_proxy_average": "mean of 16 heldout proxy indicators",
        "query_average": "mean over eligible fixed states, then equal weight over eligible queries",
        "bootstrap_replicates": 4000,
        "bootstrap_seed": 8002,
        "bootstrap_unit": "eligible query cluster",
        "two_sided_ci_percent": 97.5,
        "ci_quantiles": [0.0125, 0.9875],
        "bootstrap_numpy_quantile_method": "linear",
        "critical_correction_definition": "proxy native Top1 wrong to learned Top1 correct; count matched-state proxy observations and distinct source queries",
        "sufficiency": {
            "eligible_queries_min": 64,
            "proxy_corrections_min": 100,
            "proxy_correction_source_queries_min": 32,
        },
        "path_degradation_counts": "report separately; no extra sufficiency threshold",
        "support": {
            "delta_proxy_max": -0.02,
            "delta_path_min": 0.005,
            "interaction_ci_lower_strict_min": 0,
        },
        "refute": {"delta_path_max": -0.02, "interaction_ci_upper_max": 0.01},
        "controls": [
            "native/native score bits and choices identical",
            "exact-reference Top1 error zero with same tie rule",
            "complete 16-training-proxy objective nonincrease",
        ],
    },
    "elapsed_sampling_seconds": time.time() - started,
}
write_json(out / "sampling_manifest.json", manifest)
logging.info(
    json.dumps(
        {
            "frozen_directory": str(out),
            "manifest_sha256": sha(out / "sampling_manifest.json"),
            **{
                k: manifest[k]
                for k in [
                    "selected_state_count",
                    "missing_state_count",
                    "unique_center_count",
                    "eligible_query_count",
                    "unavailable_proxy_layers",
                    "minimum_eligible_positions_per_layer",
                    "score_scale",
                ]
            },
        },
        indent=2,
    )
)
