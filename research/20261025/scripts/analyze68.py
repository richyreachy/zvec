from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
primary = len(sys.argv) > 1 and sys.argv[1] == "primary"
M = R / ("models68/primary" if primary else "models68")
assert ("PRIMARY68 COMPLETE" if primary else "RUN68 COMPLETE") in (
    R / "logs" / ("primary68.log" if primary else "run68.log")
).read_text()
h = np.fromfile(M / "events.bin", dtype="<u4", count=4)
assert np.array_equal(h, [1363627368, 1, 32, 768])
dtype = np.dtype(
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
events = np.memmap(M / "events.bin", dtype=dtype, mode="r", offset=16)
assert (M / "events.bin").stat().st_size == 16 + len(events) * 32
assert (events["reserved"] == 0).all()
assert np.isfinite(events["estimate"]).all()
assert np.isfinite(events["exact"]).all()
result_dtype = np.dtype(
    [("query", "<u4"), ("ids", "<u4", (10,)), ("dist", "<f4", (10,))]
)
if not primary:
    results = np.fromfile(M / "reference_results.bin", dtype=result_dtype, offset=32)
graph = np.memmap(
    R / "models64/full_graph.bin",
    dtype=np.dtype([("key", "<u8"), ("degree", "<u4"), ("ids", "<u4", (32,))]),
    mode="r",
    offset=32,
    shape=(1000000,),
)
if primary:
    inverse = np.empty(1000000, dtype=np.uint32)
    inverse[graph["key"]] = np.arange(1000000, dtype=np.uint32)
    results = np.zeros(1000, dtype=result_dtype)
    results["query"] = np.iinfo(np.uint32).max
    counters = np.zeros(1000, dtype=int)
    for label, query_text, key, score in csv.reader(
        (R / "experiments/diag68_primary/search_k10_r0_ids.csv").open()
    ):
        assert label == "20000"
        qi = int(query_text)
        j = counters[qi]
        results[qi]["ids"][j] = inverse[int(key)]
        results[qi]["dist"][j] = float.fromhex(score)
        counters[qi] += 1
    assert (counters == 10).all()
    np.save(M / "results.npy", results)
    query_rows = []
    for line in (
        Path("/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1m.1000.new.txt")
        .read_text()
        .splitlines()
    ):
        qi, values, trailer = line.split(";")
        assert trailer == ""
        assert int(qi) == len(query_rows)
        v = np.array([float(x) for x in values.split()], dtype=np.float32).astype(
            np.float64
        )
        assert v.size == 768
        v /= np.linalg.norm(v)
        query_rows.append(v.astype(np.float32))
    np.asarray(query_rows, dtype="<f4").tofile(M / "queries.f32")
ends = np.flatnonzero(events["kind"] == 3)
assert len(ends) == len(results) == (1000 if primary else 4096)
pairs = []
queries = []
start = 0
max_truth_delta = 0.0
blocks = 0
scored = 0
for qi, end in enumerate(ends):
    rows = events[start:end]
    start = end + 1
    assert (rows["query"] == qi).all()
    assert events[end]["query"] == qi
    qid = int(results[qi]["query"])
    assert primary or qid == (qi * 2654435761 + 6819) % 1000000
    expanded = {}
    pending = {}
    pairindex = {}
    loc = 0
    step = 0
    after_expansion_scores = 0
    expanded_nonself = expanded_multi = 0
    while loc < len(rows):
        r = rows[loc]
        assert r["kind"] == 1
        step += 1
        source = int(r["source"])
        assert r["target"] == source
        assert r["step"] == step
        assert source not in expanded
        expanded[source] = float(r["exact"])
        threshold = (
            sorted(expanded.values())[9] if len(expanded) >= 10 else float("inf")
        )
        seen = pending.get(source, [])
        if source != qid and step > 1:
            expanded_nonself += 1
            expanded_multi += len(seen) >= 2
        degree = int(graph[source]["degree"])
        batch = rows[loc + 1 : loc + 1 + degree]
        assert len(batch) == degree
        assert (batch["kind"] == 2).all()
        assert (batch["step"] == step).all()
        assert (batch["source"] == source).all()
        assert np.array_equal(batch["target"], graph[source]["ids"][:degree])
        blocks += 1
        scored += degree
        for rec in batch:
            target = int(rec["target"])
            if target in expanded:
                after_expansion_scores += 1
                max_truth_delta = max(
                    max_truth_delta, abs(float(rec["exact"]) - expanded[target])
                )
                continue
            item = (source, step, float(rec["estimate"]), float(rec["exact"]))
            history = pending.setdefault(target, [])
            if history:
                assert all(old[0] != source for old in history)
                max_truth_delta = max(max_truth_delta, abs(item[3] - history[0][3]))
            history.append(item)
            if len(history) == 2 and target != qid:
                first, second = history
                pairindex[target] = len(pairs)
                pairs.append(
                    [
                        qi,
                        qid,
                        target,
                        first[0],
                        second[0],
                        first[1],
                        second[1],
                        first[2],
                        second[2],
                        second[3],
                        0,
                        0,
                        0,
                        threshold,
                    ]
                )
        loc += degree + 1
    assert step == events[end]["step"]
    assert 10 <= step <= 64
    returned = {int(x) for x in results[qi]["ids"]}
    assert len(returned) == 10
    assert returned.issubset(expanded)
    for target, score in zip(results[qi]["ids"], results[qi]["dist"], strict=True):
        max_truth_delta = max(
            max_truth_delta, abs(float(score) - expanded[int(target)])
        )
    for target, index in pairindex.items():
        pairs[index][10] = int(target in expanded)
        pairs[index][11] = int(target in returned)
        pairs[index][12] = len(pending[target])
    returned_nonself = [x for x in returned if x != qid]
    returned_multi = sum(len(pending.get(x, [])) >= 2 for x in returned_nonself)
    candidates = [x for x in pending if x != qid]
    queries.append(
        [
            qi,
            qid,
            step,
            len(candidates),
            len(pairindex),
            expanded_nonself,
            expanded_multi,
            len(returned_nonself),
            returned_multi,
            after_expansion_scores,
        ]
    )
assert start == len(events)
assert max_truth_delta < 1e-06
pdtype = np.dtype(
    [
        (name, "<u4")
        for name in [
            "query",
            "query_id",
            "target",
            "source1",
            "source2",
            "step1",
            "step2",
        ]
    ]
    + [(name, "<f8") for name in ["estimate1", "estimate2", "exact"]]
    + [(name, "<u4") for name in ["expanded", "returned", "estimate_count"]]
    + [("threshold_at_second", "<f8")]
)
pairs_array = np.array([tuple(x) for x in pairs], dtype=pdtype)
np.save(M / "pairs_v2.npy", pairs_array)
queries = np.asarray(queries, dtype=np.uint32)
np.save(M / "query_stats_v2.npy", queries)
report = {
    "queries": len(results),
    "events": len(events),
    "verified_expansion_blocks": blocks,
    "verified_scored_edges": scored,
    "max_reference_distance_or_result_delta": max_truth_delta,
    "nonself_pending_candidate_occurrences": int(queries[:, 3].sum()),
    "nonself_candidates_with_second_preexpansion_estimate": len(pairs),
    "fraction_pending_with_second_estimate": float(
        queries[:, 4].sum() / queries[:, 3].sum()
    ),
    "nonself_nonentry_expanded_nodes": int(queries[:, 5].sum()),
    "nonself_nonentry_expanded_with_second_estimate": int(queries[:, 6].sum()),
    "fraction_expanded_with_second_estimate": float(
        queries[:, 6].sum() / queries[:, 5].sum()
    ),
    "nonself_returned_nodes": int(queries[:, 7].sum()),
    "nonself_returned_with_second_estimate": int(queries[:, 8].sum()),
    "fraction_returned_with_second_estimate": float(
        queries[:, 8].sum() / queries[:, 7].sum()
    ),
    "mean_expanded": float(queries[:, 2].mean()),
    "second_arrival_step_gap_percentiles": np.percentile(
        pairs_array["step2"] - pairs_array["step1"], [0, 25, 50, 75, 90, 100]
    ).tolist(),
    "queries_include_self_and_are_not_external_distribution": not primary,
    "primary_queries_already_tuned_not_blind": primary,
    "first_two_arrivals_only_for_pair_geometry": True,
    "no_qps_or_recall_claim": True,
}
(
    R
    / "results"
    / (
        "trace68_primary_availability.json"
        if primary
        else "trace68_availability_v2.json"
    )
).write_text(json.dumps(report, indent=2) + "\n")
logging.info("%s", json.dumps(report, indent=2))
