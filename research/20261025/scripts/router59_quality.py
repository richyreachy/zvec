from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
R = Path(__file__).resolve().parent.parent
records = np.memmap(
    R / "models59/anchors.bin",
    mode="r",
    dtype=[("id", "<u4"), ("key", "<u8"), ("vector", "<f4", (768,))],
    offset=20,
    shape=(65536,),
)
x = records["vector"].copy()
lookup = np.full(1000000, -1, dtype=np.int32)
lookup[records["id"]] = np.arange(65536)
q = []
for line in (
    Path("/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1m.1000.new.txt")
    .read_text()
    .splitlines()
):
    q.append(np.fromstring(line.split(";")[1], sep=" ", dtype=np.float32))
q = np.stack(q)
q = (q / np.linalg.norm(q.astype(np.float64), axis=1)[:, None]).astype(np.float32)
trace = list(csv.DictReader((R / "results/router59_entry_trace.csv").open()))
assert len(trace) == 3000
choices = {b: np.full(1000, -1, dtype=np.int32) for b in [1, 2, 4]}
for row in trace:
    choices[int(row["beam"])][int(row["query_id"])] = lookup[int(row["entry"])]
assert all(np.all(a >= 0) for a in choices.values())
collected = {b: {"rank": [], "distance": [], "gap_to_best": []} for b in choices}
bestdist = []
for begin in range(0, 1000, 100):
    scores = q[begin : begin + 100] @ x.T
    best = scores.max(axis=1)
    bestdist.extend((1 - best).tolist())
    for b, ids in choices.items():
        selected = scores[np.arange(len(scores)), ids[begin : begin + len(scores)]]
        ranks = 1 + np.sum(scores > selected[:, None], axis=1)
        collected[b]["rank"].extend(ranks.tolist())
        collected[b]["distance"].extend((1 - selected).tolist())
        collected[b]["gap_to_best"].extend((best - selected).tolist())
summary = {
    "diagnostic_only": True,
    "model_training_unchanged": True,
    "primary_queries": 1000,
    "anchor_count": 65536,
    "ranking_precision": "float32 matrix products;not mathematical exact oracle",
    "mean_closest_anchor_distance": float(np.mean(bestdist)),
    "beams": {},
}
for b, v in collected.items():
    summary["beams"][b] = {
        k: {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
        }
        for k, a in v.items()
    }
    summary["beams"][b]["rank1_fraction"] = float(np.mean(np.array(v["rank"]) == 1))
(R / "results/router59_entry_quality.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
logging.info("%s", json.dumps(summary, indent=2))
