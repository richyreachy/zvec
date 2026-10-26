from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
R = Path(__file__).resolve().parent.parent
anchors = np.memmap(
    R / "models59/anchors.bin",
    mode="r",
    dtype=[("id", "<u4"), ("key", "<u8"), ("vector", "<f4", (768,))],
    offset=20,
    shape=(65536,),
)
x = anchors["vector"].copy()
lookup = np.full(1000000, -1, dtype=np.int32)
lookup[anchors["id"]] = np.arange(65536)
q = np.stack(
    [
        np.fromstring(line.split(";")[1], sep=" ", dtype=np.float32)
        for line in Path(
            "/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1m.1000.new.txt"
        )
        .read_text()
        .splitlines()
    ]
)
q = (q / np.linalg.norm(q.astype(np.float64), axis=1)[:, None]).astype(np.float32)
choices = {}
for dim in [768, 384, 256]:
    rows = list(csv.DictReader((R / f"results/router60_trace{dim}.csv").open()))
    assert len(rows) == 1000
    choices[dim] = np.array([lookup[int(row["entry"])] for row in rows])
    assert np.all(choices[dim] >= 0)
old = np.array(
    [
        lookup[int(row["entry"])]
        for row in csv.DictReader((R / "results/router59_entry_trace.csv").open())
        if row["beam"] == "1"
    ]
)
values = {d: {"rank": [], "distance": [], "gap_to_closest": []} for d in choices}
for begin in range(0, 1000, 100):
    scores = q[begin : begin + 100] @ x.T
    best = scores.max(axis=1)
    for dim, ids in choices.items():
        chosen = scores[np.arange(len(scores)), ids[begin : begin + len(scores)]]
        values[dim]["rank"].extend(
            (1 + np.sum(scores > chosen[:, None], axis=1)).tolist()
        )
        values[dim]["distance"].extend((1 - chosen).tolist())
        values[dim]["gap_to_closest"].extend((best - chosen).tolist())
summary = {
    "diagnostic_only": True,
    "primary_queries": 1000,
    "trained_tree_unchanged": True,
    "float32_ranking_not_exact_oracle": True,
    "dimensions": {},
}
for dim, v in values.items():
    summary["dimensions"][dim] = {
        k: {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
        }
        for k, a in v.items()
    }
    summary["dimensions"][dim].update(
        rank1_fraction=float(np.mean(np.array(v["rank"]) == 1)),
        leaf_match_unrotated768=float(np.mean(choices[dim] == old)),
        leaf_match_rotated768=float(np.mean(choices[dim] == choices[768])),
    )
(R / "results/router60_quality.json").write_text(json.dumps(summary, indent=2) + "\n")
logging.info("%s", "QUALITY60 COMPLETE")
