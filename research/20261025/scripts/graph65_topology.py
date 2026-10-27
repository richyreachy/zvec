from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
R = Path(__file__).resolve().parent.parent
n = 1000000
dtype = np.dtype([("key", "<u8"), ("degree", "<u4"), ("ids", "<u4", (32,))])
paths = {
    "original": R / "models64/full_graph.bin",
    "nearest": R / "models65/full_nearest.bin",
    "angular": R / "models65/full_angular.bin",
}
original = np.memmap(paths["original"], dtype=dtype, mode="r", offset=32, shape=(n,))
key_to_id = np.empty(n, dtype=np.int64)
key_to_id[original["key"]] = np.arange(n)
gt = []
for line in (
    Path("/home/xingrui/bench/dataset/cohere/1m/neighbors.txt").read_text().splitlines()
):
    query, tail = line.split(";")
    assert int(query) == len(gt)
    gt.append([int(x) for x in tail.split()[:10]])
gt = key_to_id[np.array(gt)]
assert gt.shape == (1000, 10)
report = {}
for mode, path in paths.items():
    graph = np.memmap(path, dtype=dtype, mode="r", offset=32, shape=(n,))
    entry = int(np.fromfile(path, dtype="<u4", count=8)[5])
    seen = np.zeros(n, dtype=bool)
    frontier = np.array([entry])
    seen[frontier] = True
    while len(frontier):
        nxt = np.unique(graph["ids"][frontier].reshape(-1))
        nxt = nxt[nxt < n]
        frontier = nxt[~seen[nxt]]
        seen[frontier] = True
    indegree = np.zeros(n, dtype=np.int64)
    overlap = 0
    for lo in range(0, n, 8192):
        a = graph[lo : lo + 8192]["ids"]
        valid = a[a < n]
        np.add.at(indegree, valid, 1)
        overlap += int(
            (
                (a[:, :, None] == original[lo : lo + len(a)]["ids"][:, None, :])
                & (a[:, :, None] < n)
            )
            .any(axis=2)
            .sum()
        )
    report[mode] = {
        "reachable": int(seen.sum()),
        "unreachable_primary_top10_occurrences": int((~seen[gt]).sum()),
        "unreachable_primary_queries": int((~seen[gt]).any(axis=1).sum()),
        "indegree_zero": int((indegree == 0).sum()),
        "indegree_max": int(indegree.max()),
        "indegree_p99": float(np.quantile(indegree, 0.99)),
        "edges": int(indegree.sum()),
        "edges_shared_with_original": overlap,
        "mean_shared_per_node": overlap / n,
    }
(R / "results/graph65_topology.json").write_text(json.dumps(report, indent=2) + "\n")
logging.info("%s", json.dumps(report, indent=2))
