"""Single-thread bridge overhead with identical native HNSW/SymQG parameters."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from python4bench import CoreSearcher
from python4bench.adapters import ANNSearchAdapter, VectorDBSearchAdapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--index-params", required=True, help="core index JSON file")
    p.add_argument("--query-params", default="{}", help="core query JSON string")
    p.add_argument("--queries", required=True, help="query matrix .npy")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if args.repeats <= 0:
        p.error("repeats must be positive")
    X = np.require(np.load(args.queries), dtype=np.float32, requirements=["C", "A"])
    if X.ndim != 2 or not len(X):
        p.error("queries must be nonempty and two-dimensional")
    index_params = json.loads(Path(args.index_params).read_text())
    query_params = json.loads(args.query_params)
    start = time.perf_counter()
    search = CoreSearcher(args.path, index_params, query_params)
    prepare = time.perf_counter() - start
    ann = ANNSearchAdapter.from_searcher(search)
    vdb = VectorDBSearchAdapter.from_searcher(search)
    lists = X.tolist()
    expected = [list(row) for row in search.search_batch(X, args.k)]
    calls = {
        "thin_numpy": lambda: [search.search(q, args.k) for q in X],
        "ann_adapter": lambda: [ann.query(q, args.k) for q in X],
        "vdb_list_adapter": lambda: [vdb.search_embedding(q, args.k) for q in lists],
        "native_batch_with_output": lambda: search.search_batch(X, args.k),
    }
    samples = {key: [] for key in ["native_loop", *calls]}
    for repeat in range(args.repeats + 1):
        names = list(samples)
        names = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in names:
            if name == "native_loop":
                elapsed = search.time_native(X, args.k)
            else:
                start = time.perf_counter()
                result = calls[name]()
                elapsed = time.perf_counter() - start
                if [list(row) for row in result] != expected:
                    raise AssertionError(f"{name}: result mismatch")
            if repeat:
                samples[name].append(len(X) / elapsed)
    qps = {key: statistics.median(value) for key, value in samples.items()}
    print(json.dumps({
        "index_params": index_params, "query_params": query_params,
        "k": args.k, "threads": 1, "queries": len(X), "prepare_seconds": prepare,
        "median_qps": qps, "qps_samples": samples,
        "qps_loss_vs_native_loop": {k: 1 - v / qps["native_loop"] for k, v in qps.items()},
        "note": "Same run() including result allocation; native loop excludes Python output. Not official harness scores or recall validation.",
    }, indent=2))


if __name__ == "__main__":
    main()
