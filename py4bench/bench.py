"""Measure complete query calls on one existing snapshot; never retune search."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zvec
from py4bench import Searcher
from py4bench.adapters import ANNSearchAdapter, VectorDBSearchAdapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--queries", required=True, help="2D float32 .npy file")
    p.add_argument("--field", default="vector")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--index-type", choices=["hnsw", "vamana"], default="hnsw")
    p.add_argument("--query-params", default='{"ef":100}', help="JSON constructor arguments")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if args.threads < 1 or args.repeats < 1:
        p.error("threads and repeats must be positive")
    X = np.require(np.load(args.queries), dtype=np.float32, requirements=["C", "A"])
    if X.ndim != 2 or not len(X):
        p.error("queries must be a nonempty 2D array")
    cls = zvec.HnswQueryParam if args.index_type == "hnsw" else zvec.VamanaQueryParam
    param = cls(**json.loads(args.query_params))
    reader = zvec.open(args.path, zvec.CollectionOption(read_only=True))
    try:
        start = time.perf_counter()
        search = Searcher(reader, args.field, param)
        preparation = time.perf_counter() - start
        ann = ANNSearchAdapter(reader, args.field, param, args.threads)
        vdb = VectorDBSearchAdapter(reader, args.field, param)
        lists = X.tolist()  # Represents a harness that already supplies lists.
        def public(q):
            return [int(d.id) for d in reader.query(
                zvec.Query(args.field, vector=q, param=param), topk=args.k
            )]
        # Verify actual external IDs on every query before timing.
        expected = [public(q) for q in X]
        for q, want in zip(X, expected):
            if search.search(q, args.k).tolist() != want:
                raise AssertionError("fast/public query results differ")
        paths = {
            "public_numpy": (public, X),
            "thin_numpy": (lambda q: search.search(q, args.k), X),
            "ann_adapter": (lambda q: ann.query(q, args.k), X),
            "vdb_list_adapter": (lambda q: vdb.search_embedding(q, args.k), lists),
        }
        samples = {name: [] for name in paths}
        samples["native_batch_with_output"] = []
        # Reuse workers; pool submission overhead remains in timed single-query runs.
        with ThreadPoolExecutor(args.threads) as pool:
            for repeat in range(args.repeats + 1):
                names = list(paths)
                names = names[repeat % len(names):] + names[:repeat % len(names)]
                for name in names:
                    fn, vectors = paths[name]
                    start = time.perf_counter()
                    output = list(map(fn, vectors)) if args.threads == 1 else list(pool.map(fn, vectors))
                    elapsed = time.perf_counter() - start
                    if [list(row) for row in output] != expected:
                        raise AssertionError(f"{name}: result mismatch")
                    if repeat:
                        samples[name].append(len(X) / elapsed)
                start = time.perf_counter()
                output = search.search_batch(X, args.k, args.threads)
                elapsed = time.perf_counter() - start
                if [list(row) for row in output] != expected:
                    raise AssertionError("batch result mismatch")
                if repeat:
                    samples["native_batch_with_output"].append(len(X) / elapsed)
        print(json.dumps({
            "configuration": vars(args), "queries": len(X),
            "preparation_seconds": preparation,
            "note": "Local bridge diagnostic; not an official harness score or C++ kernel baseline",
            "median_qps": {key: statistics.median(v) for key, v in samples.items()},
            "qps_samples": samples,
        }, indent=2))
    finally:
        reader.close()


if __name__ == "__main__":
    main()
