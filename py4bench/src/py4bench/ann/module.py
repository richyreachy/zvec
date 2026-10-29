"""ANN-Benchmarks algorithm entry point (optional ann_benchmarks dependency)."""
from __future__ import annotations

from tempfile import TemporaryDirectory

import zvec
from ann_benchmarks.algorithms.base.module import BaseANN

from ..adapters import ANNSearchAdapter, fp32
from .. import CoreSearcher


class ZvecBench(BaseANN):
    """Build numeric row IDs and reuse the same native adapter for each sweep."""

    def __init__(self, metric, index_type="hnsw", index_params=None, threads=1):
        metrics = {"euclidean": zvec.MetricType.L2, "angular": zvec.MetricType.COSINE}
        if metric not in metrics or index_type not in ("hnsw", "vamana"):
            raise ValueError("supported: euclidean/angular with hnsw/vamana")
        self.metric = metrics[metric]
        self.index_type = index_type
        self.index_params = dict(index_params or {})
        self.threads = threads
        self.collection = None
        self.directory = None
        self.adapter = None
        self.name = f"Zvec-py4bench({metric},{index_type},{self.index_params},threads={threads})"

    def fit(self, X):
        self.done()
        X = fp32(X)
        if X.ndim != 2:
            raise ValueError("training vectors must be two-dimensional")
        params = dict(self.index_params)
        if isinstance(params.get("quantize_type"), str):
            params["quantize_type"] = getattr(zvec.QuantizeType, params["quantize_type"])
        cls = zvec.HnswIndexParam if self.index_type == "hnsw" else zvec.VamanaIndexParam
        schema = zvec.CollectionSchema(
            name="ann_bench", vectors=[zvec.VectorSchema(
                "vector", zvec.DataType.VECTOR_FP32, X.shape[1],
                index_param=cls(metric_type=self.metric, **params),
            )],
        )
        self.directory = TemporaryDirectory(prefix="zvec-ann-")
        path = self.directory.name + "/index"
        writer = zvec.create_and_open(path, schema)
        try:
            for start in range(0, len(X), 1024):
                statuses = writer.insert([
                    zvec.Doc(id=str(i), vectors={"vector": X[i]})
                    for i in range(start, min(start + 1024, len(X)))
                ])
                if not all(s.ok() for s in statuses):
                    raise RuntimeError("ANN training insert failed")
            writer.optimize()
        finally:
            writer.close()
        self.collection = zvec.open(path, zvec.CollectionOption(read_only=True))
        self.set_query_arguments({})

    def set_query_arguments(self, params):
        cls = zvec.HnswQueryParam if self.index_type == "hnsw" else zvec.VamanaQueryParam
        self.adapter = ANNSearchAdapter(
            self.collection, param=cls(**params), threads=self.threads
        )
        self.name = (f"Zvec-py4bench({self.index_type},{self.index_params},"
                     f"query={params},threads={self.threads})")

    def query(self, q, n):
        return self.adapter.query(q, n)

    def batch_query(self, X, n):
        self.adapter.batch_query(X, n)

    def get_batch_results(self):
        return self.adapter.get_batch_results()

    def done(self):
        self.adapter = None
        if self.collection is not None:
            self.collection.close()
            self.collection = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None


class SymQGBench(BaseANN):
    """Core HNSW graph with optional SymQG projection, using the core JSON schema."""

    def __init__(self, metric, index_params=None, threads=1):
        metrics = {"euclidean": "kL2sq", "angular": "kCosine"}
        if metric not in metrics:
            raise ValueError("SymQGBench supports euclidean/angular")
        self.params = dict(index_params or {})
        self.params.update(index_type="kHNSW", data_type="DT_FP32", metric_type=metrics[metric])
        self.params.setdefault("use_quantized_graph", True)
        self.params.setdefault("use_id_map", True)
        self.threads = threads
        self.directory = None
        self.adapter = None
        self.name = f"SymQG-py4bench({self.params},threads={threads})"

    def fit(self, X):
        self.done()
        X = fp32(X)
        if X.ndim != 2:
            raise ValueError("training vectors must be two-dimensional")
        self.params["dimension"] = X.shape[1]
        self.directory = TemporaryDirectory(prefix="symqg-ann-")
        self.path = self.directory.name + "/graph.index"
        CoreSearcher.build(self.path, X, self.params)
        self.set_query_arguments({})

    def set_query_arguments(self, params):
        self.adapter = None
        self.adapter = ANNSearchAdapter.from_searcher(
            CoreSearcher(self.path, self.params, params), self.threads
        )
        self.name = f"SymQG-py4bench({self.params},query={params},threads={self.threads})"

    def query(self, q, n):
        return self.adapter.query(q, n)

    def batch_query(self, X, n):
        self.adapter.batch_query(X, n)

    def get_batch_results(self):
        return self.adapter.get_batch_results()

    def done(self):
        self.adapter = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
