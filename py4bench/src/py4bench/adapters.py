"""Small adapters that preserve each harness's query/return conventions."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from . import Searcher


def fp32(value):
    """List/dtype/layout conversion is deliberately inside the timed call."""
    if (isinstance(value, np.ndarray) and value.dtype == np.float32
            and value.flags.c_contiguous and value.flags.aligned):
        return value
    return np.require(value, dtype=np.float32, requirements=["C", "A"])


class ANNSearchAdapter:
    """Query side of ANN-Benchmarks for an already-built collection.

    The algorithm's fit() builds/opens its collection and creates this object;
    set_query_arguments() replaces it when query parameters change. No prepared
    query hooks are exposed, so conversion and result materialization are timed.
    """

    def __init__(self, collection, field="vector", param=None, threads=1):
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.searcher = Searcher(collection, field, param)
        self.threads = threads
        self.res = None

    @classmethod
    def from_searcher(cls, searcher, threads=1):
        """Use an explicit CoreSearcher, labeling results as core-index ANN."""
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        obj = cls.__new__(cls)
        obj.searcher, obj.threads, obj.res = searcher, threads, None
        return obj

    def query(self, q, n):
        return self.searcher.search(fp32(q), n)

    def batch_query(self, X, n):
        self.res = self.searcher.search_batch(fp32(X), n, self.threads)

    def get_batch_results(self):
        if self.res is None:
            raise RuntimeError("batch_query must run before get_batch_results")
        return self.res


class VectorDBSearchAdapter:
    """VectorDBBench search_embedding implementation for a fixed filter."""

    def __init__(self, collection, field="dense", param=None, filter=""):
        self.searcher = Searcher(collection, field, param, filter)

    @classmethod
    def from_searcher(cls, searcher):
        """Core-only diagnostic; not a database/filter benchmark implementation."""
        obj = cls.__new__(cls)
        obj.searcher = searcher
        return obj

    def search_embedding(self, query, k=100, filters=None):
        if filters:
            raise ValueError("prepare the filter expression before creating the adapter")
        return self.searcher.search_list(fp32(query), k)


class VectorDBBenchMixin:
    """Place before the upstream Zvec client in its MRO.

    Build/insert/optimize and prepare_filter remain upstream operations. The
    searcher is prepared inside each worker's init() only for read-only sessions.
    """

    @contextmanager
    def init(self):
        with super().init():
            self._bench_search = None
            if self.option.read_only:
                self._bench_search = VectorDBSearchAdapter(
                    self.collection, "dense", self.query_param, self.expr
                )
            try:
                yield
            finally:
                self._bench_search = None

    def search_embedding(self, query, k=100, filters=None):
        if self._bench_search is None:
            raise RuntimeError("prepare_filter and a read-only init session are required")
        return self._bench_search.search_embedding(query, k, filters)
