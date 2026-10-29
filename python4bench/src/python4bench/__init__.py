"""Experimental FP32 search bindings; build Zvec with BUILD_PYTHON4BENCH=ON."""
from __future__ import annotations


class Searcher:
    """Prepare a read-only collection with numeric dataset primary keys.

    The native methods accept aligned contiguous float32 arrays without copies.
    Inputs must not be mutated while a search is running. Results own their
    memory. Search calls may run concurrently; explicit collection.close()
    invalidates the searcher. Configuration is copied at preparation time.
    """

    def __init__(self, collection, field="vector", param=None, filter=""):
        try:
            from zvec._zvec import _BenchSearcher
        except ImportError as exc:
            raise ImportError(
                "Rebuild this branch of zvec with BUILD_PYTHON4BENCH=ON; "
                "see python4bench/README.md"
            ) from exc
        self._collection = collection
        self._native = _BenchSearcher(collection._obj, field, param, filter or "")
        # Cache bound native methods: no Python query objects on the hot path.
        self.search = self._native.search
        self.search_list = self._native.search_list
        self.search_batch = self._native.search_batch
        self.time_native = self._native.time_native


class CoreSearcher:
    """Read-only native HNSW/SymQG index; stored keys must be dataset IDs.

    index_params/query_params use the core JSON schema, including
    use_quantized_graph, quantized_graph_model and quantized_graph_max_expansions.
    This entry point does not implement database scalar filtering or external
    reference-index refinement. Label its measurements as core-index results.
    """

    def __init__(self, path, index_params, query_params=None):
        import json
        from zvec._zvec import _BenchSearcher

        query_params = dict(query_params or {})
        allowed = {
            "index_type", "ef_search", "radius", "is_linear", "prefetch_offset", "prefetch_lines",
            "quantized_graph_margin", "quantized_graph_bucket_width",
            "quantized_graph_max_expansions", "quantized_graph_route_beam",
            "quantized_graph_uncertainty_bonus",
        }
        unknown = query_params.keys() - allowed
        if unknown:
            raise ValueError(f"Unsupported core query parameters: {sorted(unknown)}; pass k to search()")
        if query_params.get("index_type", "kHNSW") != "kHNSW":
            raise ValueError("CoreSearcher requires kHNSW query parameters")
        query_params["index_type"] = "kHNSW"
        self._native = _BenchSearcher(str(path), json.dumps(index_params),
                                      json.dumps(query_params))
        self.search = self._native.search
        self.search_list = self._native.search_list
        self.search_batch = self._native.search_batch
        self.time_native = self._native.time_native

    @staticmethod
    def build(path, vectors, index_params):
        """Build a full-precision HNSW graph with row-number IDs at a new path."""
        import json
        from zvec._zvec import _BuildBenchCoreIndex

        _BuildBenchCoreIndex(str(path), vectors, json.dumps(index_params))


__all__ = ["CoreSearcher", "Searcher"]
