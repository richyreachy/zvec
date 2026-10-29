import os

import numpy as np
import pytest

from py4bench import CoreSearcher
from py4bench.adapters import ANNSearchAdapter, VectorDBSearchAdapter


def test_core_build_search_and_batch(tmp_path):
    X = np.random.default_rng(13).normal(size=(100, 32)).astype(np.float32)
    params = dict(index_type="kHNSW", data_type="DT_FP32", metric_type="kL2sq",
                  dimension=32, m=16, ef_construction=100, use_id_map=True)
    path = str(tmp_path / "graph.index")
    CoreSearcher.build(path, X, params)
    with pytest.raises(ValueError, match="overwrite"):
        CoreSearcher.build(path, X, params)
    search = CoreSearcher(path, params, {"ef_search": 100})
    assert search.time_native(X[:8], 10) > 0
    wanted = np.argsort(((X[None, :, :] - X[:8, None, :]) ** 2).sum(axis=2), axis=1)[:, :10]
    got = search.search_batch(X[:8], 10, 3)
    np.testing.assert_array_equal(got, wanted)
    assert search.search_list(X[0], 10) == wanted[0].tolist()
    ann = ANNSearchAdapter.from_searcher(search)
    vdb = VectorDBSearchAdapter.from_searcher(search)
    assert ann.query(X[0], 10).tolist() == vdb.search_embedding(X[0].tolist(), 10)
    with pytest.raises(ValueError):
        search.search(X[0, ::2], 10)


def test_core_rejects_invalid_json(tmp_path):
    with pytest.raises(ValueError):
        CoreSearcher(str(tmp_path / "missing"), {})
    with pytest.raises(ValueError, match="Unsupported"):
        CoreSearcher("unused", {}, {"is_using_refiner": True})


@pytest.mark.skipif(os.environ.get("ZVEC_TEST_SYMQG") != "1",
                    reason="set ZVEC_TEST_SYMQG=1 on an AVX512/VNNI host")
def test_symqg_projection(tmp_path):
    X = np.random.default_rng(7).normal(size=(256, 64)).astype(np.float32)
    params = dict(index_type="kHNSW", data_type="DT_FP32", metric_type="kL2sq",
                  dimension=64, m=16, ef_construction=100, use_id_map=True,
                  use_quantized_graph=True)
    path = str(tmp_path / "qg.index")
    CoreSearcher.build(path, X, params)
    search = CoreSearcher(path, params, {
        "ef_search": 100, "quantized_graph_max_expansions": 200,
        "quantized_graph_route_beam": 0,
    })
    single = [search.search(q, 10).tolist() for q in X[:8]]
    assert [row.tolist() for row in search.search_batch(X[:8], 10, 2)] == single
    assert [row[0] for row in single] == list(range(8))
