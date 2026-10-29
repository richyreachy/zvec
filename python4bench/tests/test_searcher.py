from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import zvec

from python4bench import Searcher
from python4bench.adapters import ANNSearchAdapter, VectorDBSearchAdapter


@pytest.fixture(params=["flat", "hnsw", "vamana"])
def dataset(tmp_path, request):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 32)).astype(np.float32)
    cls = {"flat": zvec.FlatIndexParam, "hnsw": zvec.HnswIndexParam,
           "vamana": zvec.VamanaIndexParam}[request.param]
    schema = zvec.CollectionSchema(name="bench", fields=[
        zvec.FieldSchema("rank", zvec.DataType.INT64),
    ], vectors=[zvec.VectorSchema("vector", zvec.DataType.VECTOR_FP32, 32,
                                index_param=cls(metric_type=zvec.MetricType.L2))])
    path = str(tmp_path / "index")
    writer = zvec.create_and_open(path, schema)
    # Deliberately unrelated to insertion order; delete one row as well.
    keys = [str(1000 + 7 * i) for i in range(len(X))]
    assert all(s.ok() for s in writer.insert([
        zvec.Doc(id=keys[i], fields={"rank": i}, vectors={"vector": v})
        for i, v in enumerate(X)
    ]))
    writer.delete(keys[4])
    with pytest.raises(ValueError, match="read-only"):
        Searcher(writer)
    writer.optimize()
    writer.close()
    reader = zvec.open(path, zvec.CollectionOption(read_only=True))
    try:
        yield reader, X
    finally:
        reader.close()


def expected(reader, q, k, filter="", param=None):
    return [int(d.id) for d in reader.query(
        zvec.Query("vector", vector=q, param=param), topk=k, filter=filter
    )]


def test_equivalence_mapping_padding_and_lifetime(dataset):
    reader, X = dataset
    searcher = Searcher(reader)
    saved = searcher.search(X[0], 100)
    want = expected(reader, X[0], 100)
    assert len(saved) == 79
    assert saved.tolist() == want
    assert searcher.search_list(X[1], 10) == expected(reader, X[1], 10)
    searcher.search(X[2], 10)
    np.testing.assert_array_equal(saved, want)
    assert saved.dtype == np.int64


def test_batch_concurrent_and_adapters(dataset):
    reader, X = dataset
    searcher = Searcher(reader)
    want = [expected(reader, q, 8) for q in X[:12]]
    for threads in (1, 3):
        assert [a.tolist() for a in searcher.search_batch(X[:12], 8, threads)] == want
    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(lambda q: searcher.search(q, 8).tolist(), X[:12]))
    assert got == want
    ann = ANNSearchAdapter(reader)
    ann.batch_query(X[:12], 8)
    assert [a.tolist() for a in ann.get_batch_results()] == want
    assert ann.query(X[0], 8).tolist() == want[0]
    vdb = VectorDBSearchAdapter(reader, "vector")
    assert vdb.search_embedding(X[0].tolist(), 8) == want[0]
    assert searcher.search_batch(X[:0], 8) == []


def test_filter(dataset):
    reader, X = dataset
    searcher = Searcher(reader, filter="rank >= 50")
    assert searcher.search(X[0], 50).tolist() == expected(reader, X[0], 50, "rank >= 50")


def test_invalid_input(dataset):
    reader, X = dataset
    s = Searcher(reader)
    unaligned = np.ndarray((32,), dtype=np.float32, buffer=bytearray(129), offset=1)
    for q in (X[0].astype(np.float64), X[0, ::2], X[0, :-1], X[:1], unaligned):
        with pytest.raises(ValueError):
            s.search(q)
    with pytest.raises(TypeError):
        s.search(X[0].tolist())
    with pytest.raises(ValueError):
        s.search(X[0], 0)
    with pytest.raises(ValueError):
        s.search_batch(X, 10, 0)


def test_closed_collection(dataset):
    reader, X = dataset
    s = Searcher(reader)
    ids = s.search(X[0], 5)
    saved = ids.copy()
    reader.close()
    np.testing.assert_array_equal(ids, saved)
    with pytest.raises(RuntimeError):
        s.search(X[0])


@pytest.mark.parametrize("keys", [["row-1"], ["1", "01"], ["-2"]])
def test_reject_ambiguous_ids(tmp_path, keys):
    schema = zvec.CollectionSchema(name="ids", vectors=[zvec.VectorSchema(
        "vector", zvec.DataType.VECTOR_FP32, 4, index_param=zvec.FlatIndexParam()
    )])
    path = str(tmp_path / "index")
    w = zvec.create_and_open(path, schema)
    w.insert([zvec.Doc(id=k, vectors={"vector": [1., 2., 3., 4.]}) for k in keys])
    w.close()
    r = zvec.open(path, zvec.CollectionOption(read_only=True))
    try:
        with pytest.raises(ValueError):
            Searcher(r)
    finally:
        r.close()
