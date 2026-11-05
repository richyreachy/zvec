import numpy as np
import pytest
from pathlib import Path

pytest.importorskip("ann_benchmarks")
from py4bench.ann.module import SymQGBench, ZvecBench


def test_framework_config():
    pytest.importorskip("yaml")
    from ann_benchmarks.definitions import get_definitions, instantiate_algorithm
    root = Path(__file__).resolve().parents[1] / "integrations"
    definitions = get_definitions(16, "float", "euclidean", base_dir=str(root))
    assert len(definitions) == 1
    assert definitions[0].query_argument_groups == [
        [{"ef_search": 100, "quantized_graph_max_expansions": 200}],
        [{"ef_search": 200, "quantized_graph_max_expansions": 200}],
    ]
    algo = instantiate_algorithm(definitions[0])
    assert isinstance(algo, SymQGBench)
    algo.done()


@pytest.mark.parametrize("core", [False, True])
def test_ann_lifecycle(core):
    X = np.random.default_rng(2).normal(size=(64, 16)).astype(np.float32)
    algo = (SymQGBench("euclidean", {"use_quantized_graph": False}) if core
            else ZvecBench("euclidean"))
    try:
        algo.fit(X)
        algo.set_query_arguments({"ef_search" if core else "ef": 100})
        assert algo.query(X[0], 5)[0] == 0
        algo.batch_query(X[:3], 5)
        assert [row[0] for row in algo.get_batch_results()] == [0, 1, 2]
        algo.fit(X)  # cleanup and rebuild
        assert algo.query(X[3], 5)[0] == 3
    finally:
        algo.done()
