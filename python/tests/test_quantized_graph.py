# Copyright 2026-present the zvec project
# Licensed under the Apache License, Version 2.0.

import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from zvec import QuantizedGraph


@unittest.skipUnless(QuantizedGraph.supported(), "quantized graph ISA unavailable")
class QuantizedGraphTest(unittest.TestCase):
    def test_row_ids_and_concurrent_queries(self):
        rows = np.random.default_rng(7).standard_normal((256, 32)).astype(np.float32)
        for metric in ("l2", "cosine"):
            with self.subTest(metric=metric):
                graph = QuantizedGraph.build(
                    rows, metric=metric, options={"threads": 1}
                )
                graph.configure({"capacity": 128})
                expected = [graph.search(row, 3) for row in rows[:24]]
                np.testing.assert_array_equal(
                    [ids[0] for ids in expected], np.arange(24)
                )
                with ThreadPoolExecutor(max_workers=2) as pool:
                    repeated = list(pool.map(lambda q: graph.search(q, 3), rows[:24]))
                np.testing.assert_array_equal(repeated, expected)
                self.assertEqual(expected[0].dtype, np.uint64)

    def test_input_validation(self):
        rows = np.random.default_rng(9).random((128, 32), dtype=np.float32)
        with self.assertRaises(TypeError):
            QuantizedGraph.build(rows.astype(np.float64))
        with self.assertRaises(ValueError):
            QuantizedGraph.build(rows, metric="unknown")
        with self.assertRaises(ValueError):
            QuantizedGraph.build(rows, options={"typo": 4})
        with self.assertRaises(ValueError):
            QuantizedGraph.build(rows, options={"encoding": {"codec": "unknown"}})
        graph = QuantizedGraph.build(rows, metric="cosine")
        with self.assertRaises(ValueError):
            graph.configure({"capacity": 0})
        with self.assertRaises(ValueError):
            graph.search(rows[0], 0)
        with self.assertRaises(ValueError):
            graph.search(rows[0, :8])
        with self.assertRaises(ValueError):
            graph.search(np.zeros(32, dtype=np.float32))
