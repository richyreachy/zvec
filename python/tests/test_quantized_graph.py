# Copyright 2026-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import unittest
from pathlib import Path
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

    def test_segmented_index_roundtrip(self):
        rows = np.random.default_rng(8).random((256, 32), dtype=np.float32)
        for metric in ("l2", "cosine"):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "index.zvec")
                graph = QuantizedGraph.build(rows, metric=metric, options={"threads": 1})
                expected = [graph.search(q, 10) for q in rows[:12]]
                graph.save(path)
                self.assertEqual(list(Path(tmp).iterdir()), [Path(path)])
                reopened = QuantizedGraph.open(path)
                np.testing.assert_array_equal([reopened.search(q, 10) for q in rows[:12]], expected)
                reopened.save(path)
                np.testing.assert_array_equal(QuantizedGraph.open(path).search(rows[0], 10), expected[0])
                with open(path, "r+b") as f:
                    f.truncate(100)
                with self.assertRaises((ValueError, RuntimeError)):
                    QuantizedGraph.open(path)

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
