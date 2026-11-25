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

import numpy as np

from zvec import QuantizedGraph


@unittest.skipUnless(QuantizedGraph.supported(), "quantized graph ISA unavailable")
class QuantizedGraphTest(unittest.TestCase):
    def test_segmented_index_roundtrip(self):
        rows = np.random.default_rng(8).random((256, 32), dtype=np.float32)
        for metric in ("l2", "cosine"):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "index.zvec")
                graph = QuantizedGraph.build(rows, metric=metric, options={"threads": 1})
                graph.save(path)
                self.assertEqual(list(Path(tmp).iterdir()), [Path(path)])
                reopened = QuantizedGraph.open(path)
                reopened.save(path)
                self.assertIsInstance(QuantizedGraph.open(path), QuantizedGraph)
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
