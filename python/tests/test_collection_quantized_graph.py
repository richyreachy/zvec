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

import pickle
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zvec


class TestCollectionQuantizedGraph(unittest.TestCase):
    def test_reopen_query_and_row_mapping(self):
        rows = np.random.default_rng(21).standard_normal((256, 32)).astype("float32")
        options = {"threads": 1, "router_anchors": 32, "router_dimension": 8}
        for metric in (zvec.MetricType.L2, zvec.MetricType.COSINE):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "docs")
                schema = zvec.CollectionSchema(
                    "graphs",
                    fields=zvec.FieldSchema("bucket", zvec.DataType.INT32),
                    vectors=zvec.VectorSchema(
                        "v", zvec.DataType.VECTOR_FP32, dimension=32
                    ),
                )
                c = zvec.create_and_open(path, schema)
                docs = [
                    zvec.Doc(id=f"k-{i}", fields={"bucket": i % 2}, vectors={"v": row})
                    for i, row in enumerate(rows)
                ]
                self.assertTrue(all(s.ok() for s in c.insert(docs)))
                param = zvec.QuantizedGraphIndexParam(metric, options)
                try:
                    c.create_index("v", param)
                except RuntimeError as exc:
                    c.close()
                    if str(exc).endswith(
                        ("quantized graph unavailable", "QG unavailable on this platform")
                    ):
                        self.skipTest(str(exc))
                    raise
                c.optimize()
                c.close()
                c = zvec.open(path, zvec.CollectionOption(read_only=True))
                self.addCleanup(c.close)
                saved = c.schema.vector("v").index_param
                self.assertEqual(saved.type, zvec.IndexType.QUANTIZED_GRAPH)
                self.assertEqual(saved.options, options)
                self.assertEqual(saved.to_dict()["type"], "QUANTIZED_GRAPH")
                self.assertEqual(pickle.loads(pickle.dumps(param)).options, options)
                p = pickle.loads(
                    pickle.dumps(zvec.QuantizedGraphQueryParam(capacity=128))
                )
                batch = rows[:3]
                for filt in (None, "bucket = 1", "bucket = 2"):
                    args = dict(topk=7, filter=filt, return_scores=True)
                    expected = [
                        c.query_internal_ids(zvec.Query("v", vector=v, param=p), **args)
                        for v in batch
                    ]
                    for data in (batch[0], batch[:1], batch):
                        ids, scores = c.query_internal_ids("v", data, param=p, **args)
                        for got, col in ((ids, 0), (scores, 1)):
                            ref = [r[col] for r in expected]
                            ref = ref[0] if data.ndim == 1 else ref[:len(data)]
                            np.testing.assert_array_equal(got, ref)
                for bad in (batch.astype("float64"), batch[:, ::-1], batch[:, :-1].copy(),
                            batch[:0], batch[None, ...]):
                    with self.assertRaises((TypeError, ValueError)):
                        c.query_internal_ids("v", bad, param=p)

                def search(v):
                    return c.query_internal_ids("v", v, topk=3, param=p)

                ref = search(rows[:32])
                self.assertEqual(ref.dtype, np.int64)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    np.testing.assert_array_equal(list(pool.map(search, rows[:32])), ref)
                for i in range(0, len(rows), 33):
                    q = rows[i]
                    query = zvec.Query(field_name="v", vector=q, param=p)
                    hits = c.query(query, topk=10, include_vector=True)
                    got, scores = c.query_internal_ids(
                        query, topk=10, return_scores=True
                    )
                    self.assertEqual(c.resolve_internal_ids(got.tolist()), [h.id for h in hits])
                    self.assertEqual(hits[0].id, f"k-{i}")
                    vectors = rows[[int(h.id[2:]) for h in hits]]
                    if metric == zvec.MetricType.L2:
                        distances = np.square(vectors - q).sum(axis=1)
                    else:
                        distances = 1 - vectors @ q / (
                            np.linalg.norm(vectors, axis=1) * np.linalg.norm(q)
                        )
                    np.testing.assert_allclose(scores, distances, atol=2e-5, rtol=2e-5)
                    np.testing.assert_allclose(
                        [h.score for h in hits], distances, atol=2e-5, rtol=2e-5
                    )
                    np.testing.assert_array_equal(hits[0].vectors["v"], vectors[0])
                hits = c.query(query, topk=10, filter="bucket = 1")
                self.assertEqual(len(hits), 10)
                self.assertTrue(all(h.fields["bucket"] == 1 for h in hits))
                c.close()
                c = zvec.open(path)
                self.assertTrue(c.delete("k-0").ok())
                self.assertTrue(
                    c.insert(
                        zvec.Doc(
                            id="replacement",
                            fields={"bucket": 0},
                            vectors={"v": rows[0]},
                        )
                    ).ok()
                )
                c.optimize()
                c.close()
                c = zvec.open(path, zvec.CollectionOption(read_only=True))
                self.addCleanup(c.close)
                hits = c.query(
                    zvec.Query(field_name="v", vector=rows[0], param=p), topk=10
                )
                self.assertEqual(hits[0].id, "replacement")
                self.assertNotIn("k-0", [h.id for h in hits])
                c.close()

    def test_named_build_options(self):
        original = {"passes": 2}
        param = zvec.QuantizedGraphIndexParam(
            options=original, threads=1, encoding={"codebook": "gaussian"}
        )
        self.assertEqual(original, {"passes": 2})
        self.assertEqual(param.options, {
            "passes": 2, "threads": 1, "encoding": {"codebook": "gaussian"}
        })
        self.assertEqual(pickle.loads(pickle.dumps(param)).options, param.options)
        with self.assertRaises(ValueError):
            zvec.QuantizedGraphIndexParam(options=original, passes=1)
        with self.assertRaises(ValueError):
            zvec.QuantizedGraphIndexParam(thread=1)

    def test_invalid_build_and_query_options(self):
        with self.assertRaises(ValueError):
            zvec.QuantizedGraphQueryParam(capacity=0)
        for metric, opts in [
            (zvec.MetricType.IP, {}),
            (zvec.MetricType.L2, {"unknown": 2}),
            (
                zvec.MetricType.L2,
                {"encoding": {"block_dimension": 3, "codebook": "hadamard"}},
            ),
        ]:
            with self.subTest(options=opts), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises((RuntimeError, ValueError)):
                    zvec.create_and_open(
                        str(Path(tmp) / "invalid"),
                        zvec.CollectionSchema(
                            "graphs",
                            vectors=zvec.VectorSchema(
                                "v",
                                zvec.DataType.VECTOR_FP32,
                                dimension=32,
                                index_param=zvec.QuantizedGraphIndexParam(metric, opts),
                            ),
                        ),
                    )
