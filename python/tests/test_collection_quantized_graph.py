# Copyright 2026-present the zvec project
# Licensed under the Apache License, Version 2.0.

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zvec


@unittest.skipUnless(zvec.QuantizedGraph.supported(), "QG unavailable on this CPU")
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
                c.create_index("v", param)
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
                graph = zvec.QuantizedGraph.build(
                    rows,
                    metric="l2" if metric == zvec.MetricType.L2 else "cosine",
                    options=options,
                )
                for q in rows[::33]:
                    query = zvec.Query(field_name="v", vector=q, param=p)
                    expected = graph.search(q, 10)
                    hits = c.query(query, topk=10, include_vector=True)
                    np.testing.assert_array_equal(
                        [int(h.id[2:]) for h in hits], expected
                    )
                    got, scores = c.query_internal_ids(
                        query, topk=10, return_scores=True
                    )
                    np.testing.assert_array_equal(got, expected)
                    vectors = rows[expected]
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
