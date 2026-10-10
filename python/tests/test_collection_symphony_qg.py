# Copyright 2025-present the zvec project
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
from __future__ import annotations

import platform
import sys

import pytest
import math
import zvec

pytestmark = pytest.mark.skipif(
    not (sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")),
    reason="SymphonyQG only supported on Linux x86_64",
)
from zvec import (
    Collection,
    CollectionOption,
    DataType,
    Doc,
    FieldSchema,
    HnswIndexParam,
    HnswQueryParam,
    MetricType,
    VectorSchema,
    Query,
)


@pytest.mark.parametrize("enable_mmap", [True, False])
@pytest.mark.parametrize("use_contiguous_memory", [True, False])
def test_symphony_qg_optimize_reopen(tmp_path, enable_mmap, use_contiguous_memory):
    schema = zvec.CollectionSchema(
        name="symphony_qg",
        vectors=[
            VectorSchema(
                "embedding",
                DataType.VECTOR_FP32,
                dimension=128,
                index_param=HnswIndexParam(
                    metric_type=MetricType.L2,
                    m=17,
                    symphony_qg=True,
                    use_contiguous_memory=use_contiguous_memory,
                ),
            )
        ],
    )
    path = str(tmp_path / "collection")
    option = CollectionOption(enable_mmap=enable_mmap)
    collection = zvec.create_and_open(path=path, schema=schema, option=option)
    docs = [
        Doc(id=str(i), vectors={"embedding": [i / 100.0] * 128}) for i in range(1200)
    ]
    # Keep all 1200 documents while staying below the per-write batch limit.
    batch_size = 512
    for offset in range(0, len(docs), batch_size):
        batch = docs[offset : offset + batch_size]
        statuses = collection.insert(batch)
        assert len(statuses) == len(batch)
        assert all(status.ok() for status in statuses)
    collection.optimize()
    query = Query(
        field_name="embedding", vector=[1.42] * 128, param=HnswQueryParam(ef=100)
    )
    before = collection.query(query, topk=5)
    assert before[0].id == "142"
    # Keep a second, unoptimized block across reopen.
    assert collection.insert(Doc(id="later", vectors={"embedding": [20.0] * 128})).ok()
    del collection
    reopened = zvec.open(path=path, option=option)
    try:
        assert reopened.schema.vector("embedding").index_param.symphony_qg
        after = reopened.query(query, topk=5)
        assert after[0].id == "142"
        assert {doc.id for doc in before} == {doc.id for doc in after}
    finally:
        reopened.destroy()


@pytest.mark.parametrize("enable_mmap", [True, False])
@pytest.mark.parametrize("use_contiguous_memory", [True, False])
def test_symphony_qg_cosine_optimize_reopen(
    tmp_path, enable_mmap, use_contiguous_memory
):
    dimension = 128
    schema = zvec.CollectionSchema(
        name="symphony_cosine",
        fields=[FieldSchema("eligible", DataType.BOOL)],
        vectors=[
            VectorSchema(
                "embedding",
                DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=HnswIndexParam(
                    metric_type=MetricType.COSINE,
                    symphony_qg=True,
                    m=17,
                    use_contiguous_memory=use_contiguous_memory,
                ),
            )
        ],
    )
    path = str(tmp_path / "cosine")
    option = CollectionOption(enable_mmap=enable_mmap)
    collection = zvec.create_and_open(path=path, schema=schema, option=option)
    vectors = {}
    for i in range(1200):
        angle = 2 * math.pi * i / 1200
        scale = 0.25 + i % 53
        vectors[str(i)] = [scale * math.cos(angle), scale * math.sin(angle)] + [0.0] * (
            dimension - 2
        )
    vectors["zero"] = [0.0] * dimension
    docs = [
        Doc(id=id, fields={"eligible": id != "142"}, vectors={"embedding": vector})
        for id, vector in vectors.items()
    ]
    for offset in range(0, len(docs), 512):
        statuses = collection.insert(docs[offset : offset + 512])
        assert all(status.ok() for status in statuses)
    collection.optimize()
    query_vector = [value * 7 for value in vectors["142"]]
    query = Query(
        field_name="embedding", vector=query_vector, param=HnswQueryParam(ef=200)
    )

    def verify(current):
        hits = current.query(query, topk=5, include_vector=True)
        assert hits[0].id == "142"
        for hit in hits:
            vector = vectors[hit.id]
            assert hit.vectors["embedding"] == pytest.approx(vector, rel=2e-6, abs=1e-6)
            norm = math.sqrt(sum(x * x for x in vector))
            query_norm = math.sqrt(sum(x * x for x in query_vector))
            expected = (
                1
                - sum(x * y for x, y in zip(vector, query_vector)) / (norm * query_norm)
                if norm
                else 1.0
            )
            assert hit.score == pytest.approx(expected, abs=2e-6)
        filtered = current.query(query, topk=5, filter="eligible = true")
        assert len(filtered) == 5
        assert all(hit.id != "142" for hit in filtered)
        zero_hits = current.query(
            Query(
                field_name="embedding",
                vector=[0.0] * dimension,
                param=HnswQueryParam(ef=200),
            ),
            topk=5,
        )
        assert len(zero_hits) == 5
        assert all(hit.score == pytest.approx(1.0, abs=1e-6) for hit in zero_hits)
        return {hit.id for hit in hits}

    try:
        verify(collection)
        # Leave a mutable block beside the optimized cosine index.
        vectors["later"] = [0.0, -20.0] + [0.0] * (dimension - 2)
        assert collection.insert(
            Doc(
                id="later",
                fields={"eligible": True},
                vectors={"embedding": vectors["later"]},
            )
        ).ok()
        before = verify(collection)
        collection.close()
        collection = None
        collection = zvec.open(path=path, option=option)
        params = collection.schema.vector("embedding").index_param
        assert params.symphony_qg
        assert params.metric_type == MetricType.COSINE
        assert verify(collection) == before
    finally:
        if collection is not None:
            collection.destroy()
