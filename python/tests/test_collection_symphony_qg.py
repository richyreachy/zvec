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
    assert all(status.ok() for status in collection.insert(docs))
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
