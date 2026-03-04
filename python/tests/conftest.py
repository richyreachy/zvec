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
"""
Pytest configuration and shared fixtures for zvec tests.

This file is automatically discovered by pytest and provides fixtures
to all test modules without requiring explicit imports.
"""
from __future__ import annotations

import logging
import pytest
from typing import Any, Generator

import zvec
from zvec import (
    CollectionOption,
    InvertIndexParam,
    HnswIndexParam,
    IVFIndexParam,
    FieldSchema,
    VectorSchema,
    CollectionSchema,
    Collection,
)

from tests.helpers.constants import (
    SCALAR_FIELD_NAMES,
    VECTOR_FIELD_NAMES,
    DEFAULT_VECTOR_DIMENSION,
)


# =============================================================================
# Session-scoped Fixtures (created once per test session)
# =============================================================================


@pytest.fixture(scope="session")
def basic_schema() -> CollectionSchema:
    """
    Create a basic collection schema with common field types.

    Fields: id (INT64), name (STRING), weight (FLOAT)
    Vectors: dense (VECTOR_FP32, dim=128), sparse (SPARSE_VECTOR_FP32)
    """
    return CollectionSchema(
        name="test_collection",
        fields=[
            FieldSchema(
                "id",
                zvec.DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            FieldSchema(
                "name", zvec.DataType.STRING, nullable=False, index_param=InvertIndexParam()
            ),
            FieldSchema("weight", zvec.DataType.FLOAT, nullable=True),
        ],
        vectors=[
            VectorSchema(
                "dense",
                zvec.DataType.VECTOR_FP32,
                dimension=128,
                index_param=HnswIndexParam(),
            ),
            VectorSchema(
                "sparse", zvec.DataType.SPARSE_VECTOR_FP32, index_param=HnswIndexParam()
            ),
        ],
    )


@pytest.fixture(scope="session")
def full_schema() -> CollectionSchema:
    """
    Create a full schema with all supported scalar and vector field types.
    """
    fields = [
        FieldSchema(name, dtype, nullable=False)
        for dtype, name in SCALAR_FIELD_NAMES.items()
    ]
    vectors = [
        VectorSchema(name, dtype, dimension=DEFAULT_VECTOR_DIMENSION)
        for dtype, name in VECTOR_FIELD_NAMES.items()
    ]

    return CollectionSchema(
        name="full_collection",
        fields=fields,
        vectors=vectors,
    )


# =============================================================================
# Function-scoped Fixtures (created per test function)
# =============================================================================


@pytest.fixture(scope="function")
def collection_option() -> CollectionOption:
    """Default collection option configuration."""
    return CollectionOption(read_only=False, enable_mmap=True)


@pytest.fixture(scope="function")
def collection_temp_dir(tmp_path_factory) -> str:
    """
    Create a temporary directory for collection storage.

    Uses pytest's tmp_path_factory to ensure isolation between tests.
    """
    temp_dir = tmp_path_factory.mktemp("zvec")
    return str(temp_dir / "test_collection")


# =============================================================================
# Collection Management Fixtures
# =============================================================================


def _create_collection(
    path: str,
    schema: CollectionSchema,
    option: CollectionOption,
) -> Generator[Collection, None, None]:
    """Helper to create and manage collection lifecycle."""
    coll = zvec.create_and_open(path=path, schema=schema, option=option)

    assert coll is not None
    assert coll.path == path
    assert coll.schema.name == schema.name

    try:
        yield coll
    finally:
        if hasattr(coll, "destroy") and coll is not None:
            try:
                coll.destroy()
            except Exception as e:
                logging.warning(f"Failed to destroy collection: {e}")


@pytest.fixture(scope="function")
def basic_collection(
    collection_temp_dir: str,
    basic_schema: CollectionSchema,
    collection_option: CollectionOption,
) -> Generator[Collection, None, None]:
    """Create a collection with basic schema."""
    yield from _create_collection(
        collection_temp_dir, basic_schema, collection_option
    )


@pytest.fixture(scope="function")
def full_collection(
    collection_temp_dir: str,
    full_schema: CollectionSchema,
    collection_option: CollectionOption,
) -> Generator[Collection, None, None]:
    """Create a collection with full schema."""
    yield from _create_collection(
        collection_temp_dir, full_schema, collection_option
    )
