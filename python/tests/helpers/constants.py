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
Test constants and configuration values.
"""
from __future__ import annotations

from zvec import DataType

# =============================================================================
# Schema Field Names
# =============================================================================

SCALAR_FIELD_NAMES = {
    DataType.BOOL: "bool_field",
    DataType.FLOAT: "float_field",
    DataType.DOUBLE: "double_field",
    DataType.INT32: "int32_field",
    DataType.INT64: "int64_field",
    DataType.UINT32: "uint32_field",
    DataType.UINT64: "uint64_field",
    DataType.STRING: "string_field",
    DataType.ARRAY_BOOL: "array_bool_field",
    DataType.ARRAY_FLOAT: "array_float_field",
    DataType.ARRAY_DOUBLE: "array_double_field",
    DataType.ARRAY_INT32: "array_int32_field",
    DataType.ARRAY_INT64: "array_int64_field",
    DataType.ARRAY_UINT32: "array_uint32_field",
    DataType.ARRAY_UINT64: "array_uint64_field",
    DataType.ARRAY_STRING: "array_string_field",
}

VECTOR_FIELD_NAMES = {
    DataType.VECTOR_FP16: "vector_fp16_field",
    DataType.VECTOR_FP32: "vector_fp32_field",
    DataType.VECTOR_INT8: "vector_int8_field",
    DataType.SPARSE_VECTOR_FP32: "sparse_vector_fp32_field",
    DataType.SPARSE_VECTOR_FP16: "sparse_vector_fp16_field",
}


# =============================================================================
# Vector Dimensions
# =============================================================================

DEFAULT_VECTOR_DIMENSION = 128
MIN_VECTOR_DIMENSION = 1
MAX_VECTOR_DIMENSION = 20000
MAX_SPARSE_VECTOR_DIMENSION = 4096


# =============================================================================
# Schema Limits
# =============================================================================

COLLECTION_NAME_MAX_LENGTH = 64
FIELD_LIST_MAX_LENGTH = 1024
VECTOR_LIST_MAX_LENGTH = 5
FIELD_NAME_MAX_LENGTH = 64


# =============================================================================
# Error Messages
# =============================================================================

ERROR_MESSAGES = {
    "schema_validate": "schema validate failed",
    "create_read_only": "Unable to create collection with read-only mode",
    "incompatible_constructor": "incompatible constructor arguments",
    "incompatible_function": "incompatible function arguments",
    "invalid_path": "path validate failed",
    "index_nonexistent": "not found in schema",
    "access_destroyed": "is already destroyed",
    "collection_not_exist": "not exist",
    "not_support_add_column": "Only support basic numeric data type",
    "column_not_exists": "Column not exists",
}


# =============================================================================
# Supported Types
# =============================================================================

SUPPORTED_SCALAR_TYPES = [
    DataType.BOOL,
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.INT32,
    DataType.INT64,
    DataType.UINT32,
    DataType.UINT64,
    DataType.STRING,
    DataType.ARRAY_BOOL,
    DataType.ARRAY_FLOAT,
    DataType.ARRAY_DOUBLE,
    DataType.ARRAY_INT32,
    DataType.ARRAY_INT64,
    DataType.ARRAY_UINT32,
    DataType.ARRAY_UINT64,
    DataType.ARRAY_STRING,
]

SUPPORTED_VECTOR_TYPES = [
    DataType.VECTOR_FP16,
    DataType.VECTOR_FP32,
    DataType.VECTOR_INT8,
    DataType.SPARSE_VECTOR_FP32,
    DataType.SPARSE_VECTOR_FP16,
]

SUPPORTED_ADD_COLUMN_TYPES = [
    DataType.INT32,
    DataType.UINT32,
    DataType.INT64,
    DataType.UINT64,
    DataType.FLOAT,
    DataType.DOUBLE,
]
