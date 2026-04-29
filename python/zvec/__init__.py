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

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.metadata import PackageNotFoundError


# ------------------------------------------------------------------------
# Load the C extension _zvec with RTLD_GLOBAL on Linux so that symbols
# defined in _zvec.so (notably the STT_GNU_UNIQUE template-singletons such
# as zvec::ailego::Factory<IndexBuilder>::Instance()::factory) participate
# in the process's global unique-symbol table from the very first load.
#
# The optional DiskAnn runtime plugin (libzvec_diskann_plugin.so) is dlopen'd
# later with RTLD_GLOBAL and relies on its INDEX_FACTORY_REGISTER_* static
# initializers writing into THE SAME Factory singletons that _zvec.so reads
# from at CreateBuilder() time. Python's default is RTLD_LOCAL, which would
# place _zvec.so's unique symbols in a private namespace — retroactively
# promoting it via dlopen(RTLD_NOLOAD | RTLD_GLOBAL) does not reliably merge
# the unique-symbol scopes on all glibc versions, so we set the flags up
# front instead.
if sys.platform.startswith("linux"):
    import os as _os

    _zvec_prev_dlopen_flags = sys.getdlopenflags()
    sys.setdlopenflags(_zvec_prev_dlopen_flags | _os.RTLD_GLOBAL | _os.RTLD_NOW)
    try:
        import _zvec as _zvec
    finally:
        sys.setdlopenflags(_zvec_prev_dlopen_flags)
    del _os, _zvec_prev_dlopen_flags


# ==============================
# Public API — grouped by category
# ==============================

from . import model as model

# —— Extensions ——
from .extension import (
    BM25EmbeddingFunction,
    DefaultLocalDenseEmbedding,
    DefaultLocalReRanker,
    DefaultLocalSparseEmbedding,
    DenseEmbeddingFunction,
    OpenAIDenseEmbedding,
    OpenAIFunctionBase,
    QwenDenseEmbedding,
    QwenFunctionBase,
    QwenReRanker,
    QwenSparseEmbedding,
    ReRanker,
    RrfReRanker,
    SentenceTransformerFunctionBase,
    SparseEmbeddingFunction,
    WeightedReRanker,
)

# —— Typing ——
from .model import param as param
from .model import schema as schema

# —— Core data structures ——
from .model.collection import Collection
from .model.doc import Doc

# —— Query & index parameters ——
from .model.param import (
    AddColumnOption,
    AlterColumnOption,
    CollectionOption,
    FlatIndexParam,
    HnswIndexParam,
    HnswQueryParam,
    HnswRabitqIndexParam,
    HnswRabitqQueryParam,
    IndexOption,
    InvertIndexParam,
    IVFIndexParam,
    IVFQueryParam,
    OptimizeOption,
)
from .model.param.vector_query import VectorQuery

# —— Schema & field definitions ——
from .model.schema import CollectionSchema, CollectionStats, FieldSchema, VectorSchema

# —— Optional runtime plugins ——
from .plugin import (
    DiskAnnPluginStatus,
    is_diskann_plugin_loaded,
    is_libaio_available,
    load_diskann_plugin,
    unload_diskann_plugin,
)

# —— tools ——
from .tool import require_module
from .typing import (
    DataType,
    IndexType,
    MetricType,
    QuantizeType,
    Status,
    StatusCode,
)
from .typing.enum import LogLevel, LogType

# —— lifecycle ——
from .zvec import create_and_open, init, open

# ==============================
# Public interface declaration
# ==============================
__all__ = [
    # Zvec functions
    "create_and_open",
    "init",
    "open",
    # Core classes
    "Collection",
    "Doc",
    # Schema
    "CollectionSchema",
    "FieldSchema",
    "VectorSchema",
    "CollectionStats",
    # Parameters
    "VectorQuery",
    "InvertIndexParam",
    "HnswIndexParam",
    "HnswRabitqIndexParam",
    "FlatIndexParam",
    "IVFIndexParam",
    "CollectionOption",
    "IndexOption",
    "OptimizeOption",
    "AddColumnOption",
    "AlterColumnOption",
    "HnswQueryParam",
    "HnswRabitqQueryParam",
    "IVFQueryParam",
    # Extensions
    "DenseEmbeddingFunction",
    "SparseEmbeddingFunction",
    "QwenFunctionBase",
    "OpenAIFunctionBase",
    "SentenceTransformerFunctionBase",
    "ReRanker",
    "DefaultLocalDenseEmbedding",
    "DefaultLocalSparseEmbedding",
    "BM25EmbeddingFunction",
    "OpenAIDenseEmbedding",
    "QwenDenseEmbedding",
    "QwenSparseEmbedding",
    "RrfReRanker",
    "WeightedReRanker",
    "DefaultLocalReRanker",
    "QwenReRanker",
    # Typing
    "DataType",
    "MetricType",
    "QuantizeType",
    "IndexType",
    "LogLevel",
    "LogType",
    "Status",
    "StatusCode",
    # Tools
    "require_module",
    # Plugins
    "DiskAnnPluginStatus",
    "is_libaio_available",
    "load_diskann_plugin",
    "is_diskann_plugin_loaded",
    "unload_diskann_plugin",
]

# ==============================
# Version handling
# ==============================
__version__: str

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # Python < 3.8

try:
    __version__ = version("zvec")
except Exception:
    __version__ = "unknown"
