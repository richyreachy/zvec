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

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_binary(
    NAME txt2vecs
    STRICT PACKED
    SRCS txt2vecs.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags core_framework zvec_ailego
  )

cc_binary(
    NAME local_builder
    STRICT PACKED
    SRCS local_builder.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf core_interface core_knn_diskann core_quantized_graph_builder
  )

cc_binary(
    NAME recall
    STRICT PACKED
    SRCS recall.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface core_knn_diskann
  )

cc_binary(
    NAME bench
    STRICT PACKED
    SRCS bench.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface core_knn_diskann rabitqlib zvec_turbo core_quantized_graph_builder
)

# Historical runners build this target and invoke qg_bench directly.
add_custom_target(qg_bench ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:bench>"
            "$<TARGET_FILE_DIR:bench>/qg_bench${CMAKE_EXECUTABLE_SUFFIX}"
    DEPENDS bench
    VERBATIM
)
install(PROGRAMS "$<TARGET_FILE:bench>" DESTINATION "${CMAKE_INSTALL_BINDIR}"
        RENAME "qg_bench${CMAKE_EXECUTABLE_SUFFIX}")

cc_binary(
    NAME recall_original
    STRICT PACKED
    SRCS recall_original.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface core_knn_diskann
)

cc_binary(
    NAME bench_original
    STRICT PACKED
    SRCS bench_original.cc
    INCS ${PROJECT_ROOT_DIR}/src/core/
    LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf roaring core_interface core_knn_diskann
)

cc_binary(
        NAME local_builder_original
        STRICT PACKED
        SRCS local_builder_original.cc
        INCS ${PROJECT_ROOT_DIR}/src/core/
        LIBS gflags yaml-cpp magic_enum core_framework core_metric core_quantizer core_utility core_knn_flat core_knn_flat_sparse core_knn_hnsw core_knn_hnsw_sparse core_knn_hnsw_rabitq core_knn_ivf_rabitq core_knn_vamana core_knn_cluster core_knn_ivf core_interface core_knn_diskann
)
