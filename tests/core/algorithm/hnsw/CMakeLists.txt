include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      LIBS core_quantized_graph rabitqlib zvec_ailego core_framework core_utility core_metric core_quantizer core_knn_hnsw core_knn_flat
      SRCS ${CC_SRCS}
      INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm/hnsw
    )
endforeach()
# Explicit opt-in executable; performance comparisons are not unit-test gates.
cc_binary(
    NAME pq_graph_benchmark
    SRCS pq_graph_benchmark.cc
    LIBS core_quantized_graph core_metric core_quantizer
    INCS ${PROJECT_ROOT_DIR}/src/core
)
