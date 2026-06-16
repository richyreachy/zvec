include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

set(HNSW_TEST_LIBS zvec_ailego core_framework core_utility core_metric core_quantizer core_knn_hnsw core_knn_flat zvec_turbo)
if(RABITQ_SUPPORTED)
  list(APPEND HNSW_TEST_LIBS core_knn_hnsw_rabitq core_knn_cluster rabitqlib)
endif()

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      LIBS ${HNSW_TEST_LIBS}
      SRCS ${CC_SRCS}
      INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm/hnsw ${PROJECT_ROOT_DIR}/src/core/algorithm
    )
endforeach()