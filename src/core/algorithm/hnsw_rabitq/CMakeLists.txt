include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)


cc_library(
    NAME core_knn_hnsw_rabitq
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS core_framework rabitqlib sparsehash
    INCS . ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    CXXFLAGS "${MATH_MARCH_FLAG_AVX512}"
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )