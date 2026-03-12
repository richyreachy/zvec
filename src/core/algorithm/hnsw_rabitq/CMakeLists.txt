include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

foreach(FILE ${HNSW_RABITQ_FILES})
    set_source_files_properties(
        ${FILE}
        PROPERTIES
        COMPILE_FLAGS "${MATH_MARCH_FLAG_AVX512}"
    )
endforeach()

cc_library(
    NAME core_knn_hnsw_rabitq
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS core_framework rabitqlib sparsehash
    INCS . ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )