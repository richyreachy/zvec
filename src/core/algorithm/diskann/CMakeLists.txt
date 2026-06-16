include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

# DiskAnn is now statically linked like other index algorithms (HNSW, IVF, etc.)
# No libaio dependency — all I/O uses synchronous pread.

cc_library(
    NAME core_knn_diskann
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS core_framework core_utility
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    VERSION "${PROXIMA_ZVEC_VERSION}"
)
