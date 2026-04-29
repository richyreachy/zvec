include(${CMAKE_SOURCE_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_SRCS *.cc *.c)

set(CORE_KNN_DISKANN_LIBS zvec_ailego core_framework core_knn_cluster)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386")
    list(APPEND CORE_KNN_DISKANN_LIBS aio)
endif()

cc_library(
    NAME core_knn_diskann
    SHARED STRICT
    SRCS *.cc
    LIBS ${CORE_KNN_DISKANN_LIBS}
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    VERSION "${PROXIMA_ZVEC_VERSION}"
)

# Rename the artifact to libzvec_diskann_plugin.so so it is clearly identified
# as an optional plugin that users load at runtime via zvec::LoadDiskAnnPlugin().
set_target_properties(core_knn_diskann PROPERTIES
    OUTPUT_NAME zvec_diskann_plugin
)