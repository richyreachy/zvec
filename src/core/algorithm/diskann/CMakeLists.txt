include(${CMAKE_SOURCE_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_SRCS *.cc *.c)

set(CORE_KNN_DISKANN_LIBS zvec_ailego core_framework core_knn_cluster)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386")
    list(APPEND CORE_KNN_DISKANN_LIBS aio)
endif()

cc_library(
    NAME core_knn_diskann 
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS ${CORE_KNN_DISKANN_LIBS} 
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    VERSION "${PROXIMA_ZVEC_VERSION}"
)