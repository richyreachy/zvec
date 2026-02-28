include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE)
    find_library(LIB_RT NAMES rt)
else()
    set(LIB_RT "")
endif()

git_version(GIT_SRCS_VER ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE ALL_SRCS *.cc *.c *.h)

set(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

if(UNIX AND NOT APPLE)
    list(APPEND EXTRA_LIBS ${LIB_RT})
endif()

set(CMAKE_CXX_FLAGS "-march=broadwell")
set(CMAKE_C_FLAGS "-march=broadwell")

set(MATH_MARCH_FLAG_AVX2 "-march=core-avx2")
set(MATH_MARCH_FLAG_AVX512 "-march=sapphirerapids")
# set(MATH_MARCH_FLAG_NEON "-march=armv8-a")

file(GLOB_RECURSE MATH_FILES_AVX2
    ${CMAKE_CURRENT_SOURCE_DIR}/math/*_avx2.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/*_avx2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_avx2.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_avx2.c
)

file(GLOB_RECURSE MATH_FILES_AVX512
    ${CMAKE_CURRENT_SOURCE_DIR}/math/*_dispatch.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/*_dispatch.c
    ${CMAKE_CURRENT_SOURCE_DIR}/math/*_avx512.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math/*_avx512.c
    ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_avx512.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_avx512.c
)

# file(GLOB_RECURSE MATH_FILES_NEON
#     ${CMAKE_CURRENT_SOURCE_DIR}/math/*_neon.cc
#     ${CMAKE_CURRENT_SOURCE_DIR}/math/*_neon.c
#     ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_neon.cc
#     ${CMAKE_CURRENT_SOURCE_DIR}/math_batch/*_neon.c
# )

foreach(MATH_FILE ${MATH_FILES_AVX2})
    set_source_files_properties(
        ${MATH_FILE}
        PROPERTIES
        COMPILE_FLAGS "${MATH_MARCH_FLAG_AVX2}"
    )
endforeach()

foreach(MATH_FILE ${MATH_FILES_AVX512})
    set_source_files_properties(
        ${MATH_FILE}
        PROPERTIES
        COMPILE_FLAGS "${MATH_MARCH_FLAG_AVX512}"
    )
endforeach()

# foreach(MATH_FILE ${MATH_FILES_NEON})
#     set_source_files_properties(
#         ${MATH_FILE}
#         PROPERTIES
#         COMPILE_FLAGS "${MATH_MARCH_FLAG_NEON}"
#     )
# endforeach()

cc_library(
    NAME zvec_ailego STATIC STRICT PACKED
    SRCS    ${ALL_SRCS}
    LIBS    ${EXTRA_LIBS}
            Arrow::arrow_static
            Arrow::parquet_static
    VERSION "${GIT_SRCS_VER}"
)
