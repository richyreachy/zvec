include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if (HOST_ARCH MATCHES "^(x86|x64)$")
        setup_compiler_march_for_x86(TURBO_MARCH_FLAG_SSE TURBO_MARCH_FLAG_AVX2 TURBO_MARCH_FLAG_AVX512 TURBO_MARCH_FLAG_AVX512FP16)
    elseif (HOST_ARCH MATCHES "^(arm|arm64)$")
        # ARM64 architecture - no special march flags needed for now
        # NEON implementations can be added here if needed
        message(STATUS "turbo: ARM64 detected, skipping x86-specific optimizations")
    endif()
endif()

file(GLOB_RECURSE ALL_SRCS *.cc *.c *.h)

# Exclude diskann/rabit_distance_estimator files when DiskAnn is not supported.
if(NOT DISKANN_SUPPORTED)
  list(FILTER ALL_SRCS EXCLUDE REGEX ".*/diskann/.*")
endif()

# Set per-file compile flags for AVX512-VNNI sources.
# set_source_files_properties is directory-scoped, so it must be called in the
# same directory that adds the sources to a target (i.e. here, not in a
# subdirectory).
if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if (HOST_ARCH MATCHES "^(x86|x64)$")
        file(GLOB_RECURSE AVX512_VNNI_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/distance/avx512_vnni/*.cc)
        set_source_files_properties(
            ${AVX512_VNNI_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX512}"
        )
    endif()
endif()

# Set per-file RABITQ_ARCH_FLAG for rabit_quantizer sources (AVX2/AVX512).
# rabitqlib requires AVX2 or AVX-512 intrinsics; compile the rabit_quantizer
# translation unit with the appropriate -march flag.
if(RABITQ_SUPPORTED AND AUTO_DETECT_ARCH)
    file(GLOB_RECURSE RABITQ_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/quantizer/rabit_quantizer/*.cc)
    set_source_files_properties(
        ${RABITQ_SRCS}
        PROPERTIES
        COMPILE_FLAGS "${RABITQ_ARCH_FLAG}"
    )
endif()

# rabitqlib is an INTERFACE (header-only) library needed when RABITQ_SUPPORTED.
set(TURBO_LIBS zvec_ailego)
if(RABITQ_SUPPORTED)
    list(APPEND TURBO_LIBS rabitqlib)
endif()

cc_library(
    NAME zvec_turbo STATIC STRICT ALWAYS_LINK PACKED
    SRCS ${ALL_SRCS}
    LIBS ${TURBO_LIBS}
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/distance ${PROJECT_ROOT_DIR}/src/include ${PROJECT_ROOT_DIR}/src
)
