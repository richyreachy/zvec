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

if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if (HOST_ARCH MATCHES "^(x86|x64)$")
        # Exclude ARM sources on x86 builds
        file(GLOB_RECURSE ARM_SRCS
          ${CMAKE_CURRENT_SOURCE_DIR}/*/armv8/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/*/armv8/*.c
          ${CMAKE_CURRENT_SOURCE_DIR}/*/armv8/*.h)
        list(LENGTH ARM_SRCS ARM_SRCS_LEN)
        if(ARM_SRCS_LEN GREATER 0)
            list(REMOVE_ITEM ALL_SRCS ${ARM_SRCS})
        endif()

        file(GLOB_RECURSE AVX512_AVX512FP16_SRCS 
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512_fp16/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512_fp16/*.c)
        set_source_files_properties(
            ${AVX512_AVX512FP16_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX512FP16}"
        )

        # Set per-file compile flags for AVX512-VNNI sources.
        # set_source_files_properties is directory-scoped, so it must be called in the
        # same directory that adds the sources to a target (i.e. here, not in a
        # subdirectory).
        file(GLOB_RECURSE AVX512_VNNI_SRCS 
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512_vnni/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512_vnni/*.c)
        set_source_files_properties(
            ${AVX512_VNNI_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX512}"
        )

        file(GLOB_RECURSE AVX512_SRCS 
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512/*.c)
        set_source_files_properties(
            ${AVX512_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX512}"
        )
    
        file(GLOB_RECURSE AVX2_SRCS 
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx2/*.cc 
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx2/*.c
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx/*.c)
        set_source_files_properties(
            ${AVX2_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX2}"
        )

        file(GLOB_RECURSE SSE_SRCS 
          ${CMAKE_CURRENT_SOURCE_DIR}/sse/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/sse/*.c)
        set_source_files_properties(
            ${SSE_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_SSE}"
        )
    elseif (HOST_ARCH MATCHES "^(arm|arm64)$")
        # Exclude x86 sources on ARM builds
        file(GLOB_RECURSE X86_SRCS
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx/*.c
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx/*.h
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx2/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx2/*.c
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx2/*.h
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512/*.c
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512/*.h
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512_fp16/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512_fp16/*.c
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512_fp16/*.h
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512_vnni/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512_vnni/*.c
          ${CMAKE_CURRENT_SOURCE_DIR}/*/avx512_vnni/*.h
          ${CMAKE_CURRENT_SOURCE_DIR}/sse/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/sse/*.c
          ${CMAKE_CURRENT_SOURCE_DIR}/sse/*.h)
        list(LENGTH X86_SRCS X86_SRCS_LEN)
        if(X86_SRCS_LEN GREATER 0)
            list(REMOVE_ITEM ALL_SRCS ${X86_SRCS})
        endif()

        set(TURBO_MARCH_FLAG_NEON "-march=armv8-a")

        file(GLOB_RECURSE NEON_SRCS
          ${CMAKE_CURRENT_SOURCE_DIR}/*/armv8/*.cc
          ${CMAKE_CURRENT_SOURCE_DIR}/*/armv8/*.c
        )

        set_source_files_properties(
            ${NEON_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_NEON}"
        )
    endif()
endif()

cc_library(
    NAME zvec_turbo STATIC STRICT PACKED
    SRCS ${ALL_SRCS}
    LIBS zvec_ailego
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/distance ${PROJECT_ROOT_DIR}/src/include
)
