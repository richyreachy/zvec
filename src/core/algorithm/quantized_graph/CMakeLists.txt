include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

# symphony_qg_kernels_avx512.cc uses AVX512 intrinsics guarded by
# __AVX512F__/__AVX512BW__ and runtime-gates on CpuFeatures. Compile it with
# an AVX512-capable -march so the intrinsics are emitted; the entry points
# return false and callers use scalar fallbacks when the flags are absent.
# Source-file properties are directory-scoped, so this must also be set here
# for the core_quantized_graph target (and in src/core/CMakeLists.txt for the
# zvec_core glob).
if(NOT ANDROID AND AUTO_DETECT_ARCH)
  if(HOST_ARCH MATCHES "^(x86|x64)$")
    setup_compiler_march_for_x86(
        _SYMQG_MARCH_SSE _SYMQG_MARCH_AVX2
        _SYMQG_MARCH_AVX512 _SYMQG_MARCH_AVX512FP16)
    if(_SYMQG_MARCH_AVX512)
      set_source_files_properties(
          ${CMAKE_CURRENT_SOURCE_DIR}/symphony_qg_kernels_avx512.cc
          PROPERTIES
          COMPILE_FLAGS "${_SYMQG_MARCH_AVX512}"
      )
    endif()
  endif()
endif()

cc_library(
    NAME core_quantized_graph
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS core_framework zvec_ailego rabitqlib
    INCS . ${PROJECT_ROOT_DIR}/src/core
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
