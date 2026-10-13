include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

# Only zvec-specific training, serialization, factory integration and the
# compatibility facade live here. ISA kernels come from turbo::turbo.
file(GLOB_RECURSE ALL_SRCS CONFIGURE_DEPENDS
    quantizer/*.cc quantizer/*.h preprocessor/*.cc preprocessor/*.h)
list(APPEND ALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/turbo.cc")

set(ZVEC_TURBO_LIBRARY_OPTIONS)
if(WIN32 AND (BUILD_ZVEC_CORE_SHARED OR BUILD_ZVEC_SHARED))
  list(APPEND ZVEC_TURBO_LIBRARY_OPTIONS OBJECTS)
  if(BUILD_ZVEC_CORE_SHARED)
    list(APPEND ZVEC_TURBO_LIBRARY_OPTIONS
      EXPORT_DEF ZVEC_TURBO_BUILD_SHARED
    )
  endif()
endif()

cc_library(
    NAME zvec_turbo STATIC STRICT ALWAYS_LINK PACKED
    ${ZVEC_TURBO_LIBRARY_OPTIONS}
    SRCS ${ALL_SRCS}
    LIBS zvec_ailego turbo::turbo
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_ROOT_DIR}/src/include
)

# OPQ training remains a zvec adapter. Prefer a standalone Eigen installation;
# fall back to the existing vendored headers for regular zvec builds.
find_package(Eigen3 3.3 QUIET CONFIG)
set(ZVEC_TURBO_EIGEN_INCLUDE_DIR
    "${PROJECT_ROOT_DIR}/thirdparty/RaBitQ-Library/RaBitQ-Library-0.3.8/include/rabitqlib/third"
    CACHE PATH "Eigen include directory when Eigen3 is not installed")
foreach(_turbo_tgt zvec_turbo zvec_turbo_objects zvec_turbo_export_objects)
  if(TARGET ${_turbo_tgt})
    if(TARGET Eigen3::Eigen)
      target_include_directories(${_turbo_tgt} SYSTEM PRIVATE
          $<TARGET_PROPERTY:Eigen3::Eigen,INTERFACE_INCLUDE_DIRECTORIES>)
    else()
      if(NOT EXISTS "${ZVEC_TURBO_EIGEN_INCLUDE_DIR}/Eigen/Dense")
        message(FATAL_ERROR "OPQ training requires Eigen3: install Eigen3, initialize the RaBitQ submodule, or set ZVEC_TURBO_EIGEN_INCLUDE_DIR")
      endif()
      target_include_directories(${_turbo_tgt} SYSTEM PRIVATE "${ZVEC_TURBO_EIGEN_INCLUDE_DIR}")
    endif()
  endif()
endforeach()
