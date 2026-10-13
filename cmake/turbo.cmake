# Independent Turbo kernels. zvec retains quantizer/model/index adapters only.
include_guard(GLOBAL)
set(ZVEC_TURBO_PROVIDER "FETCH" CACHE STRING "Turbo provider: FETCH, SOURCE, or PACKAGE")
set_property(CACHE ZVEC_TURBO_PROVIDER PROPERTY STRINGS FETCH SOURCE PACKAGE)
set(ZVEC_TURBO_SOURCE_DIR "" CACHE PATH "Standalone Turbo source directory (SOURCE provider)")
set(ZVEC_TURBO_GIT_TAG "63e844e7d75e8221eb54a21906ab295d1e5981cf" CACHE STRING "Pinned standalone Turbo commit")

if(NOT TARGET turbo::turbo)
  if(ZVEC_TURBO_PROVIDER STREQUAL "PACKAGE")
    find_package(Turbo 0.1.0 EXACT CONFIG REQUIRED)
  elseif(ZVEC_TURBO_PROVIDER STREQUAL "SOURCE" OR ZVEC_TURBO_PROVIDER STREQUAL "FETCH")
    # Normal variables keep these choices local; do not overwrite caller caches.
    set(TURBO_BUILD_TESTS OFF)
    set(TURBO_BUILD_EXAMPLES OFF)
    set(TURBO_ENABLE_INSTALL ON)
    set(TURBO_BUILD_SHARED OFF)
    if(ZVEC_TURBO_PROVIDER STREQUAL "SOURCE")
      if(NOT EXISTS "${ZVEC_TURBO_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "SOURCE provider requires ZVEC_TURBO_SOURCE_DIR pointing to standalone Turbo")
      endif()
      add_subdirectory("${ZVEC_TURBO_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/turbo-build")
    else()
      include(FetchContent)
      FetchContent_Declare(zvec_turbo_kernels
        GIT_REPOSITORY https://github.com/richyreachy/zvec-turbo.git
        GIT_TAG "${ZVEC_TURBO_GIT_TAG}")
      FetchContent_MakeAvailable(zvec_turbo_kernels)
    endif()
  else()
    message(FATAL_ERROR "Unknown ZVEC_TURBO_PROVIDER: ${ZVEC_TURBO_PROVIDER}")
  endif()
endif()

# Static kernels are embedded into zvec's C/C++/Python shared artifacts. This
# avoids introducing a second runtime library or exporting Turbo's private ISA
# symbols. A preinstalled package must also be built with PIC and matching CRT.
get_target_property(_zvec_turbo_type turbo::turbo TYPE)
if(NOT _zvec_turbo_type STREQUAL "STATIC_LIBRARY")
  message(FATAL_ERROR "zvec requires a static Turbo library; rebuild with TURBO_BUILD_SHARED=OFF")
endif()
message(STATUS "zvec Turbo provider: ${ZVEC_TURBO_PROVIDER}")
