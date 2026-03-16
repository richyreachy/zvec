cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0077 NEW)
project(zvec)
set(CC_CXX_STANDARD 17)

if(MSVC)
    set(INTTYPES_FORMAT VC7)
    add_compile_options(/FS) #handle .pdb
    add_compile_options(/EHsc) # def c++ exception behavior
    add_compile_options(/Zc:preprocessor)
    add_compile_options(/we4716) # -Werror=return-type


    ###### reduce output length to make vibe coding work better :), which should be removed or solved later
    add_compile_options(/wd4267 /wd4244)
    add_compile_options(/wd4146) # unary minus operator applied to unsigned type #usage: uint32_t seg_id_{-1U};
    add_compile_options(/wd4310) # warning C4310: cast truncates constant value
    add_compile_options(/wd4530) #  warning C4530: C++ exception handler used, but unwind semantics are not enabled. Specify /EHsc
    add_compile_options(/wd4251) #protobuf
    add_compile_options(/wd4245 /wd4334 /wd4702  /wd4305 /wd4099)
    add_compile_options(/wd4200) # flexible structure

    # //warning C4334: '<<': result of 32-bit shift implicitly converted to 64 bits (was 64-bit shift intended?)
    # keeps |= (1 << k);
    add_compile_options(/wd4334)
    ######

    #TODO(windows): to be verified
    add_definitions(-DARROW_STATIC -DPARQUET_STATIC -DARROW_ACERO_STATIC -DARROW_DS_STATIC -DARROW_COMPUTE_STATIC)


else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=return-type")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror=return-type")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

if(NOT DEFINED PROJECT_ROOT_DIR OR NOT PROJECT_ROOT_DIR)
    set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Root directory of the project" FORCE)
endif()

message(STATUS "PROJECT_ROOT_DIR = ${PROJECT_ROOT_DIR}")

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

include_directories(${PROJECT_ROOT_DIR}/src/include)
include_directories(${PROJECT_ROOT_DIR}/src)
include_directories(${PROJECT_ROOT_DIR})

option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)
message(STATUS "BUILD_PYTHON_BINDINGS:${BUILD_PYTHON_BINDINGS}")

option(BUILD_TOOLS "Build tools" ON)
message(STATUS "BUILD_TOOLS:${BUILD_TOOLS}")

option(USE_OSS_MIRROR "Use OSS mirror for faster third-party downloads" OFF)
if(DEFINED ENV{USE_OSS_MIRROR} AND NOT "$ENV{USE_OSS_MIRROR}" STREQUAL "")
  set(USE_OSS_MIRROR "$ENV{USE_OSS_MIRROR}" CACHE BOOL "Use OSS mirror for faster third-party downloads" FORCE)
endif()
message(STATUS "USE_OSS_MIRROR:${USE_OSS_MIRROR}")

cc_directory(thirdparty)
cc_directories(src)
cc_directories(tests)

if(BUILD_TOOLS)
    cc_directories(tools)
endif()

git_version(GIT_SRCS_VER ${PROJECT_ROOT_DIR})
set(CPACK_PACKAGE_VERSION ${GIT_SRCS_VER})
set(CPACK_PACKAGE_NAME zvec)
include(CPack)

if(BUILD_PYTHON_BINDINGS)
    if(APPLE)
        set(CMAKE_STRIP "")
        message(STATUS "Disabled strip on macOS to preserve code signature")
    endif()

    include(GNUInstallDirs)
    if(DEFINED SKBUILD_PLATLIB_DIR)
        set(ZVEC_PY_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}")
    elseif(DEFINED Python_SITEARCH)
        set(ZVEC_PY_INSTALL_DIR "${Python_SITEARCH}")
    else()
        set(ZVEC_PY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    endif()

    message(STATUS "Zvec install path: ${ZVEC_PY_INSTALL_DIR}")
    install(TARGETS _zvec LIBRARY DESTINATION ${ZVEC_PY_INSTALL_DIR})
endif()