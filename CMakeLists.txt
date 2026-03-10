cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0077 NEW)
project(zvec VERSION 0.3.0)
set(CC_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=return-type")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror=return-type")

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

option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)
message(STATUS "BUILD_PYTHON_BINDINGS:${BUILD_PYTHON_BINDINGS}")

option(BUILD_TOOLS "Build tools" ON)
message(STATUS "BUILD_TOOLS:${BUILD_TOOLS}")

option(USE_OSS_MIRROR "Use OSS mirror for faster third-party downloads" ON)
message(STATUS "USE_OSS_MIRROR:${USE_OSS_MIRROR}")

option(BUILD_EXAMPLES "Build examples" ON)
message(STATUS "BUILD_EXAMPLES:${BUILD_EXAMPLES}")

# Option to build static libraries for RPM packaging
option(BUILD_STATIC_LIBS "Build static libraries" OFF)
message(STATUS "BUILD_STATIC_LIBS:${BUILD_STATIC_LIBS}")

# Option to enable RPM packaging
option(ENABLE_RPM_PACKAGING "Enable RPM packaging support" OFF)
message(STATUS "ENABLE_RPM_PACKAGING:${ENABLE_RPM_PACKAGING}")

# Option to build fat libraries (all-in-one with static-linked third-party deps)
# When ON: libzvec_c_api.so and libzvec_c_api_static.a include all third-party libs
# When OFF: Libraries depend on external .so files for third-party deps
option(BUILD_FAT_LIBS "Build fat libraries with all dependencies statically linked" ON)
message(STATUS "BUILD_FAT_LIBS:${BUILD_FAT_LIBS}")

cc_directory(thirdparty)
cc_directories(src)
cc_directories(tests)
if(BUILD_EXAMPLES)
    cc_directories(examples)
endif()

if(BUILD_TOOLS)
    cc_directories(tools)
endif()

git_version(GIT_SRCS_VER ${PROJECT_ROOT_DIR})
set(CPACK_PACKAGE_VERSION ${GIT_SRCS_VER})
set(CPACK_PACKAGE_NAME zvec)

# =============================================================================
# RPM Packaging Configuration
# =============================================================================
if(ENABLE_RPM_PACKAGING OR BUILD_STATIC_LIBS)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Generate version helpers
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/zvecConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    # Configure CMake config file
    configure_package_config_file(
        "${PROJECT_ROOT_DIR}/cmake/zvec_config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/zvecConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/zvec"
    )

    # Configure pkg-config file
    configure_file(
        "${PROJECT_ROOT_DIR}/cmake/zvec.pc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/zvec.pc"
        @ONLY
    )

    # Install CMake configuration files
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/zvecConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/zvecConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/zvec"
    )

    # Install pkg-config file
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/zvec.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
    )

    # Install documentation
    install(FILES
        "${PROJECT_ROOT_DIR}/README.md"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/zvec"
    )

    # Create a simple CMake targets file
    # Since we can't export complex target hierarchies with all dependencies,
    # we create a simplified import file
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/zvecTargets.cmake"
"# Zvec CMake Targets File

# Import target: zvec::zvec_c_api
if(NOT TARGET zvec::zvec_c_api)
    add_library(zvec::zvec_c_api SHARED IMPORTED)
    set_target_properties(zvec::zvec_c_api PROPERTIES
        IMPORTED_LOCATION \"\${CMAKE_CURRENT_LIST_DIR}/../../../${CMAKE_INSTALL_LIBDIR}/libzvec_c_api.so\"
        INTERFACE_INCLUDE_DIRECTORIES \"\${CMAKE_CURRENT_LIST_DIR}/../../../${CMAKE_INSTALL_INCLUDEDIR}\"
        INTERFACE_LINK_LIBRARIES \"Threads::Threads\"
    )
endif()

# Import target: zvec::zvec_c_api_static (if available)
# This is a 'fat' static library that includes all zvec code (core, db, ailego)
# But you still need to link third-party dependencies separately
if(BUILD_STATIC_LIBS AND EXISTS \"\${CMAKE_CURRENT_LIST_DIR}/../../../${CMAKE_INSTALL_LIBDIR}/libzvec_c_api_static.a\")
    if(NOT TARGET zvec::zvec_c_api_static)
        add_library(zvec::zvec_c_api_static STATIC IMPORTED)
        set_target_properties(zvec::zvec_c_api_static PROPERTIES
            IMPORTED_LOCATION \"\${CMAKE_CURRENT_LIST_DIR}/../../../${CMAKE_INSTALL_LIBDIR}/libzvec_c_api_static.a\"
            INTERFACE_INCLUDE_DIRECTORIES \"\${CMAKE_CURRENT_LIST_DIR}/../../../${CMAKE_INSTALL_INCLUDEDIR}\"
            # Note: Third-party dependencies (Arrow, RocksDB, etc.) must be linked by the user
            INTERFACE_LINK_LIBRARIES \"Threads::Threads\"
        )
    endif()
endif()
")

    # Install the targets file
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/zvecTargets.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/zvec"
        COMPONENT devel
    )

    # Include RPM-specific installation rules
    include(${PROJECT_ROOT_DIR}/cmake/rpm_install.cmake)
endif()

# Configure CPack for RPM generation
if(ENABLE_RPM_PACKAGING)
    set(CPACK_GENERATOR "RPM" CACHE STRING "Generators to support")
    set(CPACK_PACKAGE_VERSION ${GIT_SRCS_VER})
    set(CPACK_PACKAGE_NAME zvec)
    set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
    set(CPACK_RPM_PACKAGE_URL "https://github.com/alibaba/zvec")
    set(CPACK_RPM_PACKAGE_DESCRIPTION "Zvec - High-performance in-process vector database library with self-contained shared library")
    set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
    set(CPACK_RPM_PACKAGE_VENDOR "Alibaba")
    set(CPACK_RPM_PACKAGE_SUMMARY "High-performance in-process vector database library")

    # RPM specific settings
    set(CPACK_PACKAGE_FILE_NAME "zvec-${CPACK_PACKAGE_VERSION}-1")

    # Single RPM package - no components
    # Only the self-contained shared library (fat lib)
    set(CPACK_RPM_COMPONENT_INSTALL OFF)

    include(CPack)
endif()

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