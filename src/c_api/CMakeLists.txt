# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# C API library source files
set(ZVEC_C_API_SOURCES
    c_api.cc
)

# C API library header files
set(ZVEC_C_API_HEADERS
    ${PROJECT_SOURCE_DIR}/src/include/zvec/c_api.h
)

# Create shared library
add_library(zvec_c_api SHARED
    ${ZVEC_C_API_SOURCES}
    ${ZVEC_C_API_HEADERS}
)

# Set library properties
set_target_properties(zvec_c_api PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "zvec_c_api"
    POSITION_INDEPENDENT_CODE ON
)

find_package(Threads REQUIRED)

# Link dependencies based on BUILD_FAT_LIBS option
# FAT mode: statically link all third-party libs (self-contained, no external deps)
# NON-FAT mode: dynamically link third-party libs (smaller, but requires external .so files)

# Common: always statically link all zvec code into the shared library
set(ZVEC_C_API_COMMON_LINK_OPTS
    "$<TARGET_OBJECTS:zvec_ailego_objects>"
    "$<TARGET_OBJECTS:zvec_core_objects>"
    "$<TARGET_OBJECTS:zvec_db_objects>"
)

if(BUILD_FAT_LIBS)
    # FAT mode: embed all third-party libraries
    if(APPLE)
        target_link_libraries(zvec_c_api
            PRIVATE
                # Force load all object files from zvec components
                ${ZVEC_C_API_COMMON_LINK_OPTS}
                # Statically link third-party libraries
                roaring
                Arrow::arrow_static
                Arrow::parquet_static
                Arrow::arrow_compute
                Arrow::arrow_dataset
                Arrow::arrow_acero
                rocksdb
                glog
                libprotobuf
                antlr4
                sparsehash
                magic_enum
                Threads::Threads
        )
        target_link_options(zvec_c_api PRIVATE
            LINKER:-force_load,$<TARGET_FILE:roaring>
            LINKER:-force_load,$<TARGET_FILE:Arrow::arrow_static>
            LINKER:-force_load,$<TARGET_FILE:Arrow::parquet_static>
            LINKER:-force_load,$<TARGET_FILE:Arrow::arrow_compute>
            LINKER:-force_load,$<TARGET_FILE:Arrow::arrow_dataset>
            LINKER:-force_load,$<TARGET_FILE:Arrow::arrow_acero>
            LINKER:-force_load,$<TARGET_FILE:rocksdb>
            LINKER:-force_load,$<TARGET_FILE:glog>
            LINKER:-force_load,$<TARGET_FILE:libprotobuf>
            LINKER:-force_load,$<TARGET_FILE:antlr4>
        )
    else()
        target_link_libraries(zvec_c_api
            PRIVATE
                "-Wl,--whole-archive" ${ZVEC_C_API_COMMON_LINK_OPTS}
                # Statically link third-party libraries
                "-Wl,--whole-archive"
                    roaring
                    Arrow::arrow_static
                    Arrow::parquet_static
                    Arrow::arrow_compute
                    Arrow::arrow_dataset
                    Arrow::arrow_acero
                    rocksdb
                    glog
                    libprotobuf
                    antlr4
                    sparsehash
                    magic_enum
                "-Wl,--no-whole-archive"
                Threads::Threads
                ${CMAKE_DL_LIBS}
        )
    endif()
else()
    # NON-FAT mode: link zvec_db and zvec_ailego to get third-party dependencies dynamically
    if(APPLE)
        set(FORCE_LOAD_FLAGS "-Wl,-force_load")
        target_link_libraries(zvec_c_api
            PRIVATE
                ${FORCE_LOAD_FLAGS} ${ZVEC_C_API_COMMON_LINK_OPTS}
                # Link targets to get third-party dependencies
                ${FORCE_LOAD_FLAGS} zvec_db
                ${FORCE_LOAD_FLAGS} zvec_ailego
                Threads::Threads
        )
    else()
        target_link_libraries(zvec_c_api
            PRIVATE
                "-Wl,--whole-archive" ${ZVEC_C_API_COMMON_LINK_OPTS}
                "-Wl,--no-whole-archive"
                # Link targets to get third-party dependencies
                zvec_db
                zvec_ailego
                Threads::Threads
        )
    endif()
endif()

# Include directories
target_include_directories(zvec_c_api
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

# Compile options
target_compile_options(zvec_c_api PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
)

# Installation rules for shared library
install(TARGETS zvec_c_api
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Installation rules for headers
install(FILES ${PROJECT_SOURCE_DIR}/src/include/zvec/c_api.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zvec
)

# Create static library version (optional)
# Note: In FAT mode (BUILD_FAT_LIBS=ON), we only build shared library because
# creating a self-contained fat static library is complex and error-prone.
# Users who need static linking should use the shared library or link third-party libs manually.
if(BUILD_STATIC_LIBS AND NOT BUILD_FAT_LIBS)
    # NON-FAT mode only: Create static library without third-party dependencies
    # User needs to link third-party libraries manually

    # Collect all object files from zvec dependencies
    set(ZVEC_C_API_STATIC_DEPS
        $<TARGET_OBJECTS:zvec_core_objects>
        $<TARGET_OBJECTS:zvec_db_objects>
        $<TARGET_OBJECTS:zvec_ailego_objects>
    )

    # Create static library with zvec code only
    add_library(zvec_c_api_static STATIC
        ${ZVEC_C_API_SOURCES}
        ${ZVEC_C_API_HEADERS}
        ${ZVEC_C_API_STATIC_DEPS}
    )

    set_target_properties(zvec_c_api_static PROPERTIES
        OUTPUT_NAME "zvec_c_api_static"
        POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(zvec_c_api_static
        PUBLIC
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )

    target_include_directories(zvec_c_api_static
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
            $<INSTALL_INTERFACE:include>
    )

    # Install the static library
    install(TARGETS zvec_c_api_static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()
