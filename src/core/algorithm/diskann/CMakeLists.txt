include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_SRCS *.cc *.c)

# DiskAnn plugin build configuration.
#
# On Linux the plugin is loaded at runtime via dlopen(RTLD_GLOBAL), so its
# undefined references to internal zvec symbols are resolved at load time
# against the hosting binary. On Windows the plugin DLL is loaded via
# LoadLibrary and must be linked against the zvec core import libraries at
# build time because Windows has no RTLD_GLOBAL equivalent.
set(CORE_KNN_DISKANN_LIBS "")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386")
    list(APPEND CORE_KNN_DISKANN_LIBS aio)
endif()

cc_library(
    NAME core_knn_diskann
    SHARED STRICT
    SRCS *.cc
    LIBS ${CORE_KNN_DISKANN_LIBS}
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm
    VERSION "${PROXIMA_ZVEC_VERSION}"
)

# Expose internal zvec lib include dirs. On Linux, symbol resolution happens
# at dlopen time via RTLD_GLOBAL so we do NOT add link dependencies. On
# Windows, we must link against the import libraries because Windows has no
# equivalent of RTLD_GLOBAL.
foreach(_dep zvec_ailego core_framework core_knn_cluster)
    if(TARGET ${_dep})
        target_include_directories(core_knn_diskann PRIVATE
            $<TARGET_PROPERTY:${_dep},INTERFACE_INCLUDE_DIRECTORIES>)
        add_dependencies(core_knn_diskann ${_dep})
        if(MSVC)
            target_link_libraries(core_knn_diskann PRIVATE ${_dep})
        endif()
    endif()
endforeach()

# On Linux, allow the plugin to have unresolved symbols that will be satisfied
# at dlopen(RTLD_NOW | RTLD_GLOBAL) time by the hosting binary.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(core_knn_diskann PRIVATE
        "LINKER:--unresolved-symbols=ignore-all")
endif()

# Rename the artifact so it is clearly identified as an optional plugin that
# users load at runtime via zvec::LoadDiskAnnPlugin().
# On Linux: libzvec_diskann_plugin.so
# On Windows: zvec_diskann_plugin.dll
set_target_properties(core_knn_diskann PROPERTIES
    OUTPUT_NAME zvec_diskann_plugin
)