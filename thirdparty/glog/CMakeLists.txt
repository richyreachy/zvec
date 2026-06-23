set(BUILD_TESTING OFF CACHE BOOL "Disable Unit Tests" FORCE)
set(WITH_GFLAGS OFF CACHE BOOL "Disable find_package(gflags) in glog" FORCE)
set(WITH_UNWIND OFF CACHE BOOL "Disable find_package(unwind) in glog" FORCE)
set(HAVE_LIB_GFLAGS TRUE CACHE BOOL "")

if(NOT MSVC)
add_compile_options(-Wno-deprecated-declarations)
endif()

set(GLOG_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glog-0.5.0)
if (ANDROID)
    set(GLOG_ANDROID_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/glog.android.patch)
    apply_patch_once("glog_android_fix" "${GLOG_SRC_DIR}" "${GLOG_ANDROID_PATCH}")
else()
    set(GLOG_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/glog.patch)
    apply_patch_once("glog_fix" "${GLOG_SRC_DIR}" "${GLOG_PATCH}")
endif()

# Force glog to build as a STATIC library. glog 0.5.0 declares several
# internal helpers (e.g. google::glog_internal_namespace_::
# IsGoogleLoggingInitialized()) without GOOGLE_GLOG_DLL_DECL, so with the
# default hidden visibility they remain local symbols in a shared libglog and
# cannot be resolved by consumers like libzvec.so. Building glog static makes
# those symbols linkable.
set(_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)

set(_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
set(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${EXTERNAL_LIB_DIR})
add_subdirectory(glog-0.5.0)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
unset(_SAVED_CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
set(BUILD_SHARED_LIBS ${_SAVED_BUILD_SHARED_LIBS})
unset(_SAVED_BUILD_SHARED_LIBS)

set(BUILD_SHARED_LIBS ${_SAVED_BUILD_SHARED_LIBS})
unset(_SAVED_BUILD_SHARED_LIBS)

add_dependencies(glog gflags)

mark_target_includes_system(glog)

if(MSVC)
    # Modern MSVC (1900+/VS2015+) provides snprintf in the CRT.
    # Suppress glog's own declaration which conflicts with the SDK header.
    target_compile_definitions(glog PRIVATE HAVE_SNPRINTF)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(glog PRIVATE -Wno-sign-compare)
endif()

get_target_property(glog_BINARY_DIR glog BINARY_DIR)
get_target_property(glog_SOURCE_DIR glog SOURCE_DIR)
# get_target_property(GLOG_INCLUDE_DIRS glog INTERFACE_INCLUDE_DIRECTORIES)

set(GLOG_INCLUDE_DIRS ${glog_BINARY_DIR} ${glog_SOURCE_DIR}/src)
set(GLOG_FOUND TRUE PARENT_SCOPE)
set(GLOG_INCLUDE_DIR ${GLOG_INCLUDE_DIRS} PARENT_SCOPE)
set(GLOG_INCLUDE_DIRS ${GLOG_INCLUDE_DIRS} PARENT_SCOPE)
set(GLOG_LIBRARY $<TARGET_FILE:glog> PARENT_SCOPE)
set(GLOG_LIBRARIES $<TARGET_FILE:glog> PARENT_SCOPE)
