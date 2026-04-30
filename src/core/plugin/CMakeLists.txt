##
##  Copyright (C) The Software Authors. All rights reserved.
##
##  \file     CMakeLists.txt
##  \brief    Build script for the zvec plugin-loading glue library.
##            This small library lives in the main zvec_core artifact and
##            provides zvec::LoadDiskAnnPlugin()/IsLibAioAvailable() so that
##            users can lazily bring in the DiskAnn implementation (which
##            ships separately as libzvec_diskann_plugin.so).
##

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

set(CORE_PLUGIN_LIBS zvec_ailego core_framework)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND CORE_PLUGIN_LIBS ${CMAKE_DL_LIBS})
endif()

cc_library(
    NAME core_plugin
    STATIC STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS ${CORE_PLUGIN_LIBS}
    INCS . ${PROJECT_ROOT_DIR}/src/core
    VERSION "${PROXIMA_ZVEC_VERSION}"
)
