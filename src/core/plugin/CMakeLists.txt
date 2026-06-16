##
##  Copyright (C) The Software Authors. All rights reserved.
##
##  \file     CMakeLists.txt
##  \brief    Build script for the zvec internal plugin-loading glue library.
##            DiskAnn is now statically linked; this provides no-op stubs for
##            backward compatibility with call-sites that reference the
##            load/unload APIs.
##

include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

set(CORE_PLUGIN_LIBS zvec_ailego core_framework)

cc_library(
    NAME core_plugin
    STATIC STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS ${CORE_PLUGIN_LIBS}
    INCS . ${PROJECT_ROOT_DIR}/src/core
    VERSION "${PROXIMA_ZVEC_VERSION}"
)
