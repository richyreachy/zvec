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

# Install rules for RPM packaging (single package, no components)

include(GNUInstallDirs)

# Install all public headers for zvec
# These are the headers that C API users will need
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/include/zvec/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zvec
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install proto headers if they exist
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/proto")
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/proto/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zvec/proto
        FILES_MATCHING PATTERN "*.pb.h"
    )
endif()

# Install static library only in NON-FAT mode
# In FAT mode, we only build the self-contained shared library
if(BUILD_STATIC_LIBS AND NOT BUILD_FAT_LIBS)
    if(TARGET zvec_c_api_static)
        install(TARGETS zvec_c_api_static
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()
endif()
