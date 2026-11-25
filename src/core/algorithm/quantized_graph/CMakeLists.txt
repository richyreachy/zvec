# Copyright 2026-present the zvec project
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

if(ZVEC_QUANTIZED_GRAPH_SUPPORTED)
  set(qg_native_libs zvec_turbo rabitqlib)
  set(qg_sources quantized_graph_builder.cc quantized_graph_searcher.cc quantized_graph_streamer.cc)
else()
  set(qg_sources quantized_graph_stub.cc)
endif()
list(TRANSFORM qg_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

cc_library(
    NAME core_quantized_graph_builder STATIC STRICT ALWAYS_LINK
    SRCS ${qg_sources}
    LIBS core_framework zvec_ailego ${qg_native_libs}
    INCS ${PROJECT_ROOT_DIR}/src/core
)

add_library(core_quantized_graph INTERFACE)
target_include_directories(core_quantized_graph INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_ROOT_DIR}/src/core)
target_link_libraries(core_quantized_graph INTERFACE core_quantized_graph_builder)
