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

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  set(QG_TEST_LIBS)
  if(CC_TARGET STREQUAL "quantized_graph_index_test")
    list(APPEND QG_TEST_LIBS core_quantized_graph)
  elseif(CC_TARGET STREQUAL "quantized_graph_builder_test")
    list(APPEND QG_TEST_LIBS core_quantized_graph_builder core_metric)
  endif()

  cc_gtest(
    NAME ${CC_TARGET} STRICT
    LIBS zvec_ailego core_framework core_utility ${QG_TEST_LIBS}
    SRCS ${CC_SRCS}
    INCS . ${PROJECT_ROOT_DIR}/src/core/
  )
  cc_test_suite(zvec_ailego ${CC_TARGET})
endforeach()
