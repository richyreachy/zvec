
include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  set(QG_TEST_LIBS)
  if(CC_TARGET STREQUAL "quantized_graph_index_test")
    list(APPEND QG_TEST_LIBS core_quantized_graph)
  endif()

  cc_gtest(
    NAME ${CC_TARGET} STRICT
    LIBS zvec_ailego core_framework core_utility ${QG_TEST_LIBS}
    SRCS ${CC_SRCS}
    INCS . ${PROJECT_ROOT_DIR}/src/core/
  )
  cc_test_suite(zvec_ailego ${CC_TARGET})
endforeach()
