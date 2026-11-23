# Graph traversal and codec adapters are header-only. Computational kernels
# are compiled once by Turbo, including its per-ISA source flags.
add_library(core_quantized_graph INTERFACE)

target_include_directories(core_quantized_graph INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_ROOT_DIR}/src/core)

target_link_libraries(core_quantized_graph INTERFACE
    core_framework zvec_ailego zvec_turbo rabitqlib)

if(ZVEC_QUANTIZED_GRAPH_SUPPORTED)
  set(qg_sources quantized_graph_builder.cc quantized_graph_searcher.cc quantized_graph_streamer.cc)
else()
  set(qg_sources quantized_graph_stub.cc)
endif()
list(TRANSFORM qg_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

cc_library(
    NAME core_quantized_graph_builder STATIC STRICT ALWAYS_LINK
    SRCS ${qg_sources}
    LIBS core_quantized_graph
    INCS ${PROJECT_ROOT_DIR}/src/core
)
