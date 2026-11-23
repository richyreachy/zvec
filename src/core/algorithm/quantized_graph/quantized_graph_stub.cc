// Copyright 2026-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include "quantized_graph_builder.h"

namespace zvec::core {

int QuantizedGraphBuilder::build(std::vector<float>, size_t, bool,
                                 const Options &, Result &) {
  return IndexError_Unsupported;
}

int QuantizedGraphBuilder::init(const IndexMeta &, const ailego::Params &) {
  return IndexError_Unsupported;
}

int QuantizedGraphBuilder::train(IndexThreads::Pointer, IndexHolder::Pointer) {
  return IndexError_InvalidArgument;
}

int QuantizedGraphBuilder::build(IndexThreads::Pointer, IndexHolder::Pointer) {
  return IndexError_InvalidArgument;
}

int QuantizedGraphBuilder::dump(const IndexDumper::Pointer &) {
  return IndexError_Unsupported;
}

int QuantizedGraphBuilder::dump(IndexStorage &) {
  return IndexError_Unsupported;
}

int QuantizedGraphBuilder::cleanup() {
  return 0;
}
}  // namespace zvec::core
