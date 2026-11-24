// Copyright 2026-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

int QuantizedGraphBuilder::Result::save(const std::string &,
                                        const std::vector<uint64_t> &) const {
  return IndexError_Unsupported;
}

int QuantizedGraphBuilder::Result::open(const std::string &, const ailego::Params &,
                                        std::vector<uint64_t> *) {
  return IndexError_Unsupported;
}

int QuantizedGraphBuilder::Result::load(IndexStorage &, const ailego::Params &,
                                        std::vector<uint64_t> *) {
  return IndexError_Unsupported;
}

int QuantizedGraphBuilder::cleanup() {
  return 0;
}
}  // namespace zvec::core
