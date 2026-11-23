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
#pragma once

#include <zvec/core/framework/index_searcher.h>
#include "quantized_graph_builder.h"

namespace zvec::core {
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
class QuantizedGraphSearcher final : public IndexSearcher {
 public:
  int init(const ailego::Params &p) override {
    params_ = p;
    return 0;
  }
  const ailego::Params &params() const override {
    return params_;
  }
  const Stats &stats() const override {
    return stats_;
  }
  const IndexMeta &meta() const override {
    return meta_;
  }
  int cleanup() override;
  int unload() override {
    return cleanup();
  }
  int load(IndexStorage::Pointer storage, IndexMetric::Pointer metric) override;
  Context::Pointer create_context() const override;
  int search_impl(const void *v, const IndexQueryMeta &m,
                  Context::Pointer &ctx) const override {
    return search_impl(v, m, 1, ctx);
  }
  int search_impl(const void *vectors, const IndexQueryMeta &meta,
                  uint32_t count, Context::Pointer &context) const override;

 private:
  Stats stats_;
  IndexMeta meta_;
  ailego::Params params_;
  std::vector<uint64_t> keys_;
  std::shared_ptr<const QuantizedGraphData> data_;
  std::shared_ptr<QuantizedGraphIndex> graph_;
  IndexMetric::MatrixDistance distance_;
};
#endif
}  // namespace zvec::core
