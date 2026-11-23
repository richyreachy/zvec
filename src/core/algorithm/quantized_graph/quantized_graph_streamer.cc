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
#include <zvec/core/framework/index_factory.h>
#include "quantized_graph_searcher.h"

namespace zvec::core {
class QuantizedGraphStreamer final : public IndexStreamer {
 public:
  int init(const IndexMeta &meta, const ailego::Params &params) override {
    if (!QuantizedGraphSupportsMeta(meta)) return IndexError_InvalidArgument;
    meta_ = meta;
    return searcher_.init(params);
  }
  int open(IndexStorage::Pointer s) override {
    int rc = searcher_.load(std::move(s), nullptr);
    if (!rc && (searcher_.meta().dimension() != meta_.dimension() ||
                searcher_.meta().metric_name() != meta_.metric_name())) {
      close();
      return IndexError_InvalidFormat;
    }
    return rc;
  }
  int close() override {
    return searcher_.unload();
  }
  int cleanup() override {
    return close();
  }
  int flush(uint64_t) override {
    return 0;
  }
  const IndexMeta &meta() const override {
    return meta_;
  }
  const Stats &stats() const override {
    return searcher_.stats();
  }
  Context::Pointer create_context() const override {
    return searcher_.create_context();
  }
  int search_impl(const void *v, const IndexQueryMeta &m,
                  Context::Pointer &c) const override {
    return searcher_.search_impl(v, m, c);
  }
  int search_impl(const void *v, const IndexQueryMeta &m, uint32_t n,
                  Context::Pointer &c) const override {
    return searcher_.search_impl(v, m, n, c);
  }

 private:
  IndexMeta meta_;
  QuantizedGraphSearcher searcher_;
};

INDEX_FACTORY_REGISTER_STREAMER(QuantizedGraphStreamer);
}  // namespace zvec::core
