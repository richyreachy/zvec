// Copyright 2025-present the zvec project
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
#include "hnsw_quantized_graph.h"
#include "algorithm/quantized_graph/quantized_graph.h"
#include "algorithm/quantized_graph/sq8_graph_codec.h"
#include "hnsw_context.h"
#if RABITQ_SUPPORTED
#include <rabitqlib/utils/cpu_features.hpp>
#include "algorithm/quantized_graph/symphony_qg_codec.h"
#endif

namespace zvec::core {
namespace {
struct HnswGraphAdapter {
  using Vector = IndexStorage::MemoryBlock;
  const HnswEntity &entity;
  bool valid(uint32_t id) const {
    return entity.get_key(id) != kInvalidKey;
  }
  int get_vector(uint32_t id, Vector &vector) const {
    return entity.get_vector(id, vector);
  }
  Neighbors neighbors(uint32_t id) const {
    return entity.get_neighbors(0, id);
  }
};

struct HnswQueryAdapter {
  HnswContext &ctx;
  TopkHeap &reset_results() {
    return ctx.search_heap().reset<TopkHeap>(std::max(ctx.topk(), ctx.ef()));
  }
  uint32_t topk() const {
    return ctx.topk();
  }
  const void *query() const {
    return ctx.dist_calculator().query();
  }
  float distance(const void *vector) {
    return ctx.dist_calculator().dist(vector);
  }
  bool excluded(uint32_t id) const {
    return ctx.filter().is_valid() &&
           ctx.filter()(ctx.get_entity().get_key(id));
  }
  bool reach_scan_limit() const {
    return ctx.reach_scan_limit();
  }
  void on_expand() {
    if (ctx.debugging()) ++(*ctx.mutable_stats_get_neighbors());
  }
};
}  // namespace

struct HnswQuantizedGraph::Impl {
  virtual ~Impl() = default;
  virtual int prebuild(const HnswEntity &, size_t, size_t) = 0;
  virtual int search(uint32_t, HnswContext &) const = 0;
  virtual void clear() = 0;
  virtual uint32_t entry() const = 0;
};

// Runtime selection happens once per operation; traversal retains static codec
// dispatch, including inlining the batch distance kernel.
template <class Codec>
struct HnswQuantizedGraph::Model final : Impl {
  Model(size_t dimension, size_t max_neighbors, bool cosine,
        size_t stored_dimension)
      : graph(dimension, Codec(dimension, cosine), max_neighbors,
              stored_dimension) {}
  int prebuild(const HnswEntity &entity, size_t count,
               size_t threads) override {
    return graph.prebuild(HnswGraphAdapter{entity}, count, threads);
  }
  int search(uint32_t entry_id, HnswContext &ctx) const override {
    HnswQueryAdapter query{ctx};
    return graph.search(entry_id, HnswGraphAdapter{ctx.get_entity()}, query);
  }
  void clear() override {
    graph.clear();
  }
  uint32_t entry() const override {
    return graph.entry();
  }
  QuantizedGraph<Codec> graph;
};

bool HnswQuantizedGraph::supported(const std::string &codec) {
  if (codec == "sq8") return true;
#if RABITQ_SUPPORTED
  if (codec == "symphony_qg") {
    return rabitqlib::cpu::has_avx2() || rabitqlib::cpu::has_avx512_core();
  }
#endif
  return false;
}

HnswQuantizedGraph::HnswQuantizedGraph(size_t dimension, size_t max_neighbors,
                                       bool cosine, size_t stored_dimension,
                                       const std::string &codec) {
  if (!supported(codec)) return;
  if (codec == "sq8") {
    impl_ = std::make_unique<Model<SQ8GraphCodec>>(dimension, max_neighbors,
                                                   cosine, stored_dimension);
  }
#if RABITQ_SUPPORTED
  else if (codec == "symphony_qg") {
    impl_ = std::make_unique<Model<SymphonyQGCodec>>(dimension, max_neighbors,
                                                     cosine, stored_dimension);
  }
#endif
}
int HnswQuantizedGraph::prebuild(const HnswEntity &entity, size_t count,
                                 size_t threads) {
  return impl_ ? impl_->prebuild(entity, count, threads)
               : IndexError_Unsupported;
}
int HnswQuantizedGraph::search(uint32_t entry_id, HnswContext &ctx) const {
  return impl_ ? impl_->search(entry_id, ctx) : IndexError_Unsupported;
}
void HnswQuantizedGraph::clear() {
  if (impl_) impl_->clear();
}
uint32_t HnswQuantizedGraph::entry() const {
  return impl_ ? impl_->entry() : UINT32_MAX;
}
HnswQuantizedGraph::~HnswQuantizedGraph() = default;
}  // namespace zvec::core
