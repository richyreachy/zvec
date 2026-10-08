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
#include "hnsw_symphony_qg.h"
#include "hnsw_context.h"
#if RABITQ_SUPPORTED
#include "algorithm/quantized_graph/quantized_graph.h"
#include "algorithm/quantized_graph/symphony_qg_codec.h"
#endif

namespace zvec::core {
#if RABITQ_SUPPORTED
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

struct HnswSymphonyQG::Impl {
  Impl(size_t dimension, size_t max_neighbors)
      : graph(dimension, SymphonyQGCodec(dimension), max_neighbors) {}
  QuantizedGraph<SymphonyQGCodec> graph;
};

HnswSymphonyQG::HnswSymphonyQG(size_t dimension, size_t max_neighbors)
    : impl_(std::make_unique<Impl>(dimension, max_neighbors)) {}

int HnswSymphonyQG::prebuild(const HnswEntity &entity, size_t doc_cnt,
                             size_t threads) {
  return impl_->graph.prebuild(HnswGraphAdapter{entity}, doc_cnt, threads);
}
int HnswSymphonyQG::search(uint32_t entry, HnswContext &ctx) const {
  HnswQueryAdapter query{ctx};
  return impl_->graph.search(entry, HnswGraphAdapter{ctx.get_entity()}, query);
}
void HnswSymphonyQG::clear() {
  impl_->graph.clear();
}
uint32_t HnswSymphonyQG::entry() const {
  return impl_->graph.entry();
}
#else
struct HnswSymphonyQG::Impl {};
HnswSymphonyQG::HnswSymphonyQG(size_t, size_t) {}
int HnswSymphonyQG::prebuild(const HnswEntity &, size_t, size_t) {
  return IndexError_Unsupported;
}
int HnswSymphonyQG::search(uint32_t, HnswContext &) const {
  return IndexError_Unsupported;
}
void HnswSymphonyQG::clear() {}
uint32_t HnswSymphonyQG::entry() const {
  return kInvalidNodeId;
}
#endif
HnswSymphonyQG::~HnswSymphonyQG() = default;
}  // namespace zvec::core
