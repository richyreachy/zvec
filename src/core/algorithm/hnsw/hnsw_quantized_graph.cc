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
#include "algorithm/quantized_graph/pq_graph_codec.h"
#include "algorithm/quantized_graph/pq_residual_graph_codec.h"
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
  uint64_t key(uint32_t id) const {
    return entity.get_key(id);
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
  // Research oracle: preserve the production expansion scan budget while
  // separately counting diagnostic neighbor-distance evaluations.
  float reference_distance(const void *vector) {
    return ctx.dist_calculator().dist(vector, ctx.dist_calculator().query());
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
#if defined(QG_HIERARCHICAL)
// Compact the reachable level-1 graph. Its IDs are separate from level-0 IDs.
struct UpperGraphAdapter {
  using Vector = IndexStorage::MemoryBlock;
  const HnswEntity &entity;
  const std::vector<uint32_t> &original;
  const std::vector<std::vector<uint32_t>> &edges;
  bool valid(uint32_t id) const {
    return id < original.size() && entity.get_key(original[id]) != kInvalidKey;
  }
  int get_vector(uint32_t id, Vector &v) const {
    return entity.get_vector(original[id], v);
  }
  uint64_t key(uint32_t id) const {
    return entity.get_key(original[id]);
  }
  const std::vector<uint32_t> &neighbors(uint32_t id) const {
    return edges[id];
  }
};
struct UpperQueryAdapter {
  HnswQueryAdapter &base;
  TopkHeap &heap;
  TopkHeap &reset_results() {
    heap.clear();
    return heap;
  }
  uint32_t topk() const {
    return 1;
  }
  const void *query() const {
    return base.query();
  }
  float distance(const void *v) {
    return base.distance(v);
  }
  float reference_distance(const void *v) {
    return base.reference_distance(v);
  }
  bool excluded(uint32_t) const {
    return false;
  }
  bool reach_scan_limit() const {
    return base.reach_scan_limit();
  }
  void on_expand() {
    base.on_expand();
  }
};
#endif
}  // namespace

struct HnswQuantizedGraph::Impl {
  virtual ~Impl() = default;
  virtual int prebuild(const HnswEntity &, size_t, size_t) = 0;
  virtual int search(uint32_t, HnswContext &, size_t) const = 0;
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
              stored_dimension, cosine)
#if defined(QG_HIERARCHICAL)
        ,
        upper(dimension, Codec(dimension, cosine), max_neighbors,
              stored_dimension, cosine)
#endif
  {
  }
  int prebuild(const HnswEntity &entity, size_t count,
               size_t threads) override {
    int ret = graph.prebuild(HnswGraphAdapter{entity}, count, threads);
#if defined(QG_HIERARCHICAL)
    originals.clear();
    edges.clear();
    if (ret == 0 && entity.cur_max_level() >= 1) {
      std::unordered_map<uint32_t, uint32_t> local;
      originals.push_back(entity.entry_point());
      local[originals[0]] = 0;
      for (size_t i = 0; i < originals.size(); ++i) {
        const auto adjacent = entity.get_neighbors(1, originals[i]);
        std::vector<uint32_t> row;
        for (size_t j = 0; j < adjacent.size(); ++j) {
          const auto id = adjacent[j];
          if (entity.get_key(id) == kInvalidKey) continue;
          auto item = local.emplace(id, uint32_t(originals.size()));
          if (item.second) originals.push_back(id);
          row.push_back(item.first->second);
        }
        edges.push_back(std::move(row));
      }
      ret = upper.prebuild(UpperGraphAdapter{entity, originals, edges},
                           originals.size(), threads);
      LOG_INFO("Research upper quantized graph nodes=%zu", originals.size());
    }
#endif
    return ret;
  }
  int search(uint32_t entry_id, HnswContext &ctx, size_t) const override {
    HnswQueryAdapter query{ctx};
#if defined(QG_HIERARCHICAL)
    if (!originals.empty()) {
      const size_t upper_ef =
          std::getenv("QG_UPPER_EF")
              ? std::strtoul(std::getenv("QG_UPPER_EF"), nullptr, 10)
              : 1;
      TopkHeap heap(std::max(size_t{1}, upper_ef));
      UpperQueryAdapter uq{query, heap};
      const int rc = upper.search(
          upper.entry(), UpperGraphAdapter{ctx.get_entity(), originals, edges},
          uq);
      if (rc != 0) return rc;
      if (!heap.empty()) {
        const auto best = std::min_element(
            heap.begin(), heap.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });
        entry_id = originals[best->first];
      }
    }
#endif
    return graph.search(entry_id, HnswGraphAdapter{ctx.get_entity()}, query);
  }
  void clear() override {
    graph.clear();
#if defined(QG_HIERARCHICAL)
    upper.clear();
    originals.clear();
    edges.clear();
#endif
  }
  uint32_t entry() const override {
    return graph.entry();
  }
  QuantizedGraph<Codec> graph;
#if defined(QG_HIERARCHICAL)
  QuantizedGraph<Codec> upper;
  std::vector<uint32_t> originals;
  std::vector<std::vector<uint32_t>> edges;
#endif
};

// A trained codec needs a corpus before its graph can be constructed. Empty
// streamers defer training until the first search after insertion. The streamer
// excludes mutation/prebuild/clear from searches; this mutex only serializes
// initial lazy publication against other searches and entry() reads.
template <class Codec>
struct HnswQuantizedGraph::TrainedModel final : Impl {
  TrainedModel(size_t dimension, size_t max_neighbors, bool cosine,
               size_t stored_dimension)
      : dimension(dimension),
        max_neighbors(max_neighbors),
        cosine(cosine),
        stored_dimension(stored_dimension) {}

  int initialize(const HnswEntity &entity, size_t count, size_t threads) const {
    Codec codec(dimension, cosine);
    int ret = codec.train(HnswGraphAdapter{entity}, count, threads);
    if (ret != 0 || !codec.trained()) return ret;
    graph = std::make_unique<QuantizedGraph<Codec>>(
        dimension, std::move(codec), max_neighbors, stored_dimension, cosine);
    return 0;
  }
  int prebuild(const HnswEntity &entity, size_t count,
               size_t threads) override {
    std::lock_guard<std::mutex> lock(mutex);
    graph.reset();
    int ret = initialize(entity, count, threads);
    if (ret != 0 || !graph) return ret;
    ret = graph->prebuild(HnswGraphAdapter{entity}, count, threads);
    if (ret != 0) graph.reset();
    return ret;
  }
  int search(uint32_t entry_id, HnswContext &ctx, size_t count) const override {
    QuantizedGraph<Codec> *ready;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!graph) {
        const auto &entity = ctx.get_entity();
        int ret = initialize(entity, count, 1);
        if (ret != 0) return ret;
      }
      ready = graph.get();
    }
    HnswQueryAdapter query{ctx};
    if (!ready) {
      query.reset_results().clear();
      return 0;
    }
    return ready->search(entry_id, HnswGraphAdapter{ctx.get_entity()}, query);
  }
  void clear() override {
    std::lock_guard<std::mutex> lock(mutex);
    // Keep the immutable codebook across insertions; only neighbor blocks and
    // the centroid entry become stale. Explicit prebuild/open trains anew.
    if (graph) graph->clear();
  }
  uint32_t entry() const override {
    std::lock_guard<std::mutex> lock(mutex);
    return graph ? graph->entry() : UINT32_MAX;
  }

  size_t dimension;
  size_t max_neighbors;
  bool cosine;
  size_t stored_dimension;
  mutable std::mutex mutex;
  mutable std::unique_ptr<QuantizedGraph<Codec>> graph;
};

bool HnswQuantizedGraph::supported(const std::string &codec) {
  if (codec == "sq8" || codec == "pq" || codec == "pq_residual") return true;
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
  if (codec == "pq") {
    impl_ = std::make_unique<TrainedModel<PQGraphCodec>>(
        dimension, max_neighbors, cosine, stored_dimension);
  } else if (codec == "pq_residual") {
    impl_ = std::make_unique<TrainedModel<PQResidualGraphCodec>>(
        dimension, max_neighbors, cosine, stored_dimension);
  } else if (codec == "sq8") {
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
  return search(entry_id, ctx, ctx.get_entity().doc_cnt());
}
int HnswQuantizedGraph::search(uint32_t entry_id, HnswContext &ctx,
                               size_t doc_cnt) const {
  return impl_ ? impl_->search(entry_id, ctx, doc_cnt) : IndexError_Unsupported;
}
void HnswQuantizedGraph::clear() {
  if (impl_) impl_->clear();
}
uint32_t HnswQuantizedGraph::entry() const {
  return impl_ ? impl_->entry() : UINT32_MAX;
}
HnswQuantizedGraph::~HnswQuantizedGraph() = default;
}  // namespace zvec::core
