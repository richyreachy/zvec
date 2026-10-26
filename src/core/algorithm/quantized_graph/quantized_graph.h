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
#pragma once

#if defined(QG_CENTER_FP16) || defined(QG_CENTER_SQ8)
#define QG_CENTER_QUANTIZED 1
#endif
#include <algorithm>
#if defined(QG_TOPOLOGY_STATS) || defined(QG_CENTER_SHIFT)
#include <cstdio>
#include <cstdlib>
#endif
#if defined(QG_BUCKET)
#include "quantized_graph_bucket.h"
#endif
#ifndef QG_EXPERIMENT
#define QG_EXPERIMENT 0
#endif
#ifndef QG_PRUNE_ALPHA
#define QG_PRUNE_ALPHA 1.0f
#endif
#ifndef QG_DIAG
#define QG_DIAG 0
#endif
#if QG_DIAG || defined(QG_PATH_TRACE) || defined(QG_QUEUE_STATS) || \
    defined(QG_RUNTIME_ADAPTIVE) || defined(QG_CENTER_STATS) ||     \
    defined(QG_TAIL_REFINE) || defined(QG_LOCAL_COMPLETION)
#include <cstdio>
#include <cstdlib>
#endif
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include "utility/graph_search.h"
#include "quantized_graph_beam.h"
#if defined(QG_COVER_PRUNE)
#include "quantized_graph_cover.h"
#endif
#if defined(QG_PROFILE)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#endif
#if defined(QG_PROJECTION_ROUTER) || defined(QG_ROTATED_CENTER)
#include <cstdlib>
#include "quantized_graph_router.h"
#endif
#if defined(QG_CENTER_QUANTIZED)
#include "quantized_graph_half_center.h"
#if !defined(QG_ADAPTIVE) && !defined(QG_DEFERRED_COSINE)
#error Half center experiment requires an exact Top-K stopping state
#endif
#if defined(QG_DEFERRED_COSINE) && defined(QG_ADAPTIVE) && \
    !defined(QG_DEFERRED_ADAPTIVE)
#error Deferred adaptive navigation requires interval upper bounds
#endif
#if defined(QG_DEFERRED_ADAPTIVE) && \
    (!defined(QG_ADAPTIVE) || !defined(QG_DEFERRED_COSINE))
#error Interval adaptive mode requires deferred cosine and adaptive navigation
#endif
#endif
#if defined(QG_ROTATED_CENTER)
#if !defined(QG_CENTER_SQ8) || !defined(QG_DEFERRED_COSINE) || \
    !defined(QG_REANCHOR) || !defined(QG_ROTATION_768)
#error Rotated center prototype requires SQ8, deferred cosine, reanchoring and the 768D rotation
#endif
#endif
#if defined(QG_CENTER_SHIFT)
#if !defined(QG_CENTER_SQ8) || !defined(QG_DEFERRED_COSINE) || \
    !defined(QG_REANCHOR) || defined(QG_ROTATED_CENTER)
#error Shifted center prototype requires SQ8, deferred cosine and reanchoring, without rotated centers
#endif
#endif
#if defined(QG_CENTER_CONTROL_VARIATE) && !defined(QG_CENTER_SHIFT)
#error Center control variate requires a shared shift
#endif
#if defined(QG_INTEGER_CENTER)
#if !defined(QG_CENTER_SHIFT) || defined(QG_CENTER_CONTROL_VARIATE)
#error Integer center prototype requires shared shift without the FP32 control variate
#endif
#include "quantized_graph_integer_center.h"
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace zvec::core {

// Fixed-batch quantized neighbor search over an externally owned FP32 graph.
// Graph: valid(id), neighbors(id) (size/index access), and get_vector(id,
// Vector&). A Graph::Vector owns/pins its FP32 data until destruction; data()
// exposes it. IDs occupy [0, doc_cnt), with holes reported by valid(). Reads
// must be safe across prebuild workers and concurrent searches. Codec:
// kBatchSize, encoded_dim(), batch_bytes(), transform(), encode(), and
// Query/prepare_query()/scan(). Estimates and exact scores must use the same
// metric; centroid selection and diversity pruning use squared L2 on the
// first dimension coordinates. Optional trailing FP32 metadata is retained
// in cached centers for exact scoring but excluded from graph geometry.
// Context: reset_results() returns a bounded heap with limit/size/emplace;
// query(), topk(), distance(), excluded(id), reach_scan_limit(), on_expand().
// distance() accounts for the exact scan budget; excluded() means EXCLUDE.
// The caller serializes clear/prebuild/graph mutations against searches. Clear
// the cache before any mutation; lazy blocks then rebuild from the source
// graph. Templates keep backend and codec calls inline in the traversal hot
// loop.
template <class Codec>
class QuantizedGraph {
 public:
  using NodeId = uint32_t;
  static constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::max();

  // stored_dimension defaults to dimension; larger values preserve auxiliary
  // fields following the coordinates (e.g. cosine storage's original norm).
  QuantizedGraph(size_t dimension, Codec codec, size_t max_neighbors = 32,
                 size_t stored_dimension = 0, bool cosine = false);
  void clear();
  template <class Graph>
  int prebuild(const Graph &graph, size_t doc_cnt, size_t threads);
  template <class Graph, class Context>
  int search(NodeId entry, const Graph &graph, Context &ctx) const;
  NodeId entry() const {
    return entry_;
  }

 private:
#if defined(QG_SHARED_DOMINANCE)
  using LocalVisitSet = QuantizedGraphSharedVisit;
#else
  class LocalVisitSet {
   public:
    void reset(size_t expect) {
#if defined(QG_VISIT_EXPECT)
      expect = QG_VISIT_EXPECT;
#endif
      size_t cap = 64;
      while (cap < expect * 8) cap *= 2;
      if (table_.size() != cap)
        table_.assign(cap, kEmpty);
      else
        std::fill(table_.begin(), table_.end(), kEmpty);
      mask_ = cap - 1;
      overflow_.clear();
    }

    bool visited(NodeId id) const {
      size_t i = id & mask_;
      for (size_t n = 0; n <= mask_; ++n) {
        const NodeId v = table_[i];
        if (v == id) return true;
        if (v == kEmpty) return false;
        i = (i + 1) & mask_;
      }
      // Table completely full: entries beyond it live in the overflow set.
      return overflow_.find(id) != overflow_.end();
    }

    void set_visited(NodeId id) {
      size_t i = id & mask_;
      for (size_t n = 0; n <= mask_; ++n) {
        const NodeId v = table_[i];
        if (v == id) return;
        if (v == kEmpty) {
          table_[i] = id;
          return;
        }
        i = (i + 1) & mask_;
      }
      overflow_.insert(id);
    }

   private:
    static constexpr NodeId kEmpty = 0xFFFFFFFFu;
    std::vector<NodeId> table_;
    std::unordered_set<NodeId> overflow_;
    size_t mask_{0};
  };
#endif

  // Resolve TLS once at the query boundary. In particular, probing the visit
  // table must not call __tls_get_addr for every candidate in a shared build.
#if defined(QG_HEAP)
  struct AdaptiveHeap {
    struct Entry {
      float distance;
      NodeId id;
    };
    struct Greater {
      bool operator()(const Entry &a, const Entry &b) const {
        return a.distance > b.distance;
      }
    };
    std::vector<Entry> data;
    void reset(size_t) {
      data.clear();
    }
    void insert(NodeId id, float d) {
      data.push_back({d, id});
      std::push_heap(data.begin(), data.end(), Greater{});
    }
    bool has_next() const {
      return !data.empty();
    }
    NodeId next_id_at(size_t) const {
      return data.front().id;
    }
    float next_distance() const {
      return data.front().distance;
    }
    NodeId pop() {
      std::pop_heap(data.begin(), data.end(), Greater{});
      NodeId id = data.back().id;
      data.pop_back();
      return id;
    }
    bool is_full(float) const {
      return false;
    }
  };
#endif
  struct SearchState {
    LocalVisitSet visited;
#if defined(QG_FIRST_ESTIMATE)
    LocalVisitSet discovered;
#endif
#if defined(QG_BUCKET)
    QuantizedGraphBucket beam;
#if defined(QG_BUCKET_DOMINANCE)
#if defined(QG_DOMINANCE_CACHE)
    QuantizedGraphDominanceCache dominance;
#else
    QuantizedGraphBucketDominance dominance;
#endif
#endif
#elif defined(QG_HEAP)
    AdaptiveHeap beam;
#else
    QuantizedGraphBeam beam{1};
#endif
    typename Codec::Query query;
#if defined(QG_INTEGER_CENTER)
    IntegerCenter::Query integer_query;
#endif
#if defined(QG_PROJECTION_ROUTER)
    std::vector<float> route_scores, route_query;
#endif
    std::vector<NodeId> expanded;
#if defined(QG_DEFERRED_COSINE)
    std::vector<std::pair<float, NodeId>> refine_candidates;
    std::vector<float> refine_best;
#endif
    GraphSearchScratch scratch;
#if defined(QG_ADAPTIVE)
    std::vector<float> best_exact;
#endif
  };

  // A call boundary prevents the compiler from rematerializing the TLS
  // address inside the traversal after inlining its small policy methods.
#if defined(__GNUC__) || defined(__clang__)
  __attribute__((noinline))
#elif defined(_MSC_VER)
  __declspec(noinline)
#endif
  static SearchState &
  search_state() {
    thread_local SearchState state;
    return state;
  }

  // One fixed-stride slot per node inside a single arena allocation, like
  // upstream SymphonyQG's row layout: header, center vector, neighbor ids,
  // then codec payloads. The fixed stride makes every block address pure
  // arithmetic (base + id * stride) and keeps the index in one contiguous
  // mapping. Trivially destructible; the arena is freed wholesale.
  struct Block {
#if defined(QG_LOCAL_COMPLETION)
    static constexpr size_t kCenterOffset = 192;
#else
    static constexpr size_t kCenterOffset = 64;
#endif

    uint32_t neighbor_cnt;
    uint32_t center_floats;
    uint32_t code_bytes;
    // Rounded-up capacity of the neighbor region so the codes offset does
    // not depend on the actual degree.
    uint32_t neighbor_bytes;
#if defined(QG_LOCAL_COMPLETION)
    uint32_t local_count;
    NodeId local_neighbors[32];
#endif
#if defined(QG_CENTER_QUANTIZED)
    float center_error, center_norm2;
#if defined(QG_CENTER_SQ8)
    float center_offset, center_scale;
#if defined(QG_INTEGER_CENTER)
    double center_mean_dot;
    float center_residual_norm, center_coded_norm, integer_anchor_projection;
#endif
#if defined(QG_CENTER_CONTROL_VARIATE)
    float center_projection, center_projection_error;
#endif
#endif
#endif

    const float *center() const {
      return reinterpret_cast<const float *>(
          reinterpret_cast<const char *>(this) + kCenterOffset);
    }
    const NodeId *neighbors() const {
      return reinterpret_cast<const NodeId *>(
          reinterpret_cast<const char *>(this) + kCenterOffset +
          center_floats * sizeof(float));
    }
    const char *codes() const {
      return reinterpret_cast<const char *>(neighbors()) + neighbor_bytes;
    }
  };

  struct BlockDeleter {
    void operator()(const Block *block) const {
      ::operator delete(const_cast<Block *>(block), std::align_val_t{64});
    }
  };

  struct ArenaDeleter {
    static constexpr size_t kAlignment = 2 * 1024 * 1024;
    void operator()(char *ptr) const {
      ::operator delete(ptr, std::align_val_t{kAlignment});
    }
  };

  struct Scratch {
    std::vector<float> encoded_center;
    std::vector<float> encoded;
    std::vector<float> vector;
    std::vector<NodeId> neighbors;
    std::vector<float> selection_cache;
  };

  const Block *slot(NodeId id) const {
    return reinterpret_cast<const Block *>(arena_.get() +
                                           static_cast<size_t>(id) * stride_);
  }

  // The returned pointer is owned by the arena (prebuilt) or lazy_blocks_
  // (post-clear rebuild); the caller's exclusive lock keeps clear() from
  // destroying blocks during a search.
  template <class Graph>
  const Block *get_block(const Graph &entity, NodeId id) const;
  // Writes one fixed-stride block into slot; false on error, with no cleanup
  // (the caller owns the memory).
  template <class Graph>
  bool build_block(const Graph &entity, NodeId id, const void *center,
                   Block *slot, Scratch &scratch, int &error) const;

  size_t dimension_;
  size_t stored_dimension_;
  size_t stored_center_bytes_;
  bool compressed_center_{false};
#if defined(QG_CENTER_SHIFT)
  std::vector<float> center_shift_;
  double center_shift_norm2_{0};
#endif
  bool cosine_{false};
  size_t max_neighbors_{32};
  size_t neighbor_bytes_{0};
  Codec codec_;
#if defined(QG_PROJECTION_ROUTER)
  QuantizedGraphRouter router_;
#endif
  // Immutable after prebuild and before the next clear(); read lock-free.
  std::unique_ptr<char[], ArenaDeleter> arena_;
  NodeId arena_count_{0};
  size_t stride_{0};
  NodeId entry_{kInvalidNodeId};
  mutable std::mutex mutex_;
  mutable std::unordered_map<NodeId, std::shared_ptr<const Block>> lazy_blocks_;
};

template <class Codec>
QuantizedGraph<Codec>::QuantizedGraph(size_t dimension, Codec codec,
                                      size_t max_neighbors,
                                      size_t stored_dimension, bool cosine)
    : dimension_(dimension),
      stored_dimension_(std::max(dimension, stored_dimension)),
      codec_(std::move(codec)) {
  cosine_ = cosine;
  stored_center_bytes_ = stored_dimension_ * sizeof(float);
#if defined(QG_OMIT_COSINE_NORM)
  // The normalized cosine kernel reads only dimension_ coordinates; its
  // trailing original-norm field is not used by 1-dot(query, center).
  if (cosine_) stored_center_bytes_ = dimension_ * sizeof(float);
#endif
#if defined(QG_ALIGN_CENTERS)
  stored_center_bytes_ = (stored_center_bytes_ + 63) & ~size_t{63};
#endif
#if defined(QG_CENTER_QUANTIZED)
  compressed_center_ = !cosine;
#if defined(QG_DEFERRED_COSINE)
  compressed_center_ = cosine;
#endif
  if (compressed_center_) {
#if defined(QG_CENTER_SQ8)
    stored_center_bytes_ = (dimension_ + 63) & ~size_t{63};
#else
    stored_center_bytes_ = (dimension_ * sizeof(uint16_t) + 63) & ~size_t{63};
#endif
  }
#else
  (void)cosine;
#endif
  static_assert(Codec::kBatchSize > 0, "Codec batch size must be positive");
  constexpr size_t batch_size = Codec::kBatchSize;
  max_neighbors_ =
      std::max(batch_size, max_neighbors - max_neighbors % batch_size);
  neighbor_bytes_ = (max_neighbors_ * sizeof(NodeId) + 63) & ~size_t{63};
  const size_t code_bytes =
      (max_neighbors_ / batch_size) * codec_.batch_bytes();
  stride_ = (Block::kCenterOffset + stored_center_bytes_ + neighbor_bytes_ +
             code_bytes + 63) &
            ~size_t{63};
}

template <class Codec>
void QuantizedGraph<Codec>::clear() {
  // Called under the caller's exclusive lock; no concurrent searches.
  arena_.reset();
  arena_count_ = 0;
  entry_ = kInvalidNodeId;
#if defined(QG_PROJECTION_ROUTER)
  router_.clear();
#endif
  std::lock_guard<std::mutex> lock(mutex_);
  lazy_blocks_.clear();
}

template <class Codec>
template <class Graph>
bool QuantizedGraph<Codec>::build_block(const Graph &entity, NodeId id,
                                        const void *center_ptr, Block *slot,
                                        Scratch &scratch, int &error) const {
  const size_t dim = codec_.encoded_dim();
  constexpr size_t batch_size = Codec::kBatchSize;
  const size_t batch_bytes = codec_.batch_bytes();

  scratch.neighbors.clear();
  const auto &neighbors = entity.neighbors(id);
  scratch.neighbors.reserve(neighbors.size());
  for (uint32_t i = 0; i < neighbors.size(); ++i) {
    if (entity.valid(neighbors[i])) {
      scratch.neighbors.push_back(neighbors[i]);
    }
  }
#if defined(QG_LOCAL_COMPLETION)
  // Query-independent local result edges, separate from diverse navigation.
  std::vector<NodeId> local_ids = scratch.neighbors;
  const size_t seed_count = std::min(size_t{8}, scratch.neighbors.size());
  for (size_t i = 0; i < seed_count; ++i) {
    const auto children = entity.neighbors(
        scratch.neighbors[i * scratch.neighbors.size() / seed_count]);
    const size_t samples = std::min(size_t{16}, size_t(children.size()));
    for (size_t j = 0; j < samples; ++j)
      local_ids.push_back(children[j * children.size() / samples]);
  }
  std::sort(local_ids.begin(), local_ids.end());
  local_ids.erase(std::unique(local_ids.begin(), local_ids.end()),
                  local_ids.end());
  std::vector<std::pair<float, NodeId>> local_rank;
  for (NodeId candidate : local_ids) {
    if (candidate == id || !entity.valid(candidate)) continue;
    typename Graph::Vector raw;
    const int ret = entity.get_vector(candidate, raw);
    if (ret != 0) {
      error = ret;
      return false;
    }
    if (raw.data() == nullptr) {
      error = IndexError_Mismatch;
      return false;
    }
    const float *v = static_cast<const float *>(raw.data()),
                *u = static_cast<const float *>(center_ptr);
    float d = 0;
    for (size_t j = 0; j < dimension_; ++j) {
      float x = v[j] - u[j];
      d += x * x;
    }
    local_rank.emplace_back(d, candidate);
  }
  const size_t local_count = std::min(size_t{32}, local_rank.size());
  std::partial_sort(local_rank.begin(), local_rank.begin() + local_count,
                    local_rank.end());
#endif
#if defined(QG_AUGMENT)
  const size_t original_neighbor_count = scratch.neighbors.size();
  // Offline two-hop candidate repair. Fixed row capacity and the exact same
  // query kernel are retained. No query or GT is used during construction.
  if (QG_AUGMENT == 2 || scratch.neighbors.size() < max_neighbors_) {
    const size_t original = scratch.neighbors.size();
    std::array<NodeId, 512> seen;
    seen.fill(kInvalidNodeId);
    auto add = [&](NodeId v) {
      if (v == id || !entity.valid(v)) return;
      size_t h = (uint64_t(v) * 2654435761U) & 511U;
      while (seen[h] != kInvalidNodeId && seen[h] != v) h = (h + 1) & 511U;
      if (seen[h] == v) return;
      seen[h] = v;
      scratch.neighbors.push_back(v);
    };
    for (NodeId v : scratch.neighbors) {
      size_t h = (uint64_t(v) * 2654435761U) & 511U;
      while (seen[h] != kInvalidNodeId && seen[h] != v) h = (h + 1) & 511U;
      seen[h] = v;
    }
    const size_t seeds = std::min(original, size_t{8});
    for (size_t j = 0; j < seeds && scratch.neighbors.size() < 256; ++j) {
      const NodeId seed = scratch.neighbors[j * original / seeds];
      const auto children = entity.neighbors(seed);
      const size_t samples = std::min(size_t(children.size()), size_t{16});
      for (size_t k = 0; k < samples && scratch.neighbors.size() < 256; ++k)
        add(children[k * children.size() / samples]);
    }
  }
#endif
  // Trim to the block's degree bound like upstream SymphonyQG's fixed
  // degree_bound_, but with the HNSW diversity heuristic so the truncated
  // graph keeps short and long-range links and stays navigable: plain
  // truncation loses several recall points because HNSW's link order is
  // newest-first. Distances use squared L2 in coordinate space; for cosine
  // these are the normalized coordinates, with norm metadata excluded.
  const size_t kMaxBlockNeighbors = max_neighbors_;
  if (scratch.neighbors.size() > kMaxBlockNeighbors) {
    const size_t cnt = scratch.neighbors.size();
    const auto *center = static_cast<const float *>(center_ptr);
    scratch.selection_cache.assign(cnt * dimension_, 0.0f);
    std::vector<float> dists(cnt);
    for (size_t i = 0; i < cnt; ++i) {
      typename Graph::Vector raw;
      const int ret = entity.get_vector(scratch.neighbors[i], raw);
      if (ret != 0) {
        error = ret;
        return false;
      }
      if (raw.data() == nullptr) {
        error = IndexError_Mismatch;
        return false;
      }
      const auto *v = static_cast<const float *>(raw.data());
      std::copy(v, v + dimension_,
                scratch.selection_cache.begin() + i * dimension_);
      float d = 0.0f;
      for (size_t k = 0; k < dimension_; ++k) {
        const float diff = v[k] - center[k];
        d += diff * diff;
      }
      dists[i] = d;
    }
#if defined(QG_QUANTIZED_PRUNE)
    // Offline pruning in the score geometry used by one-bit navigation.
    // rhat preserves r dot rhat = ||r||^2. No queries/GT enter construction.
    std::vector<float> residuals(cnt * dim), reconstructed(cnt * dim),
        rotated_center, rotated;
    codec_.transform(center, rotated_center);
    for (size_t i = 0; i < cnt; ++i) {
      codec_.transform(scratch.selection_cache.data() + i * dimension_,
                       rotated);
      double norm = 0, l1 = 0;
      for (size_t j = 0; j < dim; ++j) {
        float r = rotated[j] - rotated_center[j];
        residuals[i * dim + j] = r;
        norm += double(r) * r;
        l1 += std::abs(double(r));
      }
      const float amplitude = l1 > 0 ? float(norm / l1) : 0.f;
      for (size_t j = 0; j < dim; ++j)
        reconstructed[i * dim + j] =
            residuals[i * dim + j] > 0 ? amplitude : -amplitude;
    }
    auto estimated_cross_distance = [&](uint32_t query, uint32_t target) {
      const float *r = residuals.data() + query * dim,
                  *h = reconstructed.data() + target * dim;
      float dot = 0;
      for (size_t j = 0; j < dim; ++j) dot += r[j] * h[j];
      return dists[query] + dists[target] - 2 * dot;
    };
#endif
    std::vector<uint32_t> order(cnt);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return dists[a] < dists[b]; });
    const float *cache = scratch.selection_cache.data();
    auto sq_dist = [&](uint32_t a, uint32_t b) {
      const float *x = cache + a * dimension_;
      const float *y = cache + b * dimension_;
      float d = 0.0f;
      for (size_t k = 0; k < dimension_; ++k) {
        const float diff = x[k] - y[k];
        d += diff * diff;
      }
      return d;
    };
    std::vector<uint32_t> selected;
    selected.reserve(kMaxBlockNeighbors);
#if defined(QG_COVER_PRUNE)
    // Targets and edge candidates are database-derived only. Greedy optimizes
    // local coverage radius; this is not a global navigability guarantee.
    std::vector<float> residual(cnt * dim), reconstructed(cnt * dim), rr(cnt),
        hh(cnt), rotated_center, rotated;
    codec_.transform(center, rotated_center);
    for (size_t i = 0; i < cnt; ++i) {
      codec_.transform(cache + i * dimension_, rotated);
      double norm = 0, l1 = 0;
      for (size_t j = 0; j < dim; ++j) {
        const float r = rotated[j] - rotated_center[j];
        residual[i * dim + j] = r;
        norm += double(r) * r;
        l1 += std::abs(double(r));
      }
      rr[i] = float(norm);
      const float amplitude = l1 > 0 ? float(norm / l1) : 0.f;
      hh[i] = float(dim) * amplitude * amplitude;
      for (size_t j = 0; j < dim; ++j)
        reconstructed[i * dim + j] =
            residual[i * dim + j] > 0 ? amplitude : -amplitude;
    }
    const size_t targets = std::min(size_t{64}, cnt);
    std::vector<float> gain(cnt * targets);
    for (size_t s = 0; s < cnt; ++s) {
      const uint32_t edge = order[s];
      for (size_t t = 0; t < targets; ++t) {
        const uint32_t target = order[t * cnt / targets];
        const auto dots = QuantizedGraphCoverDots(
            residual.data() + target * dim, residual.data() + edge * dim,
            reconstructed.data() + edge * dim, dim);
#if QG_COVER_PRUNE >= 3
        gain[s * targets + t] = QuantizedGraphRelativeProgress(
            rr[target], rr[edge], dots[0], dots[1], QG_COVER_PRUNE == 4);
#if defined(QG_COVER_SATURATION)
        gain[s * targets + t] =
            std::min(gain[s * targets + t], float(QG_COVER_SATURATION));
#endif
#else
        gain[s * targets + t] =
            QuantizedGraphRelativeRadius(rr[target], rr[edge], hh[edge],
                                         dots[0], dots[1], QG_COVER_PRUNE == 2);
#endif
      }
    }
    for (uint32_t pos :
         QuantizedGraphGreedyCover(gain, cnt, targets, kMaxBlockNeighbors))
      selected.push_back(order[pos]);
#else
#if defined(QG_FORCED_NEAR)
    // Reserve a small dense local neighborhood, then spend remaining slots
    // on diverse navigation edges. Construction uses database vectors only.
    for (size_t i = 0; i < std::min(size_t{QG_FORCED_NEAR},
                                    std::min(order.size(), kMaxBlockNeighbors));
         ++i)
      selected.push_back(order[i]);
#endif
#if defined(QG_AUGMENT) && QG_AUGMENT >= 3
    if (original_neighbor_count < kMaxBlockNeighbors)
      for (uint32_t i = 0; i < original_neighbor_count; ++i)
        selected.push_back(i);
#endif
    for (uint32_t idx : order) {
      if (selected.size() == kMaxBlockNeighbors) break;
#if defined(QG_FORCED_NEAR)
      if (std::find(selected.begin(), selected.end(), idx) != selected.end())
        continue;
#endif
#if defined(QG_AUGMENT) && QG_AUGMENT >= 3
      if (original_neighbor_count < kMaxBlockNeighbors &&
          idx < original_neighbor_count)
        continue;
#endif
      bool diverse = true;
      for (uint32_t s : selected) {
#if defined(QG_AUGMENT) && QG_AUGMENT == 4
        if (original_neighbor_count < kMaxBlockNeighbors) break;
#endif
        if (
#if defined(QG_QUANTIZED_PRUNE)
            QG_PRUNE_ALPHA * estimated_cross_distance(idx, s) < dists[idx]
#if QG_QUANTIZED_PRUNE == 2
            && QG_PRUNE_ALPHA * sq_dist(idx, s) < dists[idx]
#endif
#else
            QG_PRUNE_ALPHA * sq_dist(idx, s) < dists[idx]
#endif
        ) {
          diverse = false;
          break;
        }
      }
      if (diverse) selected.push_back(idx);
    }
#endif
    if (selected.size() < kMaxBlockNeighbors) {
      std::vector<char> taken(cnt, 0);
      for (uint32_t idx : selected) taken[idx] = 1;
      for (uint32_t idx : order) {
        if (selected.size() == kMaxBlockNeighbors) break;
        if (!taken[idx]) selected.push_back(idx);
      }
    }
    std::vector<NodeId> pruned;
    pruned.reserve(kMaxBlockNeighbors);
    for (uint32_t idx : selected) pruned.push_back(scratch.neighbors[idx]);
    scratch.neighbors = std::move(pruned);
  }
  const size_t neighbor_cnt = scratch.neighbors.size();
  const size_t code_bytes =
      ((neighbor_cnt + batch_size - 1) / batch_size) * batch_bytes;
  auto *block = new (slot) Block{};
#if defined(QG_LOCAL_COMPLETION)
  block->local_count = local_count;
  for (size_t i = 0; i < local_count; ++i)
    block->local_neighbors[i] = local_rank[i].second;
#endif
  const auto *center = static_cast<const float *>(center_ptr);
  block->neighbor_cnt = static_cast<uint32_t>(neighbor_cnt);
  block->center_floats =
      static_cast<uint32_t>(stored_center_bytes_ / sizeof(float));
  block->code_bytes = static_cast<uint32_t>(code_bytes);
  block->neighbor_bytes = static_cast<uint32_t>(neighbor_bytes_);
#if defined(QG_CENTER_QUANTIZED)
  if (compressed_center_) {
#if defined(QG_CENTER_SQ8)
#if defined(QG_ROTATED_CENTER)
    std::vector<float> rotated_storage;
    codec_.transform(center, rotated_storage);
    if (rotated_storage.size() != dimension_) {
      error = IndexError_Mismatch;
      return false;
    }
    const float *storage_values = rotated_storage.data();
#else
    const float *storage_values = center;
#endif
#if defined(QG_CENTER_SHIFT)
    if (center_shift_.size() != dimension_) {
      error = IndexError_Mismatch;
      return false;
    }
    float projection, projection_error;
    const bool encoded = SQ8Center::encode_shifted(
        storage_values, center_shift_.data(), dimension_,
        reinterpret_cast<uint8_t *>(const_cast<float *>(block->center())),
        block->center_error, block->center_norm2, block->center_offset,
        block->center_scale, projection, projection_error);
#if defined(QG_CENTER_CONTROL_VARIATE)
    block->center_projection = projection;
    block->center_projection_error = projection_error;
#endif
#if defined(QG_INTEGER_CENTER)
    if (encoded) {
      block->integer_anchor_projection = projection;
      IntegerCenter::annotate(
          storage_values, center_shift_.data(),
          reinterpret_cast<const uint8_t *>(block->center()), dimension_,
          block->center_offset, block->center_scale, block->center_error,
          block->center_residual_norm, block->center_coded_norm,
          block->center_mean_dot);
    }
#endif
    if (!encoded) {
#else
    if (!SQ8Center::encode(
            storage_values, dimension_,
            reinterpret_cast<uint8_t *>(const_cast<float *>(block->center())),
            block->center_error, block->center_norm2, block->center_offset,
            block->center_scale)) {
#endif
      error = IndexError_Mismatch;
      return false;
    }
#else
    if (!HalfCenter::encode(
            center, dimension_,
            reinterpret_cast<uint16_t *>(const_cast<float *>(block->center())),
            block->center_error, block->center_norm2)) {
      error = IndexError_Mismatch;
      return false;
    }
#endif
  } else
#endif
    std::copy(center,
              center +
#if defined(QG_OMIT_COSINE_NORM)
                  (cosine_ ? dimension_ : stored_dimension_),
#else
                stored_dimension_,
#endif
              const_cast<float *>(block->center()));
  std::copy(scratch.neighbors.begin(), scratch.neighbors.end(),
            const_cast<NodeId *>(block->neighbors()));

  std::fill_n(const_cast<char *>(block->codes()), code_bytes, char{0});
  if (neighbor_cnt == 0) {
    return true;
  }

  const void *encoding_center = center_ptr;
#if defined(QG_REANCHOR) && defined(QG_CENTER_QUANTIZED) && \
    !defined(QG_INTEGER_ORIGINAL_ANCHOR)
  std::vector<float> reconstructed_center;
  if (compressed_center_) {
    reconstructed_center.resize(dimension_);
#if defined(QG_CENTER_SQ8)
#if defined(QG_CENTER_SHIFT)
    SQ8Center::decode_shifted(
        reinterpret_cast<const uint8_t *>(block->center()),
        center_shift_.data(), dimension_, reconstructed_center.data(),
        block->center_offset, block->center_scale);
#else
    SQ8Center::decode(reinterpret_cast<const uint8_t *>(block->center()),
                      dimension_, reconstructed_center.data(),
                      block->center_offset, block->center_scale);
#endif
#else
    HalfCenter::decode(reinterpret_cast<const uint16_t *>(block->center()),
                       dimension_, reconstructed_center.data());
#endif
    encoding_center = reconstructed_center.data();
  }
#endif
#if defined(QG_ROTATED_CENTER)
  if (compressed_center_)
    scratch.encoded_center = reconstructed_center;
  else
#endif
    codec_.transform(encoding_center, scratch.encoded_center);
  scratch.encoded.assign(batch_size * dim, 0.0f);
  for (size_t offset = 0; offset < neighbor_cnt; offset += batch_size) {
    const size_t count = std::min(batch_size, neighbor_cnt - offset);
    for (size_t i = 0; i < count; ++i) {
      typename Graph::Vector neighbor;
      const int ret =
          entity.get_vector(scratch.neighbors[offset + i], neighbor);
      if (ret != 0) {
        error = ret;
        return false;
      }
      if (neighbor.data() == nullptr) {
        error = IndexError_Mismatch;
        return false;
      }
      codec_.transform(neighbor.data(), scratch.vector);
      std::copy(scratch.vector.begin(), scratch.vector.end(),
                scratch.encoded.begin() + i * dim);
    }
    codec_.encode(scratch.encoded_center.data(), scratch.encoded.data(), count,
                  const_cast<char *>(block->codes()) +
                      (offset / batch_size) * batch_bytes);
  }
  return true;
}

template <class Codec>
template <class Graph>
int QuantizedGraph<Codec>::prebuild(const Graph &entity, size_t doc_cnt,
                                    size_t threads) {
#if defined(QG_INTEGER_CENTER)
  if (compressed_center_ &&
      (dimension_ != 768 || !__builtin_cpu_supports("avx512vnni")))
    return IndexError_Mismatch;
  static_assert(sizeof(Block) <= Block::kCenterOffset,
                "Block header overlaps center");
#endif
#if defined(QG_CENTER_SHIFT)
  if (compressed_center_) {
    const char *path = std::getenv("QG_CENTER_MEAN_FILE");
    if (!path) return IndexError_Mismatch;
    FILE *file = std::fopen(path, "rb");
    if (!file) return IndexError_Mismatch;
    center_shift_.resize(dimension_);
    const size_t count =
        std::fread(center_shift_.data(), sizeof(float), dimension_, file);
    const int extra = std::fgetc(file);
    const bool io_error = std::ferror(file);
    std::fclose(file);
    if (count != dimension_ || extra != EOF || io_error)
      return IndexError_Mismatch;
    center_shift_norm2_ = 0;
    for (float v : center_shift_) {
      if (!std::isfinite(v)) return IndexError_Mismatch;
      center_shift_norm2_ += double(v) * v;
    }
  }
#endif
#if defined(QG_ROTATED_CENTER)
  if (compressed_center_ && (dimension_ != 768 || codec_.encoded_dim() != 768))
    return IndexError_Mismatch;
#endif
  const size_t nthreads =
      threads == 0 ? std::max(1u, std::thread::hardware_concurrency())
                   : threads;
  std::unique_ptr<char[], ArenaDeleter> arena;
  const size_t total = stride_ * doc_cnt;
  if (total > 0) {
    void *mem =
        ::operator new(total, std::align_val_t{ArenaDeleter::kAlignment});
#if defined(__linux__)
    // Hugepages for the whole index, as upstream SymphonyQG's arena: random
    // block access then walks one huge PTE instead of two 4K entries.
    ::madvise(mem, total, MADV_HUGEPAGE);
#endif
    arena.reset(static_cast<char *>(mem));
  }
  std::vector<std::vector<double>> centroid_parts(
      nthreads, std::vector<double>(dimension_, 0.0));
  std::vector<size_t> valid_counts(nthreads, 0);
  std::atomic<NodeId> next_id{0};
  std::atomic<int> first_error{0};

  auto worker = [&](size_t part) {
    Scratch scratch;
    auto &centroid = centroid_parts[part];
    while (first_error.load(std::memory_order_relaxed) == 0) {
      const NodeId id = next_id.fetch_add(1, std::memory_order_relaxed);
      if (id >= doc_cnt) break;
      auto *slot = reinterpret_cast<Block *>(arena.get() +
                                             static_cast<size_t>(id) * stride_);
      if (!entity.valid(id)) {
        // Deleted ids never become expandable (blocks only store valid keys),
        // but leave their slots as defined empty blocks instead of garbage.
        auto *empty = new (slot) Block{};
        empty->center_floats =
            static_cast<uint32_t>(stored_center_bytes_ / sizeof(float));
        empty->neighbor_bytes = static_cast<uint32_t>(neighbor_bytes_);
        std::fill_n(const_cast<float *>(empty->center()),
                    stored_center_bytes_ / sizeof(float), 0.0f);
        continue;
      }
      typename Graph::Vector raw;
      int ret = entity.get_vector(id, raw);
      if (ret != 0) {
        first_error.store(ret);
        return;
      }
      if (raw.data() == nullptr) {
        first_error.store(IndexError_Mismatch);
        return;
      }
      const auto *values = static_cast<const float *>(raw.data());
      for (size_t d = 0; d < dimension_; ++d) centroid[d] += values[d];
      ++valid_counts[part];
      int error = 0;
      if (!build_block(entity, id, raw.data(), slot, scratch, error)) {
        first_error.store(error);
        return;
      }
    }
  };

  {
    std::vector<std::thread> pool;
    for (size_t i = 0; i < nthreads; ++i) pool.emplace_back(worker, i);
    for (auto &t : pool) t.join();
  }
  if (first_error.load() != 0) {
    return first_error.load();
  }

  std::vector<double> centroid(dimension_, 0.0);
  for (const auto &part : centroid_parts)
    for (size_t d = 0; d < dimension_; ++d) centroid[d] += part[d];

  const size_t valid_count =
      std::accumulate(valid_counts.begin(), valid_counts.end(), size_t{0});
  if (valid_count != 0) {
    for (double &value : centroid) value /= valid_count;
  }

  // Entry point: level-0 node nearest the corpus centroid, matching upstream
  // SymphonyQG, whose single-level search starts there.
  std::atomic<NodeId> next_entry_id{0};
  std::vector<NodeId> best_ids(nthreads, kInvalidNodeId);
  std::vector<double> best_dists(nthreads, std::numeric_limits<double>::max());
  auto entry_worker = [&](size_t part) {
    while (true) {
      const NodeId id = next_entry_id.fetch_add(1, std::memory_order_relaxed);
      if (id >= doc_cnt) break;
      if (!entity.valid(id)) continue;
      typename Graph::Vector raw;
      if (entity.get_vector(id, raw) != 0 || raw.data() == nullptr) continue;
      const auto *values = static_cast<const float *>(raw.data());
      double dist = 0.0;
      for (size_t d = 0; d < dimension_; ++d) {
        const double diff = values[d] - centroid[d];
        dist += diff * diff;
      }
      if (dist < best_dists[part]) {
        best_dists[part] = dist;
        best_ids[part] = id;
      }
    }
  };
  {
    std::vector<std::thread> pool;
    for (size_t i = 0; i < nthreads; ++i) pool.emplace_back(entry_worker, i);
    for (auto &t : pool) t.join();
  }
  entry_ = kInvalidNodeId;
  double best = std::numeric_limits<double>::max();
  for (size_t i = 0; i < nthreads; ++i) {
    if (best_ids[i] != kInvalidNodeId && best_dists[i] < best) {
      best = best_dists[i];
      entry_ = best_ids[i];
    }
  }

  // Published under the caller's exclusive lock; searches only
  // start afterwards, so reads need no synchronization.
#if defined(QG_PROJECTION_ROUTER)
  const int route_rc =
      router_.build(entity, codec_, centroid, doc_cnt, dimension_);
  if (route_rc) return route_rc;
#endif
  lazy_blocks_.clear();
  arena_ = std::move(arena);
  arena_count_ = static_cast<NodeId>(doc_cnt);
#if defined(QG_TOPOLOGY_STATS)
  // Diagnostic only, before any query timing. Export a BFS tree for a possible
  // later mandatory-backbone experiment; no query or GT is read here.
  if (const char *path = std::getenv("QG_TOPOLOGY_FILE")) {
    std::vector<uint32_t> indegree(doc_cnt, 0), depth(doc_cnt, UINT32_MAX),
        parent(doc_cnt, UINT32_MAX), queue;
    size_t edges = 0;
    uint32_t max_indegree = 0, max_depth = 0;
    for (NodeId id = 0; id < doc_cnt; ++id)
      if (entity.valid(id)) {
        const Block *b = slot(id);
        for (size_t j = 0; j < b->neighbor_cnt; ++j) {
          NodeId v = b->neighbors()[j];
          if (v < doc_cnt) {
            ++indegree[v];
            ++edges;
          }
        }
      }
    if (entry_ < doc_cnt) {
      depth[entry_] = 0;
      queue.push_back(entry_);
    }
    for (size_t pos = 0; pos < queue.size(); ++pos) {
      NodeId id = queue[pos];
      const Block *b = slot(id);
      for (size_t j = 0; j < b->neighbor_cnt; ++j) {
        NodeId v = b->neighbors()[j];
        if (v < doc_cnt && depth[v] == UINT32_MAX) {
          depth[v] = depth[id] + 1;
          parent[v] = id;
          max_depth = std::max(max_depth, depth[v]);
          queue.push_back(v);
        }
      }
    }
    size_t zero = 0;
    for (NodeId id = 0; id < doc_cnt; ++id)
      if (entity.valid(id)) {
        zero += indegree[id] == 0;
        max_indegree = std::max(max_indegree, indegree[id]);
      }
    FILE *f = std::fopen(path, "w");
    if (!f) return IndexError_Mismatch;
    std::fprintf(f,
                 "{\"nodes\":%zu,\"valid_nodes\":%zu,\"edges\":%zu,\"entry_"
                 "node_id\":%llu,\"reachable\":%zu,\"zero_indegree\":%zu,\"max_"
                 "indegree\":%u,\"max_bfs_depth\":%u}\n",
                 doc_cnt, valid_count, edges, (unsigned long long)entry_,
                 queue.size(), zero, max_indegree, max_depth);
    std::fclose(f);
    if (const char *keys = std::getenv("QG_UNREACHABLE_KEYS_FILE")) {
      f = std::fopen(keys, "w");
      if (!f) return IndexError_Mismatch;
      for (NodeId id = 0; id < doc_cnt; ++id)
        if (entity.valid(id) && depth[id] == UINT32_MAX)
          std::fprintf(f, "%llu\n", (unsigned long long)entity.key(id));
      std::fclose(f);
    }
    if (const char *tree = std::getenv("QG_BFS_PARENT_FILE")) {
      f = std::fopen(tree, "wb");
      if (!f) return IndexError_Mismatch;
      size_t written =
          std::fwrite(parent.data(), sizeof(uint32_t), parent.size(), f);
      std::fclose(f);
      if (written != parent.size()) return IndexError_Mismatch;
    }
  }
#endif
  return 0;
}

template <class Codec>
template <class Graph>
const typename QuantizedGraph<Codec>::Block *QuantizedGraph<Codec>::get_block(
    const Graph &entity, NodeId id) const {
  if (id < arena_count_) {
    return slot(id);
  }

  // Nodes added after prebuild (e.g. following clear()) are built lazily into
  // standalone fixed-stride blocks.
  std::shared_ptr<const Block> block;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = lazy_blocks_.find(id);
    if (it != lazy_blocks_.end()) {
      block = it->second;
    }
  }
  if (block == nullptr) {
    typename Graph::Vector raw;
    int ret = entity.get_vector(id, raw);
    if (ret != 0 || raw.data() == nullptr) {
      return nullptr;
    }
    void *mem = ::operator new(stride_, std::align_val_t{64});
    auto *built = new (mem) Block{};
    Scratch scratch;
    int error = 0;
    if (!build_block(entity, id, raw.data(), built, scratch, error)) {
      BlockDeleter()(built);
      return nullptr;
    }
    // Concurrent readers may have built the same block. Publish one copy.
    std::lock_guard<std::mutex> lock(mutex_);
    block =
        lazy_blocks_
            .emplace(id, std::shared_ptr<const Block>(built, BlockDeleter()))
            .first->second;
  }
  return block.get();
}

template <class Codec>
template <class Graph, class Context>
int QuantizedGraph<Codec>::search(NodeId entry, const Graph &entity,
                                  Context &ctx) const {
#if defined(QG_PROFILE)
  const auto profile_start = std::chrono::steady_clock::now();
#endif
  auto &results = ctx.reset_results();
  results.clear();
  if (entry == kInvalidNodeId || ctx.topk() == 0) return 0;
  if (!entity.valid(entry)) return IndexError_Mismatch;
  auto &state = search_state();
  auto &visited = state.visited;
  visited.reset(results.limit());
#if defined(QG_BUCKET_DOMINANCE)
  state.dominance.reset();
#endif
#if defined(QG_FIRST_ESTIMATE)
  state.discovered.reset(std::max(size_t{64}, results.limit() * 4));
  state.discovered.set_visited(entry);
#endif
  auto &beam = state.beam;
  beam.reset(std::max(size_t{1}, results.limit()));
  beam.insert(entry, std::numeric_limits<float>::max());
#if defined(QG_RESULT_TOPK)
  // Navigation capacity was already initialized; only exported results shrink.
  results.limit(ctx.topk());
#endif
  auto &batch_query = state.query;
  codec_.prepare_query(ctx.query(), batch_query);
#if defined(QG_INTEGER_CENTER)
  if (compressed_center_ &&
      !IntegerCenter::prepare(static_cast<const float *>(ctx.query()),
                              center_shift_.data(), dimension_,
                              state.integer_query))
    return IndexError_Mismatch;
#endif
#if defined(QG_ROTATED_CENTER)
  if (compressed_center_ && !QuantizedGraphRouter::rotated(batch_query, 0))
    return IndexError_Mismatch;
#endif
#if defined(QG_PROJECTION_ROUTER)
  if (!router_.empty()) {
    const float *rotated = QuantizedGraphRouter::rotated(batch_query, 0);
    if (!rotated) {
      codec_.transform(ctx.query(), state.route_query);
      rotated = state.route_query.data();
    }
    const size_t route_dim =
        std::getenv("QG_ROUTE_DIM")
            ? std::strtoul(std::getenv("QG_ROUTE_DIM"), nullptr, 10)
            : 64;
    const size_t route_count =
        std::getenv("QG_ROUTE_COUNT")
            ? std::strtoul(std::getenv("QG_ROUTE_COUNT"), nullptr, 10)
            : 256;
    const size_t route_short =
        std::getenv("QG_ROUTE_SHORTLIST")
            ? std::strtoul(std::getenv("QG_ROUTE_SHORTLIST"), nullptr, 10)
            : 4;
    const int rc = router_.choose(ctx, entity, rotated, entry, route_dim,
                                  route_count, route_short, state.route_scores);
    if (rc) return rc;
    beam.reset(std::max(size_t{1}, results.limit()));
    beam.insert(entry, std::numeric_limits<float>::max());
  }
#endif
#if defined(QG_CENTER_QUANTIZED)
  double query_norm2_double = 0;
  const float *query_values = static_cast<const float *>(ctx.query());
#if defined(QG_INTEGER_CENTER)
  if (compressed_center_)
    query_norm2_double = state.integer_query.norm * state.integer_query.norm;
  else
#endif
    for (size_t d = 0; d < dimension_; ++d)
      query_norm2_double += double(query_values[d]) * query_values[d];
  const float query_norm2 = std::nextafter(
      float(query_norm2_double), std::numeric_limits<float>::infinity());
#endif
#if defined(QG_CENTER_CONTROL_VARIATE)
  double shift_dot = 0;
  if (compressed_center_)
    for (size_t j = 0; j < dimension_; ++j)
      shift_dot += double(query_values[j]) * center_shift_[j];
  const float center_alpha =
      center_shift_norm2_ > 0 ? float(shift_dot / center_shift_norm2_) : 0.f;
  double residual_norm = 0;
  if (compressed_center_)
    for (size_t j = 0; j < dimension_; ++j) {
      const double r =
          double(query_values[j]) - double(center_alpha) * center_shift_[j];
      residual_norm += r * r;
    }
  // Padding covers FP64 formation of residuals even when cancellation is
  // strong.
  const double residual_norm_upper =
      std::sqrt(residual_norm) +
      8. * (dimension_ + 4) * std::numeric_limits<double>::epsilon() *
          (std::sqrt(query_norm2_double) +
           std::abs(double(center_alpha)) * std::sqrt(center_shift_norm2_));
  const float residual_query_norm2 =
      std::nextafter(float(residual_norm_upper * residual_norm_upper),
                     std::numeric_limits<float>::infinity());
#endif
  auto &expanded = state.expanded;
  expanded.clear();
#if defined(QG_DEFERRED_COSINE)
  state.refine_candidates.clear();
  state.refine_best.clear();
#endif
#if defined(QG_TAIL_REFINE)
  const size_t expansion_cap =
      std::getenv("QG_EXPANSIONS")
          ? std::strtoul(std::getenv("QG_EXPANSIONS"), nullptr, 10)
          : 1000000;
  const size_t tail_budget =
      std::getenv("QG_REFINE")
          ? std::strtoul(std::getenv("QG_REFINE"), nullptr, 10)
          : 0;
#endif
#if defined(QG_ADAPTIVE)
  struct StopState {
    std::vector<float> &best;
    size_t k, stale = 0, expanded = 0;
#if defined(QG_GAP_FEEDBACK)
    float gap_blend = std::getenv("QG_GAP_BLEND")
                          ? std::strtof(std::getenv("QG_GAP_BLEND"), nullptr)
                          : .25f;
#endif
#if defined(QG_ERROR_FEEDBACK)
    float popped_estimate = std::numeric_limits<float>::quiet_NaN(),
          error_mean = 0;
    size_t error_count = 0;
    float error_blend =
        std::getenv("QG_ERROR_BLEND")
            ? std::strtof(std::getenv("QG_ERROR_BLEND"), nullptr)
            : .5f;
#endif
#if defined(QG_RUNTIME_ADAPTIVE)
    size_t patience =
        std::getenv("QG_PATIENCE")
            ? std::strtoul(std::getenv("QG_PATIENCE"), nullptr, 10)
            : QG_STOP_PATIENCE;
    float margin = std::getenv("QG_MARGIN")
                       ? std::strtof(std::getenv("QG_MARGIN"), nullptr)
                       : QG_STOP_MARGIN;
    size_t cap = std::getenv("QG_CAP")
                     ? std::strtoul(std::getenv("QG_CAP"), nullptr, 10)
                     : 1000000;
#endif
    void observe(float d, bool eligible) {
      ++expanded;
#if defined(QG_ERROR_FEEDBACK)
      if (expanded > 1 && std::isfinite(popped_estimate) && d > 1e-6f) {
        const float residual =
            std::clamp((d - popped_estimate) / d, -.25f, .25f);
        error_mean += .125f * (residual - error_mean);
        ++error_count;
      }
#endif
      if (eligible && (best.size() < k || d < best.back())) {
        best.insert(std::lower_bound(best.begin(), best.end(), d), d);
        if (best.size() > k) best.pop_back();
        stale = 0;
      } else
        ++stale;
    }
#if defined(QG_WINDOW)
    bool admit(float lower) const {
#if defined(QG_RUNTIME_ADAPTIVE)
      const size_t p = patience;
      const float m = margin;
#else
      const size_t p = QG_STOP_PATIENCE;
      const float m = QG_STOP_MARGIN;
#endif
      return p != 0 || best.size() != k || expanded < QG_STOP_MIN ||
             lower - best.back() < m * std::max(best.back(), 1e-6f);
    }
#endif
    bool stop(float next) const {
#if defined(QG_RUNTIME_ADAPTIVE)
      float calibrated_margin = margin;
#if defined(QG_GAP_FEEDBACK)
      if (best.size() == k && k >= 2) {
        const float gap = std::clamp(
            (best.back() - best[k / 2 - 1]) / std::max(best.back(), 1e-6f), 0.f,
            1.f);
        calibrated_margin = std::clamp(
            calibrated_margin + gap_blend * (.10f - gap), -.10f, .20f);
      }
#endif
#if defined(QG_ERROR_FEEDBACK)
      if (error_count >= 8) calibrated_margin -= error_blend * error_mean;
#endif
      return best.size() == k && expanded >= QG_STOP_MIN &&
             (expanded >= cap ||
              (stale >= patience &&
               next - best.back() >=
                   calibrated_margin * std::max(best.back(), 1e-6f)));
#else
      return best.size() == k && expanded >= QG_STOP_MIN &&
             stale >= QG_STOP_PATIENCE &&
             next - best.back() >=
                 QG_STOP_MARGIN * std::max(best.back(), 1e-6f);
#endif
    }
#if defined(QG_CHEAP_WINDOW)
#if defined(QG_ERROR_FEEDBACK) || defined(QG_GAP_FEEDBACK) || \
    !defined(QG_RUNTIME_ADAPTIVE)
#error Cheap window requires a constant per-query runtime margin
#endif
    float admission_cutoff(float width, float overflow_lower) const {
      if (patience != 0 || best.size() != k || expanded < QG_STOP_MIN ||
          margin < 0 || !std::isfinite(margin))
        return std::numeric_limits<float>::infinity();
      const float r = best.back(), threshold = r + margin * std::max(r, 1e-6f);
      const float pad =
          16 * std::numeric_limits<float>::epsilon() *
          (std::abs(r) + std::abs(threshold) + std::abs(width) + 1.f);
      if (!std::isfinite(threshold) || overflow_lower < threshold + pad)
        return std::numeric_limits<float>::infinity();
      // Slack admits extra candidates, avoiding a division per admission.
      // The exact stop threshold only decreases as exact Top-K improves.
      return std::max(0.f, threshold + 2 * width + pad);
    }
#endif
  };
  state.best_exact.clear();
  StopState stopping{state.best_exact, ctx.topk()};
#endif
#if defined(QG_PATH_TRACE)
  struct PathTrace {
    FILE *file;
    uint64_t serial;
    float estimate = 0;
  };
  static thread_local uint64_t path_serial = 0;
  static thread_local FILE *path_file = []() {
    const char *p = std::getenv("QG_PATH_FILE");
    return p ? std::fopen(p, "w") : nullptr;
  }();
  PathTrace path{path_file, path_serial++};
#endif
#if QG_DIAG
  struct Diagnostics {
    size_t popped = 0, pushes = 0, scored = 0, batches = 0, top1 = 0;
    double se = 0, bias = 0;
  } diag;
#endif
  struct Scan {
    enum { kVisitOnExpansion = true };
    const QuantizedGraph &index;
    const Graph &entity;
    Context &ctx;
    decltype(results) output;
    typename Codec::Query &query;
#if defined(QG_INTEGER_CENTER)
    const IntegerCenter::Query &integer_query;
#endif
    std::vector<NodeId> &expanded;
#if defined(QG_DEFERRED_COSINE)
    std::vector<std::pair<float, NodeId>> &refine_candidates;
#endif
#if defined(QG_CENTER_QUANTIZED)
    float query_norm2;
#if defined(QG_CENTER_CONTROL_VARIATE)
    float center_alpha, residual_query_norm2;
#endif
#endif
#if defined(QG_BUCKET)
    QuantizedGraphBucket *bucket;
#endif
#if defined(QG_ADAPTIVE)
    StopState *stopping;
#endif
#if defined(QG_PATH_TRACE)
    PathTrace *path;
#endif
#if QG_DIAG
    Diagnostics *diag;
#endif
    const Block *block = nullptr;
    const NodeId *neighbors = nullptr;
    const char *codes = nullptr;
    float exact = 0;
#if defined(QG_DEFERRED_ADAPTIVE)
    float interval_upper = 0;
#endif
#if defined(QG_PROFILE)
    uint64_t profile_begin_ns = 0, profile_score_ns = 0, profile_admit_ns = 0,
             profile_edges = 0;
    std::chrono::steady_clock::time_point profile_admit_start;
    void profile_admission_begin() {
      profile_admit_start = std::chrono::steady_clock::now();
    }
    void profile_admission_end() {
      profile_admit_ns +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - profile_admit_start)
              .count();
    }
#endif
#if defined(QG_CENTER_STATS)
    size_t refined_count = 0;
#endif
#if defined(QG_REANCHOR) && defined(QG_CENTER_QUANTIZED)
    float navigation_center = 0;
#endif
#if defined(QG_EXPAND_BATCH)
    void prefetch_node(NodeId id) const {
      if (id >= index.arena_count_) return;
      const char *row = reinterpret_cast<const char *>(index.slot(id));
      const size_t bytes = std::min(index.stride_, size_t{4096});
      for (size_t offset = 0; offset < bytes; offset += 64)
        __builtin_prefetch(row + offset, 0, 2);
    }
#endif
    int begin(NodeId id) {
#if defined(QG_PROFILE)
      const auto profile_t = std::chrono::steady_clock::now();
#endif
      expanded.push_back(id);
      block = index.get_block(entity, id);
      if (block == nullptr) return IndexError_Mismatch;
      neighbors = reinterpret_cast<const NodeId *>(
          reinterpret_cast<const char *>(block->center()) +
          index.stored_center_bytes_);
      codes = reinterpret_cast<const char *>(neighbors) + index.neighbor_bytes_;
#if defined(__GNUC__) || defined(__clang__)
      // Fetch the row header and the start of its packed codes while the
      // exact center distance is being computed. The layout is fixed for
      // both arena and lazy blocks, so this needs no dependent header load.
      __builtin_prefetch(block, 0, 2);
      const size_t code_offset = codes - reinterpret_cast<const char *>(block);
      const size_t bytes = std::min(
          size_t{
#if defined(QG_CODE_PREFETCH_BYTES)
              QG_CODE_PREFETCH_BYTES
#else
              4096
#endif
          },
          index.stride_ - code_offset);
      for (size_t offset = 0; offset < bytes; offset += 64)
        __builtin_prefetch(codes + offset, 0, 2);
#endif
      bool refined = true;
#if defined(QG_CENTER_QUANTIZED)
      if (index.compressed_center_) {
#if defined(QG_CENTER_SQ8)
#if defined(QG_DEFERRED_COSINE)
#if defined(QG_INTEGER_CENTER)
        exact = IntegerCenter::distance(
            integer_query, reinterpret_cast<const uint8_t *>(block->center()),
            index.dimension_, block->center_offset, block->center_scale,
            block->center_mean_dot, index.center_shift_norm2_);
#elif defined(QG_CENTER_SHIFT)
        exact = SQ8Center::cosine_distance_shifted(
            static_cast<const float *>(ctx.query()),
            reinterpret_cast<const uint8_t *>(block->center()),
            index.center_shift_.data(), index.dimension_, block->center_offset,
            block->center_scale);
#else
        exact = SQ8Center::cosine_distance(
#if defined(QG_ROTATED_CENTER)
            QuantizedGraphRouter::rotated(query, 0),
#else
            static_cast<const float *>(ctx.query()),
#endif
            reinterpret_cast<const uint8_t *>(block->center()),
            index.dimension_, block->center_offset, block->center_scale);
#endif
#else
        exact = SQ8Center::distance(
            static_cast<const float *>(ctx.query()),
            reinterpret_cast<const uint8_t *>(block->center()),
            index.dimension_, block->center_offset, block->center_scale);
#endif
#else
#if defined(QG_DEFERRED_COSINE)
        exact = HalfCenter::cosine_distance(
            static_cast<const float *>(ctx.query()),
            reinterpret_cast<const uint16_t *>(block->center()),
            index.dimension_);
#else
        exact = HalfCenter::distance(
            static_cast<const float *>(ctx.query()),
            reinterpret_cast<const uint16_t *>(block->center()),
            index.dimension_);
#endif
#endif
#if defined(QG_REANCHOR)
        navigation_center = exact;
#if defined(QG_INTEGER_CENTER) && !defined(QG_INTEGER_ORIGINAL_ANCHOR)
        // Correct the shared-mean component when the neighbor codec is
        // anchored at the decoded center. Remaining query quantization is
        // heuristic navigation error, not part of the exact-result guarantee.
        navigation_center -= block->integer_anchor_projection;
#endif
#endif
#if defined(QG_DEFERRED_COSINE)
#if defined(QG_INTEGER_CENTER)
        float lower = IntegerCenter::lower_bound(
            exact, integer_query, block->center_error,
            block->center_residual_norm, block->center_coded_norm,
            block->center_norm2, block->center_mean_dot,
            index.center_shift_norm2_, index.dimension_);
#elif defined(QG_CENTER_CONTROL_VARIATE)
        exact = std::fma(center_alpha, block->center_projection, exact);
        float lower = SQ8Center::projected_lower_bound(
            exact, block->center_error, block->center_norm2, query_norm2,
            residual_query_norm2, center_alpha, block->center_projection,
            block->center_projection_error, index.dimension_);
#else
        float lower = SQ8Center::cosine_lower_bound(
            exact, block->center_error, block->center_norm2, query_norm2,
            index.dimension_);
#endif
#if defined(QG_ROTATED_CENTER)
        lower = SQ8Center::rotated_lower_bound(
            lower, block->center_error, block->center_norm2, query_norm2);
#endif
#if defined(QG_DEFERRED_ADAPTIVE) && \
    (!defined(QG_DEFERRED_ESTIMATE_STOP) || defined(QG_CENTER_AUDIT))
        interval_upper =
            std::nextafter(float(2. * double(exact) - double(lower)),
                           std::numeric_limits<float>::infinity());
#endif
        refined = false;
        if (!ctx.excluded(id)) refine_candidates.emplace_back(lower, id);
#else
        const float lower = HalfCenter::lower_bound(
            exact, block->center_error, block->center_norm2, query_norm2,
            index.dimension_);
        refined = !ctx.excluded(id) && (stopping->best.size() < stopping->k ||
                                        lower <= stopping->best.back());
        if (refined) {
          typename Graph::Vector raw;
          int rc = entity.get_vector(id, raw);
          if (rc != 0 || raw.data() == nullptr) return IndexError_Mismatch;
          exact = ctx.distance(raw.data());
        }
#endif
#if defined(QG_CENTER_AUDIT)
        typename Graph::Vector audit_raw;
        int audit_rc = entity.get_vector(id, audit_raw);
        if (audit_rc != 0 || audit_raw.data() == nullptr)
          return IndexError_Mismatch;
        float truth = ctx.reference_distance(audit_raw.data());
#if defined(QG_DEFERRED_COSINE)
        if (lower > truth) std::abort();
#if defined(QG_DEFERRED_ADAPTIVE)
        if (interval_upper < truth) std::abort();
#endif
#else
        if (lower > truth ||
            (!refined && !ctx.excluded(id) && truth < stopping->best.back()))
          std::abort();
#endif
#endif
      } else
#endif
        exact = ctx.distance(block->center());
#if defined(QG_BUCKET)
      if (expanded.size() == 1) bucket->set_scale(exact);
#endif
#if defined(QG_PATH_TRACE)
      if (path->file)
        std::fprintf(path->file, "%llu,%zu,%zu,%llu,%.9g,%.9g\n",
                     (unsigned long long)path->serial, output.limit(),
                     expanded.size(), (unsigned long long)entity.key(id), exact,
                     path->estimate);
#endif
#if defined(QG_ADAPTIVE)
#if defined(QG_DEFERRED_ADAPTIVE)
#if defined(QG_DEFERRED_ESTIMATE_STOP)
      // Heuristic navigation threshold; interval bounds still certify terminal
      // Top-K over expanded nodes, not navigation/global recall.
      stopping->observe(exact, !ctx.excluded(id));
#else
      stopping->observe(index.compressed_center_ ? interval_upper : exact,
                        !ctx.excluded(id));
#endif
#else
      stopping->observe(exact, refined && !ctx.excluded(id));
#endif
#endif
#if defined(QG_CENTER_STATS)
      refined_count += refined;
#endif
      if (refined && !ctx.excluded(id)) {
#if defined(QG_RESULT_GATE)
        // StopState has observed this distance. Strictly worse values cannot
        // belong to the exact Top-K of the expanded set.
        if (stopping->best.size() < stopping->k ||
            exact <= stopping->best.back())
#endif
          output.emplace(id, exact);
      }
      ctx.on_expand();
#if defined(QG_PROFILE)
      profile_begin_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - profile_t)
                              .count();
#endif
      return 0;
    }
    size_t neighbor_count() const {
      return block->neighbor_cnt;
    }
    size_t batch_size() const {
      return Codec::kBatchSize;
    }
    NodeId neighbor(size_t i) const {
      return neighbors[i];
    }
    void prepare_batch(size_t) {}
    void stage(NodeId, size_t, size_t) {}
    int score(size_t offset, const NodeId *ids, size_t count,
              float *distances) {
#if defined(QG_PROFILE)
      const auto profile_t = std::chrono::steady_clock::now();
      profile_edges += count;
#endif
      float center_score = exact;
#if defined(QG_REANCHOR) && defined(QG_CENTER_QUANTIZED)
      if (index.compressed_center_) center_score = navigation_center;
#endif
      index.codec_.scan(
          codes + (offset / Codec::kBatchSize) * index.codec_.batch_bytes(),
          query, center_score, distances);
#if QG_EXPERIMENT == 1 || QG_EXPERIMENT == 3 || QG_DIAG
      std::array<float, Codec::kBatchSize> truth;
      for (size_t i = 0; i < count; ++i) {
        typename Graph::Vector raw;
        const int rc = entity.get_vector(ids[i], raw);
        if (rc != 0 || raw.data() == nullptr) return IndexError_Mismatch;
        truth[i] = ctx.reference_distance(raw.data());
#if QG_DIAG
        const double err = double(distances[i]) - truth[i];
        ++diag->scored;
        diag->se += err * err;
        diag->bias += err;
#endif
      }
#if QG_DIAG
      ++diag->batches;
      const size_t estimated =
          std::min_element(distances, distances + count) - distances;
      const size_t real =
          std::min_element(truth.begin(), truth.begin() + count) -
          truth.begin();
      diag->top1 += estimated == real;
#endif
#if QG_EXPERIMENT == 1 || QG_EXPERIMENT == 3
      std::copy_n(truth.data(), count, distances);
#endif
#endif
#if defined(QG_PROFILE)
      profile_score_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - profile_t)
                              .count();
#endif
      return 0;
    }
  } scan{*this,
         entity,
         ctx,
         results,
         batch_query,
#if defined(QG_INTEGER_CENTER)
         state.integer_query,
#endif
         expanded
#if defined(QG_DEFERRED_COSINE)
         ,
         state.refine_candidates
#endif
#if defined(QG_CENTER_QUANTIZED)
         ,
         query_norm2
#if defined(QG_CENTER_CONTROL_VARIATE)
         ,
         center_alpha,
         residual_query_norm2
#endif
#endif
#if defined(QG_BUCKET)
         ,
         &beam
#endif
#if defined(QG_ADAPTIVE)
         ,
         &stopping
#endif
#if defined(QG_PATH_TRACE)
         ,
         &path
#endif
#if QG_DIAG
         ,
         &diag
#endif
  };
  struct Frontier {
    const QuantizedGraph &index;
#if defined(QG_BUCKET)
    QuantizedGraphBucket &beam;
#elif defined(QG_HEAP)
    AdaptiveHeap &beam;
#else
    QuantizedGraphBeam &beam;
#endif
    Context &ctx;
#if defined(QG_TAIL_REFINE)
    const std::vector<NodeId> *expanded;
    size_t expansion_cap;
#endif
#if defined(QG_BUCKET_DOMINANCE)
#if defined(QG_DOMINANCE_CACHE)
    QuantizedGraphDominanceCache *dominance;
#else
    QuantizedGraphBucketDominance *dominance;
#endif
#endif
#if defined(QG_FIRST_ESTIMATE)
    LocalVisitSet *discovered;
#endif
#if defined(QG_ADAPTIVE)
    StopState *stopping;
    LocalVisitSet *visited;
#endif
#if defined(QG_PATH_TRACE)
    PathTrace *path;
#endif
#if QG_DIAG
    Diagnostics *diag;
#endif
    NodeId prefetched = kInvalidNodeId;
#if defined(QG_CHEAP_WINDOW)
    mutable float cutoff = std::numeric_limits<float>::infinity();
#endif
    bool has_next() const {
#if defined(QG_TAIL_REFINE)
      if (expanded->size() >= expansion_cap) return false;
#endif
#if defined(QG_ADAPTIVE)
      while (beam.has_next() && visited->visited(beam.next_id_at(0)))
        beam.pop();
      if (beam.has_next() && stopping->stop(beam.next_distance())) return false;
#if defined(QG_CHEAP_WINDOW)
      cutoff = stopping->admission_cutoff(
          beam.width(), float(QuantizedGraphBucket::N - 2) * beam.width());
#endif
#endif
      return beam.has_next() && !ctx.reach_scan_limit();
    }
    NodeId pop() {
#if defined(QG_ERROR_FEEDBACK)
      stopping->popped_estimate = beam.next_distance() + .5f * beam.width();
#endif
#if defined(QG_PATH_TRACE)
      path->estimate = beam.next_distance();
#endif
#if QG_DIAG
      ++diag->popped;
#endif
      return beam.pop();
    }
    bool accepts(float distance) const {
#if defined(QG_CHEAP_WINDOW)
      if (distance > cutoff) {
#if defined(QG_WINDOW_AUDIT)
        if (!stopping->stop(beam.lower_bound(distance))) std::abort();
#endif
        return false;
      }
#endif
#if defined(QG_WINDOW)
      if (!std::isfinite(distance) ||
          !stopping->admit(beam.lower_bound(distance)))
        return false;
#endif
      return !beam.is_full(distance) && std::isfinite(distance);
    }
    void push(NodeId id, float distance) {
#if defined(QG_SHARED_DOMINANCE)
      if (!visited->admit(id, beam.locate(distance))) return;
#endif
#if defined(QG_BUCKET_DOMINANCE)
      if (!dominance->admit(id, beam.locate(distance))) return;
#endif
#if defined(QG_FIRST_ESTIMATE)
      if (discovered->visited(id)) return;
      discovered->set_visited(id);
#endif
#if QG_DIAG
      ++diag->pushes;
#endif
#if QG_EXPERIMENT == 2 || QG_EXPERIMENT == 3
      beam.insert_unique(id, distance);
#else
      beam.insert(id, distance);
#endif
#if !defined(QG_PREFETCH_BATCH)
      prefetch_next();
#endif
    }
    void prefetch_next() {
#if (defined(__GNUC__) || defined(__clang__)) && \
    !defined(QG_DISABLE_FRONTIER_PREFETCH)
      if (beam.has_next()) {
        const NodeId next = beam.next_id_at(0);
        // Only prefetch published arena slots. Lazy blocks require a lookup
        // under a mutex and are not safe to speculate about here.
        if (next < index.arena_count_ && next != prefetched) {
          prefetched = next;
          const char *center =
              reinterpret_cast<const char *>(index.slot(next)->center());
          const size_t bytes = std::min(
              size_t{
#if defined(QG_PREFETCH_BYTES)
                  QG_PREFETCH_BYTES
#else
                  640
#endif
              },
              index.stride_ - Block::kCenterOffset);
          for (size_t offset = 0; offset < bytes; offset += 64) {
            __builtin_prefetch(center + offset, 0, 2);
          }
        }
      }
#endif
    }
  } frontier{*this,
             beam,
             ctx
#if defined(QG_TAIL_REFINE)
             ,
             &expanded,
             expansion_cap
#endif
#if defined(QG_BUCKET_DOMINANCE)
             ,
             &state.dominance
#endif
#if defined(QG_FIRST_ESTIMATE)
             ,
             &state.discovered
#endif
#if defined(QG_ADAPTIVE)
             ,
             &stopping,
             &visited
#endif
#if defined(QG_PATH_TRACE)
             ,
             &path
#endif
#if QG_DIAG
             ,
             &diag
#endif
  };
#if defined(QG_PROFILE)
  const auto profile_init_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - profile_start)
          .count();
#endif
  const int search_ret = SearchGraph(scan, frontier, visited, state.scratch);
  if (search_ret != 0) return search_ret;
#if defined(QG_LOCAL_COMPLETION)
#if defined(QG_CENTER_QUANTIZED)
#error Local completion prototype expects FP32 center rows
#endif
  const size_t local_seeds =
      std::getenv("QG_LOCAL_SEEDS")
          ? std::strtoul(std::getenv("QG_LOCAL_SEEDS"), nullptr, 10)
          : 1;
  const size_t local_neighbors =
      std::getenv("QG_LOCAL_NEIGHBORS")
          ? std::strtoul(std::getenv("QG_LOCAL_NEIGHBORS"), nullptr, 10)
          : 16;
  std::vector<std::pair<float, NodeId>> seeds;
  for (const auto &item : results) seeds.emplace_back(item.second, item.first);
  std::sort(seeds.begin(), seeds.end());
  for (size_t i = 0; i < std::min(local_seeds, seeds.size()); ++i) {
    const Block *seed = get_block(entity, seeds[i].second);
    if (!seed) return IndexError_Mismatch;
    for (size_t j = 0; j < std::min(local_neighbors, size_t(seed->local_count));
         ++j) {
      if (ctx.reach_scan_limit()) break;
      const NodeId candidate = seed->local_neighbors[j];
      if (visited.visited(candidate) || !entity.valid(candidate)) continue;
      visited.set_visited(candidate);
      if (ctx.excluded(candidate)) continue;
      const Block *target = get_block(entity, candidate);
      if (!target) return IndexError_Mismatch;
      results.emplace(candidate, ctx.distance(target->center()));
    }
  }
#endif
#if defined(QG_DEFERRED_COSINE)
  // Exact Top-K of the expanded set, via conservative interval ordering.
  // Navigational estimates still use reconstructed-center coordinates.
  auto &refine = state.refine_candidates;
  auto &best = state.refine_best;
  std::sort(refine.begin(), refine.end());
#if defined(QG_REFINE_BATCH)
  static_assert(QG_REFINE_BATCH >= 1 && QG_REFINE_BATCH <= 16);
  for (size_t first = 0; first < refine.size(); first += QG_REFINE_BATCH) {
#if !defined(QG_DEFERRED_FULL)
    if (best.size() == ctx.topk() && refine[first].first > best.back()) break;
#endif
    if (ctx.reach_scan_limit()) break;
    std::array<typename Graph::Vector, QG_REFINE_BATCH> raw;
    size_t count = 0;
    for (; count < QG_REFINE_BATCH && first + count < refine.size(); ++count) {
      const auto &candidate = refine[first + count];
#if !defined(QG_DEFERRED_FULL)
      if (best.size() == ctx.topk() && candidate.first > best.back()) break;
#endif
      int ret = entity.get_vector(candidate.second, raw[count]);
      if (ret) return ret;
      if (raw[count].data() == nullptr) return IndexError_Mismatch;
      const char *ptr = static_cast<const char *>(raw[count].data());
      for (size_t off = 0; off < stored_dimension_ * sizeof(float); off += 64)
        __builtin_prefetch(ptr + off, 0, 2);
    }
    for (size_t i = 0; i < count; ++i) {
      const auto &candidate = refine[first + i];
#if !defined(QG_DEFERRED_FULL)
      if (best.size() == ctx.topk() && candidate.first > best.back()) break;
#endif
      if (ctx.reach_scan_limit()) break;
      const float exact = ctx.distance(raw[i].data());
      results.emplace(candidate.second, exact);
      best.insert(std::lower_bound(best.begin(), best.end(), exact), exact);
      if (best.size() > ctx.topk()) best.pop_back();
#if defined(QG_CENTER_STATS)
      ++scan.refined_count;
#endif
    }
  }
#else
  for (const auto &candidate : refine) {
#if !defined(QG_DEFERRED_FULL)
    if (best.size() == ctx.topk() && candidate.first > best.back()) break;
#endif
    if (ctx.reach_scan_limit()) break;
    typename Graph::Vector raw;
    const int ret = entity.get_vector(candidate.second, raw);
    if (ret != 0) return ret;
    if (raw.data() == nullptr) return IndexError_Mismatch;
    const float exact = ctx.distance(raw.data());
    results.emplace(candidate.second, exact);
    best.insert(std::lower_bound(best.begin(), best.end(), exact), exact);
    if (best.size() > ctx.topk()) best.pop_back();
#if defined(QG_CENTER_STATS)
    ++scan.refined_count;
#endif
  }
#endif
#endif
#if defined(QG_TAIL_REFINE)
  // Separate navigation from result collection: after navigation stops,
  // refine remaining frontier candidates without scanning their neighbors.
  // For a fixed navigation trajectory this can only improve exact Top-K
  // distances; it does not guarantee recall under a shorter trajectory.
  size_t tail_count = 0;
#if defined(QG_TAIL_BATCH)
#if defined(QG_CENTER_QUANTIZED)
#error Tail cached-center batch requires FP32 stored centers
#endif
  std::vector<NodeId> tail_ids;
  tail_ids.reserve(std::min(tail_budget, size_t{128}));
#endif
  while (beam.has_next() && tail_count < tail_budget &&
         !ctx.reach_scan_limit()) {
    const NodeId candidate = beam.pop();
    if (visited.visited(candidate) || !entity.valid(candidate)) continue;
    visited.set_visited(candidate);
    if (ctx.excluded(candidate)) continue;
#if defined(QG_TAIL_BATCH)
    tail_ids.push_back(candidate);
    if (candidate < arena_count_) {
      const char *raw =
          reinterpret_cast<const char *>(slot(candidate)->center());
      for (size_t off = 0; off < stored_center_bytes_; off += 64)
        __builtin_prefetch(raw + off, 0, 2);
    }
#else
    typename Graph::Vector raw;
    const int ret = entity.get_vector(candidate, raw);
    if (ret != 0) return ret;
    if (raw.data() == nullptr) return IndexError_Mismatch;
    results.emplace(candidate, ctx.distance(raw.data()));
#endif
    ++tail_count;
  }
#if defined(QG_TAIL_BATCH)
  for (NodeId candidate : tail_ids) {
    if (ctx.reach_scan_limit()) break;
    const Block *node = get_block(entity, candidate);
    if (!node) return IndexError_Mismatch;
    results.emplace(candidate, ctx.distance(node->center()));
  }
#endif
#endif
#if defined(QG_CENTER_STATS)
  static thread_local FILE *center_stats = []() {
    const char *p = std::getenv("QG_CENTER_STATS_FILE");
    return p ? std::fopen(p, "w") : nullptr;
  }();
  static thread_local size_t stats_serial = 0;
  if (center_stats)
    std::fprintf(center_stats, "%zu,%u,%zu,%zu,%zu,%zu,%zu\n", stats_serial++,
                 ctx.topk(), results.limit(), expanded.size(),
                 scan.refined_count, stored_center_bytes_, stride_);
#endif
#if defined(QG_QUEUE_STATS) && defined(QG_BUCKET)
  static thread_local FILE *queue_stats = []() {
    const char *p = std::getenv("QG_STATS_FILE");
    return p ? std::fopen(p, "w") : nullptr;
  }();
  if (queue_stats)
    std::fprintf(queue_stats, "%u,%zu,%zu,%zu,%zu\n", ctx.topk(),
                 results.limit(), expanded.size(), beam.peak,
                 beam.memory_bytes());
#endif
#if QG_DIAG
  static thread_local FILE *trace = []() {
    const char *p = std::getenv("QG_TRACE");
    return p ? std::fopen(p, "a") : nullptr;
  }();
  if (trace) {
    std::fprintf(
        trace, "%d,%u,%zu,%zu,%zu,%zu,%zu,%.12g,%.12g,%zu,%zu\n", QG_EXPERIMENT,
        ctx.topk(), results.limit(), expanded.size(), diag.popped, diag.pushes,
        diag.scored, diag.scored ? std::sqrt(diag.se / diag.scored) : 0,
        diag.scored ? diag.bias / diag.scored : 0, diag.top1, diag.batches);
  }
#endif
  // As in SymphonyQG's result completion step, score unvisited neighbors if
  // duplicate estimates or filters left fewer than k eligible results. Avoid
  // building additional cached blocks and respect the exact-distance budget.
  for (NodeId id : expanded) {
    if (results.size() >= ctx.topk() || ctx.reach_scan_limit()) break;
    const auto &neighbors = entity.neighbors(id);
    for (uint32_t i = 0; i < neighbors.size(); ++i) {
      if (results.size() >= ctx.topk() || ctx.reach_scan_limit()) break;
      const NodeId candidate = neighbors[i];
      if (visited.visited(candidate) || !entity.valid(candidate)) continue;
      visited.set_visited(candidate);
      if (ctx.excluded(candidate)) continue;
      typename Graph::Vector raw;
      int ret = entity.get_vector(candidate, raw);
      if (ret != 0) return ret;
      if (raw.data() == nullptr) return IndexError_Mismatch;
      const float exact = ctx.distance(raw.data());
      results.emplace(candidate, exact);
    }
  }
#if defined(QG_PROFILE)
  const auto total = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - profile_start)
                         .count();
  static thread_local FILE *profile = []() {
    const char *p = std::getenv("QG_PROFILE_FILE");
    return p ? std::fopen(p, "w") : nullptr;
  }();
  if (profile)
    std::fprintf(profile, "%zu,%zu,%lld,%lld,%llu,%llu,%llu,%llu\n",
                 expanded.size(), results.limit(), (long long)total,
                 (long long)profile_init_ns,
                 (unsigned long long)scan.profile_begin_ns,
                 (unsigned long long)scan.profile_score_ns,
                 (unsigned long long)scan.profile_admit_ns,
                 (unsigned long long)scan.profile_edges);
#endif
  return 0;
}

}  // namespace zvec::core
