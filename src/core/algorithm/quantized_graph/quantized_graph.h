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

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include "utility/graph_search.h"
#include "quantized_graph_bucket.h"
#include "quantized_graph_integer_center.h"
#include "quantized_graph_integer_router.h"
#include "quantized_graph_sq8_center.h"
#include "quantized_graph_visit_set.h"
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
// Query/prepare_query()/scan(..., uncertainty_bonus). Estimates and exact
// scores must use the same metric; centroid selection and diversity pruning use
// squared L2 on the first dimension coordinates. Optional trailing FP32
// metadata is retained in storage vectors for exact scoring but excluded from
// graph geometry. Context: reset_results() returns a bounded heap with
// limit/size/emplace; query(), topk(), distance(), excluded(id),
// reach_scan_limit(), on_expand(), quantized_graph_options(). distance()
// accounts for the exact scan budget; excluded() means EXCLUDE. The caller
// serializes prebuild/clear against searches. The source graph remains
// immutable while this snapshot is active.
template <class Codec>
class QuantizedGraph {
 public:
  using NodeId = uint32_t;
  static constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::max();

  // Physical bytes to prefetch for each refinement vector. Prebuild adapters
  // may expose decoded coordinates independently of this storage format.
  QuantizedGraph(size_t dimension, Codec codec, size_t max_neighbors,
                 size_t stored_vector_bytes, bool cosine,
                 std::string model_data);
  void clear();
  template <class Graph>
  int prebuild(const Graph &graph, size_t doc_cnt, size_t threads);
  template <class Graph, class Context>
  int search(NodeId entry, const Graph &graph, Context &ctx) const {
    return cosine_ ? search_impl<true>(entry, graph, ctx)
                   : search_impl<false>(entry, graph, ctx);
  }
  NodeId entry() const {
    return entry_;
  }

 private:
  template <bool Cosine, class Graph, class Context>
  __attribute__((
      target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,fma"))) int
  search_impl(NodeId entry, const Graph &graph, Context &ctx) const;

  using Bucket = QuantizedGraphBucket;
  using LocalVisitSet = QuantizedGraphVisitSet;

  template <size_t Lines>
  __attribute__((always_inline)) static inline void prefetch_fixed(
      const char *ptr) {
#pragma GCC unroll 64
    for (size_t line = 0; line < Lines; ++line)
      __builtin_prefetch(ptr + line * 64, 0, 3);
  }
  __attribute__((always_inline)) static inline void prefetch_span(
      const char *ptr, size_t bytes) {
    // Unroll common center, vector and code spans; retain arbitrary lengths.
    switch ((bytes + 63) / 64) {
      case 10:
        prefetch_fixed<10>(ptr);
        break;
      case 48:
        prefetch_fixed<48>(ptr);
        break;
      case 49:
        prefetch_fixed<49>(ptr);
        break;
      case 52:
        prefetch_fixed<52>(ptr);
        break;
      case 60:
        prefetch_fixed<60>(ptr);
        break;
      case 64:
        prefetch_fixed<64>(ptr);
        break;
      default:
#pragma GCC unroll 8
        for (size_t offset = 0; offset < bytes; offset += 64)
          __builtin_prefetch(ptr + offset, 0, 3);
    }
  }

  // Resolve TLS once at the query boundary. In particular, probing the visit
  // table must not call __tls_get_addr for every candidate in a shared build.
  struct SearchState {
    LocalVisitSet visited;
    Bucket beam;
    typename Codec::Query query;
    IntegerCenter::Query integer_query;
    QuantizedGraphIntegerRouter::Query projected_route_query;
    std::vector<NodeId> expanded;
    std::vector<std::pair<float, NodeId>> refine_candidates;
    std::vector<float> refine_best;
    GraphSearchScratch scratch;
    std::vector<float> best_estimated;
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
    static constexpr size_t kCenterOffset = 64;

    uint32_t neighbor_cnt;
    uint32_t center_bytes;
    // Rounded-up capacity of the neighbor region so the codes offset does
    // not depend on the actual degree.
    uint32_t neighbor_bytes;
    float center_error, center_norm2;
    float center_offset, center_scale;
    double center_mean_dot;
    float center_residual_norm, center_coded_norm;

    const uint8_t *center() const {
      return reinterpret_cast<const uint8_t *>(this) + kCenterOffset;
    }
    uint8_t *center() {
      return const_cast<uint8_t *>(std::as_const(*this).center());
    }
    const NodeId *neighbors() const {
      return reinterpret_cast<const NodeId *>(
          reinterpret_cast<const char *>(this) + kCenterOffset + center_bytes);
    }
    const char *codes() const {
      return reinterpret_cast<const char *>(neighbors()) + neighbor_bytes;
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

  // Prebuilt blocks are immutable for the lifetime of the snapshot.
  const Block *get_block(NodeId id) const {
    return id < arena_count_ ? slot(id) : nullptr;
  }
  // Writes one fixed-stride block into slot; false on error, with no cleanup
  // (the caller owns the memory).
  template <class Graph>
  bool build_block(const Graph &entity, NodeId id, const void *center,
                   Block *slot, Scratch &scratch, int &error) const;

  std::string model_data_;
  size_t dimension_;
  size_t stored_vector_bytes_;
  size_t stored_center_bytes_;
  std::vector<float> center_shift_;
  std::vector<float> integer_shift_;
  double center_shift_norm2_{0};
  bool cosine_{false};
  size_t max_neighbors_{32};
  size_t neighbor_bytes_{0};
  Codec codec_;
  QuantizedGraphIntegerRouter integer_router_;
  // Immutable after prebuild and before the next clear(); read lock-free.
  std::unique_ptr<char[], ArenaDeleter> arena_;
  NodeId arena_count_{0};
  size_t stride_{0};
  NodeId entry_{kInvalidNodeId};
};

template <class Codec>
QuantizedGraph<Codec>::QuantizedGraph(size_t dimension, Codec codec,
                                      size_t max_neighbors,
                                      size_t stored_vector_bytes, bool cosine,
                                      std::string model_data)
    : model_data_(std::move(model_data)),
      dimension_(dimension),
      stored_vector_bytes_(stored_vector_bytes),
      codec_(std::move(codec)) {
  cosine_ = cosine;
  stored_center_bytes_ = (dimension_ + 63) & ~size_t{63};
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
  integer_router_.clear();
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
  const size_t original_neighbor_count = scratch.neighbors.size();
  // Offline two-hop candidate repair. Fixed row capacity and the exact same
  // query kernel are retained. No query or GT is used during construction.
  if (scratch.neighbors.size() < max_neighbors_) {
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
    if (original_neighbor_count < kMaxBlockNeighbors)
      for (uint32_t i = 0; i < original_neighbor_count; ++i)
        selected.push_back(i);
    for (uint32_t idx : order) {
      if (selected.size() == kMaxBlockNeighbors) break;
      if (original_neighbor_count < kMaxBlockNeighbors &&
          idx < original_neighbor_count)
        continue;
      bool diverse = true;
      for (uint32_t s : selected) {
        if (sq_dist(idx, s) < dists[idx]) {
          diverse = false;
          break;
        }
      }
      if (diverse) selected.push_back(idx);
    }
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
  const auto *center = static_cast<const float *>(center_ptr);
  block->neighbor_cnt = static_cast<uint32_t>(neighbor_cnt);
  block->center_bytes = static_cast<uint32_t>(stored_center_bytes_);
  block->neighbor_bytes = static_cast<uint32_t>(neighbor_bytes_);
  if (center_shift_.size() != dimension_) {
    error = IndexError_Mismatch;
    return false;
  }
  auto *center_codes = block->center();
  const bool encoded =
      cosine_ ? SQ8Center::encode_shifted(
                    center, center_shift_.data(), dimension_, center_codes,
                    block->center_error, block->center_norm2,
                    block->center_offset, block->center_scale)
              : SQ8Center::encode(center, dimension_, center_codes,
                                  block->center_error, block->center_norm2,
                                  block->center_offset, block->center_scale);
  if (encoded) {
    IntegerCenter::annotate(
        center, cosine_ ? center_shift_.data() : integer_shift_.data(),
        center_codes, dimension_, block->center_offset, block->center_scale,
        block->center_error, block->center_residual_norm,
        block->center_coded_norm, block->center_mean_dot);
    if (!cosine_)
      block->center_mean_dot =
          double(block->center_coded_norm) * block->center_coded_norm;
  }
  if (!encoded || !std::isfinite(block->center_error) ||
      !std::isfinite(block->center_norm2)) {
    error = IndexError_Mismatch;
    return false;
  }
  std::copy(scratch.neighbors.begin(), scratch.neighbors.end(),
            const_cast<NodeId *>(block->neighbors()));

  std::fill_n(const_cast<char *>(block->codes()), code_bytes, char{0});
  if (neighbor_cnt == 0) {
    return true;
  }

  const void *encoding_center = center_ptr;
  std::vector<float> reconstructed;
  if (!cosine_) {
    reconstructed.resize(dimension_);
    SQ8Center::decode(block->center(), dimension_, reconstructed.data(),
                      block->center_offset, block->center_scale);
    encoding_center = reconstructed.data();
  }
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
  if (dimension_ == 0 || dimension_ > 4096 ||
      !__builtin_cpu_supports("avx512vnni"))
    return IndexError_Mismatch;
  static_assert(sizeof(Block) <= Block::kCenterOffset,
                "Block header overlaps center");
  center_shift_.assign(dimension_, 0.f);
  integer_shift_.assign(dimension_, 0.f);
  if (!model_data_.empty() &&
      !QuantizedGraphIntegerRouter::read_mean_data(
          model_data_, doc_cnt, center_shift_, dimension_, cosine_))
    return IndexError_Mismatch;
  center_shift_norm2_ = 0;
  for (float v : center_shift_) {
    if (!std::isfinite(v)) return IndexError_Mismatch;
    center_shift_norm2_ += double(v) * v;
  }
  if (!model_data_.empty()) {
    std::vector<float> route_mean;
    codec_.transform(center_shift_.data(), route_mean);
    if (!integer_router_.load_data(model_data_, center_shift_, entity, doc_cnt,
                                   &route_mean, cosine_))
      return IndexError_Mismatch;
  }
  const size_t nthreads =
      threads == 0 ? std::max(1u, std::thread::hardware_concurrency())
                   : threads;
  std::unique_ptr<char[], ArenaDeleter> arena;
  if (doc_cnt > kInvalidNodeId ||
      doc_cnt > std::numeric_limits<size_t>::max() / stride_)
    return IndexError_Mismatch;
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
        empty->center_bytes = static_cast<uint32_t>(stored_center_bytes_);
        empty->neighbor_bytes = static_cast<uint32_t>(neighbor_bytes_);
        std::fill_n(empty->center(), stored_center_bytes_, uint8_t{0});
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
  arena_ = std::move(arena);
  arena_count_ = static_cast<NodeId>(doc_cnt);
  return 0;
}

template <class Codec>
template <bool Cosine, class Graph, class Context>
int QuantizedGraph<Codec>::search_impl(NodeId entry, const Graph &entity,
                                       Context &ctx) const {
  auto &results = ctx.reset_results();
  results.clear();
  if (entry == kInvalidNodeId || ctx.topk() == 0) return 0;
  if (!entity.valid(entry)) return IndexError_Mismatch;
  auto &state = search_state();
  auto &visited = state.visited;
  visited.reset(results.limit());
  auto &beam = state.beam;
  auto &batch_query = state.query;
  codec_.prepare_query(ctx.query(), batch_query);
  if (!IntegerCenter::prepare(
          static_cast<const float *>(ctx.query()),
          Cosine ? center_shift_.data() : integer_shift_.data(), dimension_,
          state.integer_query))
    return IndexError_Mismatch;
  auto &expanded = state.expanded;
  expanded.clear();
  const auto &options = ctx.quantized_graph_options();
  if (!integer_router_.empty() && options.route_beam != 0) {
    const size_t route_beam = options.route_beam;
    if (!integer_router_.prepare_projected(batch_query.rotated.data(),
                                           state.projected_route_query))
      return IndexError_Mismatch;
    const NodeId chosen =
        integer_router_.choose(state.projected_route_query, route_beam);
    if (chosen == kInvalidNodeId) return IndexError_Mismatch;
    entry = chosen;
  }
  beam.reset();
  beam.insert(entry, std::numeric_limits<float>::max());
  state.refine_candidates.clear();
  state.refine_best.clear();
  struct StopState {
    std::vector<float> &best;
    size_t k, expanded = 0;
    float margin = .02f;
    size_t cap = 64;
    void observe(float d, bool eligible) {
      ++expanded;
      if (eligible && (best.size() < k || d < best.back())) {
        best.insert(std::lower_bound(best.begin(), best.end(), d), d);
        if (best.size() > k) best.pop_back();
      }
    }
    bool stop(float next) const {
      return best.size() == k && expanded >= 10 &&
             next - best.back() >= margin * std::max(best.back(), 1e-6f);
    }
  };
  state.best_estimated.clear();
  StopState stopping{state.best_estimated, ctx.topk()};
  stopping.margin =
      options.margin >= 0 ? options.margin : (Cosine ? .02f : .085f);
  stopping.cap =
      options.max_expansions ? options.max_expansions : (Cosine ? 64 : 256);
  struct Scan {
    enum { kVisitOnExpansion = true };
    const QuantizedGraph &index;
    Context &ctx;
    typename Codec::Query &query;
    const IntegerCenter::Query &integer_query;
    std::vector<NodeId> &expanded;
    float bucket_width;
    float uncertainty_bonus;
    std::vector<std::pair<float, NodeId>> &refine_candidates;
    Bucket *bucket;
    StopState *stopping;
    const Block *block = nullptr;
    const NodeId *neighbors = nullptr;
    const char *codes = nullptr;
    float navigation_center = 0;
    int begin(NodeId id) {
      expanded.push_back(id);
      block = index.get_block(id);
      if (block == nullptr) return IndexError_Mismatch;
      neighbors = reinterpret_cast<const NodeId *>(
          reinterpret_cast<const char *>(block->center()) +
          index.stored_center_bytes_);
      codes = reinterpret_cast<const char *>(neighbors) + index.neighbor_bytes_;
#if defined(__GNUC__) || defined(__clang__)
      // Fetch the row header and the start of its packed codes while the
      // exact center distance is being computed. The layout is fixed for
      // both arena and lazy blocks, so this needs no dependent header load.
      __builtin_prefetch(block, 0, 3);
      const size_t code_offset = codes - reinterpret_cast<const char *>(block);
      const size_t bytes = std::min(size_t{4096}, index.stride_ - code_offset);
      index.prefetch_span(codes, bytes);
      const size_t batch_bytes = index.codec_.batch_bytes();
      // The end of the first block contains SymphonyQG's score factors.
      if (batch_bytes > bytes && batch_bytes >= 256)
        index.template prefetch_fixed<4>(codes + batch_bytes - 256);
#endif
      const bool eligible = !ctx.excluded(id);
      float estimate;
      if constexpr (Cosine) {
        double reconstructed;
        estimate = IntegerCenter::distance(
            integer_query, block->center(), index.dimension_,
            block->center_offset, block->center_scale, block->center_mean_dot,
            index.center_shift_norm2_, block->center_coded_norm, reconstructed);
        navigation_center = estimate;
        float lower = IntegerCenter::lower_bound(
            estimate, integer_query, block->center_error,
            block->center_residual_norm, block->center_coded_norm,
            block->center_norm2, block->center_mean_dot,
            index.center_shift_norm2_, index.dimension_, reconstructed);
        if (eligible) refine_candidates.emplace_back(lower, id);
      } else {
        double reconstructed;
        estimate = IntegerCenter::l2_distance(
            integer_query, block->center(), index.dimension_,
            block->center_offset, block->center_scale, block->center_norm2,
            block->center_coded_norm, reconstructed);
        navigation_center = float(double(estimate) + block->center_mean_dot -
                                  block->center_norm2);
        const float lower = IntegerCenter::l2_lower_bound(
            reconstructed, integer_query, block->center_error,
            block->center_coded_norm, index.dimension_);
        if (eligible) refine_candidates.emplace_back(lower, id);
      }
      if (expanded.size() == 1) bucket->set_scale(estimate, bucket_width);
      // Heuristic navigation threshold; interval bounds still certify terminal
      // Top-K over expanded nodes, not navigation/global recall.
      stopping->observe(estimate, eligible);
      ctx.on_expand();
      return 0;
    }
    size_t neighbor_count() const {
      return block->neighbor_cnt;
    }
    size_t batch_size() const {
      return Codec::kBatchSize;
    }
    const NodeId *neighbor_data(size_t offset) const {
      return neighbors + offset;
    }
    int score(size_t offset, const NodeId *, size_t count, float *distances) {
      index.codec_.scan(
          codes + (offset / Codec::kBatchSize) * index.codec_.batch_bytes(),
          query, navigation_center, distances, uncertainty_bonus);
      if (expanded.size() == 1 && offset == 0) {
        float maximum = 0.f;
        for (size_t i = 0; i < count; ++i)
          if (std::isfinite(distances[i]))
            maximum = std::max(maximum, distances[i]);
        bucket->cover_initial_range(maximum);
      }
      return 0;
    }
  } scan{*this,
         ctx,
         batch_query,
         state.integer_query,
         expanded,
         options.bucket_width > 0 ? options.bucket_width
                                  : (Cosine ? .016f : .002f),
         options.uncertainty_bonus,
         state.refine_candidates,
         &beam,
         &stopping};
  struct Frontier {
    const QuantizedGraph &index;
    Bucket &beam;
    Context &ctx;
    StopState *stopping;
    LocalVisitSet *visited;
    NodeId prefetched = kInvalidNodeId;
    bool has_next() const {
      // Enforce the hard budget before processing any more frontier entries.
      // Filtered nodes need not fill best or consume the exact-scan budget.
      if (stopping->expanded >= stopping->cap) return false;
      while (beam.has_next()) {
        const NodeId next = beam.next_id();
        if (visited->visited(next)) {
          beam.pop();
          continue;
        }
        break;
      }
      if (beam.has_next() && stopping->stop(beam.next_distance())) return false;
      return beam.has_next() && !ctx.reach_scan_limit();
    }
    NodeId pop() {
      return beam.pop();
    }
    void push_batch(const NodeId *ids, const float *distances, size_t count) {
      static_assert(Codec::kBatchSize <= 32);
      std::array<uint32_t, Codec::kBatchSize> bins;
      beam.locate_batch(distances, bins.data(), count);
      uint32_t accepted = visited->unvisited_batch(ids, distances, count);
      beam.reserve_insertions(count);
      // The estimated Top-K threshold can only decrease. Once stopping is
      // enabled, a bin that fails the stop rule now can never be expanded.
      // Visits are marked on expansion, so later better estimates still enter.
      const bool prune =
          stopping->best.size() == stopping->k && stopping->expanded >= 10;
      const float worst = prune ? stopping->best.back() : 0.f;
      const float allowance = stopping->margin * std::max(worst, 1e-6f);
      if (prune)
        accepted &= beam.admissible_mask(bins.data(), count, worst, allowance);
      while (accepted) {
        const unsigned i = __builtin_ctz(accepted);
        accepted &= accepted - 1;
        const bool changes_first = bins[i] <= beam.first();
        beam.insert_reserved(ids[i], bins[i]);
        if (changes_first) prefetch_next();
      }
      // Even when all lanes were pruned, prefetch the next queued node.
      prefetch_next();
    }
    void prefetch_next() {
#if defined(__GNUC__) || defined(__clang__)
      if (beam.has_next()) {
        const NodeId next = beam.next_id();
        // Only prefetch published arena slots.
        if (next < index.arena_count_ && next != prefetched) {
          prefetched = next;
          __builtin_prefetch(index.slot(next), 0, 3);
          const char *center =
              reinterpret_cast<const char *>(index.slot(next)->center());
          const size_t bytes =
              std::min(size_t{640}, index.stride_ - Block::kCenterOffset);
          index.prefetch_span(center, bytes);
        }
      }
#endif
    }
  } frontier{*this, beam, ctx, &stopping, &visited};
  const int search_ret = SearchGraph(scan, frontier, visited, state.scratch);
  if (search_ret != 0) return search_ret;
  // Exact Top-K of the expanded set, via conservative interval ordering.
  // Navigational estimates still use reconstructed-center coordinates.
  auto &refine = state.refine_candidates;
  auto &best = state.refine_best;
  std::sort(refine.begin(), refine.end());
  for (size_t first = 0; first < refine.size(); first += 4) {
    if (best.size() == ctx.topk() && refine[first].first > best.back()) break;
    if (ctx.reach_scan_limit()) break;
    // Keep owning/pinning handles alive until every vector in the batch has
    // been scored. Storage backends need not return permanent pointers.
    std::array<typename Graph::Vector, 4> raw;
    size_t count = 0;
    for (; count < 4 && first + count < refine.size(); ++count) {
      const auto &candidate = refine[first + count];
      if (best.size() == ctx.topk() && candidate.first > best.back()) break;
      if (candidate.second >= arena_count_) return IndexError_Mismatch;
      const int ret = entity.get_vector(candidate.second, raw[count]);
      if (ret != 0) return ret;
      if (raw[count].data() == nullptr) return IndexError_Mismatch;
      const char *ptr = static_cast<const char *>(raw[count].data());
      prefetch_span(ptr, stored_vector_bytes_);
    }
    for (size_t i = 0; i < count; ++i) {
      const auto &candidate = refine[first + i];
      if (best.size() == ctx.topk() && candidate.first > best.back()) break;
      if (ctx.reach_scan_limit()) break;
      const float exact = ctx.distance(raw[i].data());
      results.emplace(candidate.second, exact);
      best.insert(std::lower_bound(best.begin(), best.end(), exact), exact);
      if (best.size() > ctx.topk()) best.pop_back();
    }
  }
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
  return 0;
}

}  // namespace zvec::core
