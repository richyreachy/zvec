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
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace zvec::core {

// Fixed-batch quantized neighbor search over an externally owned FP32 L2 graph.
// Graph: valid(id), neighbors(id) (size/index access), and get_vector(id,
// Vector&). A Graph::Vector owns/pins its FP32 data until destruction; data()
// exposes it. IDs occupy [0, doc_cnt), with holes reported by valid(). Reads
// must be safe across prebuild workers and concurrent searches. Codec:
// kBatchSize, encoded_dim(), batch_bytes(), transform(), encode(), and
// Query/prepare_query()/scan(). Estimates and exact scores use squared L2.
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

  QuantizedGraph(size_t dimension, Codec codec, size_t max_neighbors = 32);
  void clear();
  template <class Graph>
  int prebuild(const Graph &graph, size_t doc_cnt, size_t threads);
  template <class Graph, class Context>
  int search(NodeId entry, const Graph &graph, Context &ctx) const;
  NodeId entry() const {
    return entry_;
  }

 private:
  class LocalVisitSet {
   public:
    void reset(size_t expect) {
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
    inline static thread_local std::vector<NodeId> table_;
    inline static thread_local std::unordered_set<NodeId> overflow_;
    size_t mask_{0};
  };

  // One fixed-stride slot per node inside a single arena allocation, like
  // upstream SymphonyQG's row layout: header, center vector, neighbor ids,
  // then codec payloads. The fixed stride makes every block address pure
  // arithmetic (base + id * stride) and keeps the index in one contiguous
  // mapping. Trivially destructible; the arena is freed wholesale.
  struct Block {
    static constexpr size_t kCenterOffset = 64;

    uint32_t neighbor_cnt;
    uint32_t center_floats;
    uint32_t code_bytes;
    // Rounded-up capacity of the neighbor region so the codes offset does
    // not depend on the actual degree.
    uint32_t neighbor_bytes;

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
  size_t max_neighbors_{32};
  size_t neighbor_bytes_{0};
  Codec codec_;
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
                                      size_t max_neighbors)
    : dimension_(dimension), codec_(std::move(codec)) {
  static_assert(Codec::kBatchSize > 0, "Codec batch size must be positive");
  constexpr size_t batch_size = Codec::kBatchSize;
  max_neighbors_ =
      std::max(batch_size, max_neighbors - max_neighbors % batch_size);
  neighbor_bytes_ = (max_neighbors_ * sizeof(NodeId) + 63) & ~size_t{63};
  const size_t code_bytes =
      (max_neighbors_ / batch_size) * codec_.batch_bytes();
  stride_ = (Block::kCenterOffset + dimension_ * sizeof(float) +
             neighbor_bytes_ + code_bytes + 63) &
            ~size_t{63};
}

template <class Codec>
void QuantizedGraph<Codec>::clear() {
  // Called under the caller's exclusive lock; no concurrent searches.
  arena_.reset();
  arena_count_ = 0;
  entry_ = kInvalidNodeId;
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
  // Trim to the block's degree bound like upstream SymphonyQG's fixed
  // degree_bound_, but with the HNSW diversity heuristic so the truncated
  // graph keeps short and long-range links and stays navigable: plain
  // truncation loses several recall points because HNSW's link order is
  // newest-first. Distances are squared L2, matching the metric this scanner
  // targets.
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
    for (uint32_t idx : order) {
      if (selected.size() == kMaxBlockNeighbors) break;
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
  block->center_floats = static_cast<uint32_t>(dimension_);
  block->code_bytes = static_cast<uint32_t>(code_bytes);
  block->neighbor_bytes = static_cast<uint32_t>(neighbor_bytes_);
  std::copy(center, center + dimension_, const_cast<float *>(block->center()));
  std::copy(scratch.neighbors.begin(), scratch.neighbors.end(),
            const_cast<NodeId *>(block->neighbors()));

  std::fill_n(const_cast<char *>(block->codes()), code_bytes, char{0});
  if (neighbor_cnt == 0) {
    return true;
  }

  codec_.transform(center_ptr, scratch.encoded_center);
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
        empty->center_floats = static_cast<uint32_t>(dimension_);
        empty->neighbor_bytes = static_cast<uint32_t>(neighbor_bytes_);
        std::fill_n(const_cast<float *>(empty->center()), dimension_, 0.0f);
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
  lazy_blocks_.clear();
  arena_ = std::move(arena);
  arena_count_ = static_cast<NodeId>(doc_cnt);
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
  auto &results = ctx.reset_results();
  results.clear();
  if (entry == kInvalidNodeId || ctx.topk() == 0) return 0;
  if (!entity.valid(entry)) return IndexError_Mismatch;
  LocalVisitSet visited;
  visited.reset(results.limit());
  thread_local QuantizedGraphBeam beam(1);
  beam.reset(std::max(size_t{1}, results.limit()));
  beam.insert(entry, std::numeric_limits<float>::max());
  thread_local typename Codec::Query batch_query;
  codec_.prepare_query(ctx.query(), batch_query);
  thread_local std::vector<NodeId> expanded;
  expanded.clear();
  struct Scan {
    enum { kVisitOnExpansion = true };
    const QuantizedGraph &index;
    const Graph &entity;
    Context &ctx;
    decltype(results) output;
    typename Codec::Query &query;
    std::vector<NodeId> &expanded;
    const Block *block = nullptr;
    float exact = 0;
    int begin(NodeId id) {
      expanded.push_back(id);
      block = index.get_block(entity, id);
      if (block == nullptr) return IndexError_Mismatch;
      exact = ctx.distance(block->center());
      if (!ctx.excluded(id)) output.emplace(id, exact);
      ctx.on_expand();
      return 0;
    }
    size_t neighbor_count() const {
      return block->neighbor_cnt;
    }
    size_t batch_size() const {
      return Codec::kBatchSize;
    }
    NodeId neighbor(size_t i) const {
      return block->neighbors()[i];
    }
    void prepare_batch(size_t) {}
    void stage(NodeId, size_t, size_t) {}
    int score(size_t offset, const NodeId *, size_t, float *distances) {
      index.codec_.scan(block->codes() + (offset / Codec::kBatchSize) *
                                             index.codec_.batch_bytes(),
                        query, exact, distances);
      return 0;
    }
  } scan{*this, entity, ctx, results, batch_query, expanded};
  struct Frontier {
    QuantizedGraphBeam &beam;
    Context &ctx;
    bool has_next() const {
      return beam.has_next() && !ctx.reach_scan_limit();
    }
    NodeId pop() {
      return beam.pop();
    }
    bool accepts(float distance) const {
      return !beam.is_full(distance) && std::isfinite(distance);
    }
    void push(NodeId id, float distance) {
      beam.insert(id, distance);
    }
  } frontier{beam, ctx};
  thread_local GraphSearchScratch scratch;
  const int search_ret = SearchGraph(scan, frontier, visited, scratch);
  if (search_ret != 0) return search_ret;
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
