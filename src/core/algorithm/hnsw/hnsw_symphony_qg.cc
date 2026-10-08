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
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <new>
#include <numeric>
#include <thread>
#include <unordered_set>
#if RABITQ_SUPPORTED
#include <rabitqlib/quantization/rabitq.hpp>
#include "symphony_qg_utils.h"
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace zvec::core {

HnswSymphonyQG::HnswSymphonyQG(size_t dimension, size_t max_neighbors)
    : dimension_(dimension), rotation_(dimension) {
#if RABITQ_SUPPORTED
  constexpr size_t batch_size = rabitqlib::fastscan::kBatchSize;
  max_neighbors_ =
      std::max(batch_size, max_neighbors - max_neighbors % batch_size);
  neighbor_bytes_ = (max_neighbors_ * sizeof(node_id_t) + 63) & ~size_t{63};
  const size_t code_bytes =
      (max_neighbors_ / batch_size) *
      rabitqlib::QGBatchDataMap<float>::data_bytes(rotation_.padded_dim());
  stride_ = (Block::kCenterOffset + dimension_ * sizeof(float) +
             neighbor_bytes_ + code_bytes + 63) &
            ~size_t{63};
#else
  (void)max_neighbors;
#endif
}

void HnswSymphonyQG::clear() {
  // Called under the streamer's exclusive lock; no concurrent searches.
  arena_.reset();
  arena_count_ = 0;
  entry_ = kInvalidNodeId;
  std::lock_guard<std::mutex> lock(mutex_);
  lazy_blocks_.clear();
}

#if RABITQ_SUPPORTED

namespace {
// L1-resident visit set for the symphony hot loop. The shared VisitFilter's
// ByteMap spans the whole corpus (1 MB here); its random accesses mostly miss
// L1. Search marks only expanded nodes, so a small open-addressing table sized
// by the beam keeps every probe cached, matching upstream SymphonyQG.
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

  bool visited(node_id_t id) const {
    size_t i = id & mask_;
    for (size_t n = 0; n <= mask_; ++n) {
      const node_id_t v = table_[i];
      if (v == id) return true;
      if (v == kEmpty) return false;
      i = (i + 1) & mask_;
    }
    // Table completely full: entries beyond it live in the overflow set.
    return overflow_.find(id) != overflow_.end();
  }

  void set_visited(node_id_t id) {
    size_t i = id & mask_;
    for (size_t n = 0; n <= mask_; ++n) {
      const node_id_t v = table_[i];
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
  static constexpr node_id_t kEmpty = 0xFFFFFFFFu;
  thread_local static std::vector<node_id_t> table_;
  thread_local static std::unordered_set<node_id_t> overflow_;
  size_t mask_{0};
};

thread_local std::vector<node_id_t> LocalVisitSet::table_;
thread_local std::unordered_set<node_id_t> LocalVisitSet::overflow_;
}  // namespace

bool HnswSymphonyQG::build_block(const HnswEntity &entity, node_id_t id,
                                 const void *center_ptr, Block *slot,
                                 Scratch &scratch, int &error) const {
  const size_t dim = rotation_.padded_dim();
  constexpr size_t batch_size = rabitqlib::fastscan::kBatchSize;
  const size_t batch_bytes = rabitqlib::QGBatchDataMap<float>::data_bytes(dim);

  scratch.neighbors.clear();
  const auto neighbors = entity.get_neighbors(0, id);
  scratch.neighbors.reserve(neighbors.size());
  for (uint32_t i = 0; i < neighbors.size(); ++i) {
    if (entity.get_key(neighbors[i]) != kInvalidKey) {
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
      IndexStorage::MemoryBlock raw;
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
    std::vector<node_id_t> pruned;
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
            const_cast<node_id_t *>(block->neighbors()));

  if (neighbor_cnt == 0) {
    return true;
  }

  rotation_.rotate(center_ptr, scratch.rotated_center);
  scratch.rotated.assign(batch_size * dim, 0.0f);
  for (size_t offset = 0; offset < neighbor_cnt; offset += batch_size) {
    const size_t count = std::min(batch_size, neighbor_cnt - offset);
    for (size_t i = 0; i < count; ++i) {
      IndexStorage::MemoryBlock neighbor;
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
      rotation_.rotate(neighbor.data(), scratch.vector);
      std::copy(scratch.vector.begin(), scratch.vector.end(),
                scratch.rotated.begin() + i * dim);
    }
    rabitqlib::quant::quantize_qg_batch(
        scratch.rotated.data(), scratch.rotated_center.data(), count, dim,
        const_cast<char *>(block->codes()) +
            (offset / batch_size) * batch_bytes,
        rabitqlib::METRIC_L2);
  }
  return true;
}

int HnswSymphonyQG::prebuild(const HnswEntity &entity, size_t doc_cnt,
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
  std::atomic<node_id_t> next_id{0};
  std::atomic<int> first_error{0};

  auto worker = [&](size_t part) {
    Scratch scratch;
    auto &centroid = centroid_parts[part];
    while (first_error.load(std::memory_order_relaxed) == 0) {
      const node_id_t id = next_id.fetch_add(1, std::memory_order_relaxed);
      if (id >= doc_cnt) break;
      auto *slot = reinterpret_cast<Block *>(arena.get() +
                                             static_cast<size_t>(id) * stride_);
      if (entity.get_key(id) == kInvalidKey) {
        // Deleted ids never become expandable (blocks only store valid keys),
        // but leave their slots as defined empty blocks instead of garbage.
        auto *empty = new (slot) Block{};
        empty->center_floats = static_cast<uint32_t>(dimension_);
        empty->neighbor_bytes = static_cast<uint32_t>(neighbor_bytes_);
        std::fill_n(const_cast<float *>(empty->center()), dimension_, 0.0f);
        continue;
      }
      IndexStorage::MemoryBlock raw;
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
  std::atomic<node_id_t> next_entry_id{0};
  std::vector<node_id_t> best_ids(nthreads, kInvalidNodeId);
  std::vector<double> best_dists(nthreads, std::numeric_limits<double>::max());
  auto entry_worker = [&](size_t part) {
    while (true) {
      const node_id_t id =
          next_entry_id.fetch_add(1, std::memory_order_relaxed);
      if (id >= doc_cnt) break;
      if (entity.get_key(id) == kInvalidKey) continue;
      IndexStorage::MemoryBlock raw;
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

  // Published before the streamer transitions to STATE_OPENED; searches only
  // start afterwards, so reads need no synchronization.
  arena_ = std::move(arena);
  arena_count_ = static_cast<node_id_t>(doc_cnt);
  return 0;
}

const HnswSymphonyQG::Block *HnswSymphonyQG::get_block(const HnswEntity &entity,
                                                       node_id_t id) const {
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
    IndexStorage::MemoryBlock raw;
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

int HnswSymphonyQG::search(node_id_t entry, HnswContext &ctx) const {
  const auto &entity = ctx.get_entity();
  auto &results =
      ctx.search_heap().reset<TopkHeap>(std::max(ctx.topk(), ctx.ef()));
  results.clear();
  LocalVisitSet visited;
  visited.reset(results.limit());
  thread_local SymphonyQGBeam beam(1);
  beam.reset(std::max(size_t{1}, results.limit()));
  beam.insert(entry, std::numeric_limits<float>::max());
  thread_local std::vector<float> rotated_query;
  rotation_.rotate(ctx.dist_calculator().query(), rotated_query);
  // Reused across queries so the fastscan LUT buffers stay allocation-free
  // and cache-warm.
  thread_local SymQuery batch_query;
  batch_query.reset(rotated_query.data(), rotation_.padded_dim());
  constexpr size_t batch_size = rabitqlib::fastscan::kBatchSize;
  const size_t batch_bytes =
      rabitqlib::QGBatchDataMap<float>::data_bytes(rotation_.padded_dim());
  std::array<float, batch_size> distances;
  thread_local std::vector<node_id_t> expanded;
  expanded.clear();
  while (beam.has_next() && !ctx.reach_scan_limit()) {
    const node_id_t id = beam.pop();
    // Local estimates for the same neighbor differ by center. Like SymphonyQG,
    // mark nodes on expansion, allowing a better estimate to re-enter the beam.
    if (visited.visited(id)) continue;
    visited.set_visited(id);
    expanded.push_back(id);
    const Block *block = get_block(entity, id);
    if (block == nullptr) return IndexError_Mismatch;
    const float exact = ctx.dist_calculator().dist(block->center());
    if (!ctx.filter().is_valid() || !ctx.filter()(entity.get_key(id))) {
      results.emplace(id, exact);
    }
    batch_query.set_g_add(exact);
    if (ctx.debugging()) ++(*ctx.mutable_stats_get_neighbors());
    const char *codes = block->codes();
    const node_id_t *neighbors = block->neighbors();
    const size_t neighbor_cnt = block->neighbor_cnt;
    for (size_t offset = 0; offset < neighbor_cnt; offset += batch_size) {
      const size_t count = std::min(batch_size, neighbor_cnt - offset);
      ScanSymphonyQGBatch(codes + (offset / batch_size) * batch_bytes,
                          batch_query, rotation_.padded_dim(),
                          distances.data());
      for (size_t i = 0; i < count; ++i) {
        const float distance = distances[i];
        if (beam.is_full(distance) || !std::isfinite(distance)) continue;
        const node_id_t neighbor = neighbors[offset + i];
        if (visited.visited(neighbor)) continue;
        beam.insert(neighbor, distance);
      }
    }
  }
  // As in SymphonyQG's result completion step, score unvisited neighbors if
  // duplicate estimates or filters left fewer than k eligible results. Avoid
  // building additional cached blocks and respect the exact-distance budget.
  for (node_id_t id : expanded) {
    if (results.size() >= ctx.topk() || ctx.reach_scan_limit()) break;
    const auto neighbors = entity.get_neighbors(0, id);
    for (uint32_t i = 0; i < neighbors.size(); ++i) {
      if (results.size() >= ctx.topk() || ctx.reach_scan_limit()) break;
      const node_id_t candidate = neighbors[i];
      const auto key = entity.get_key(candidate);
      if (visited.visited(candidate) || key == kInvalidKey) continue;
      visited.set_visited(candidate);
      if (ctx.filter().is_valid() && ctx.filter()(key)) continue;
      IndexStorage::MemoryBlock raw;
      int ret = entity.get_vector(candidate, raw);
      if (ret != 0) return ret;
      if (raw.data() == nullptr) return IndexError_Mismatch;
      const float exact = ctx.dist_calculator().dist(raw.data());
      results.emplace(candidate, exact);
    }
  }
  return 0;
}

#else
int HnswSymphonyQG::prebuild(const HnswEntity &, size_t, size_t) {
  return IndexError_Unsupported;
}

int HnswSymphonyQG::search(node_id_t, HnswContext &) const {
  return IndexError_Unsupported;
}
#endif
}  // namespace zvec::core
