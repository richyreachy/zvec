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
#include "hnsw_algorithm.h"
#include "utility/graph_search.h"

namespace zvec {
namespace core {

template <typename EntityType>
int HnswAlgorithm<EntityType>::add_node(node_id_t id, level_t level,
                                        HnswContext *ctx) {
  spin_lock_.lock();

  auto cur_max_level = entity_.cur_max_level();
  auto entry_point = entity_.entry_point();
  if (ailego_unlikely(entry_point == kInvalidNodeId)) {
    entity_.update_ep_and_level(id, level);
    spin_lock_.unlock();
    return 0;
  }
  spin_lock_.unlock();

  if (ailego_unlikely(level > cur_max_level)) {
    mutex_.lock();
    // re-check max level
    cur_max_level = entity_.cur_max_level();
    entry_point = entity_.entry_point();
    if (level <= cur_max_level) {
      mutex_.unlock();
    }
  }

  level_t cur_level = cur_max_level;
  dist_t dist = ctx->batch_dist(entry_point);
  for (; cur_level > level; --cur_level) {
    select_entry_point(cur_level, &entry_point, &dist, ctx);
  }

  for (; cur_level >= 0; --cur_level) {
    search_neighbors(cur_level, &entry_point, &dist, ctx->level_topk(cur_level),
                     ctx);
  }

  // add neighbors from down level to top level, to avoid upper level visible
  // to knn_search but the under layer level not ready
  for (cur_level = 0; cur_level <= level; ++cur_level) {
    add_neighbors(id, cur_level, ctx->level_topk(cur_level), ctx);
    ctx->level_topk(cur_level).clear();
  }

  if (ailego_unlikely(level > cur_max_level)) {
    spin_lock_.lock();
    entity_.update_ep_and_level(id, level);
    spin_lock_.unlock();
    mutex_.unlock();
  }

  return 0;
}

template <typename EntityType>
int HnswAlgorithm<EntityType>::search(HnswContext *ctx) const {
  spin_lock_.lock();
  auto max_level = entity_.cur_max_level();
  auto entry_point = entity_.entry_point();
  spin_lock_.unlock();

  if (ailego_unlikely(entry_point == kInvalidNodeId)) {
    return 0;
  }

  if (symphony_qg_ && !ctx->group_by_search()) {
    const node_id_t sym_entry = symphony_qg_->entry();
    if (sym_entry != kInvalidNodeId) {
      // Upstream SymphonyQG searches a single level from the node nearest
      // the corpus centroid; skip the upper-level descent entirely.
      return symphony_qg_->search(sym_entry, *ctx);
    }
    if (ailego_unlikely(ctx->has_extra_values())) {
      dist_t dist = ctx->dist_calculator().dist(entry_point);
      for (level_t cur_level = max_level; cur_level >= 1; --cur_level) {
        select_entry_point(cur_level, &entry_point, &dist, ctx);
      }
    } else {
      // Lean greedy descent for the quantized level-0 search: same closest-
      // neighbor traversal as select_entry_point, but with borrowed vector
      // blocks and no per-hop allocations.
      const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
      HnswDistCalculator &dc = ctx->dist_calculator();
      dist_t dist = dc.dist(entry_point);
      for (level_t cur_level = max_level; cur_level >= 1; --cur_level) {
        while (true) {
          const auto neighbors = entity.get_neighbors(cur_level, entry_point);
          const uint32_t size = neighbors.size();
          if (size == 0) break;
          node_id_t best = entry_point;
          dist_t best_dist = dist;
          for (uint32_t i = 0; i < size; ++i) {
            IndexStorage::MemoryBlock block;
            if (ailego_unlikely(dc.get_vector(neighbors[i], block) != 0)) {
              break;
            }
            const dist_t d = dc.dist(block.data());
            if (d < best_dist) {
              best_dist = d;
              best = neighbors[i];
            }
          }
          if (best == entry_point) break;
          entry_point = best;
          dist = best_dist;
        }
      }
    }
    return symphony_qg_->search(entry_point, *ctx);
  }

  dist_t dist = ctx->dist_calculator().dist(entry_point);
  for (level_t cur_level = max_level; cur_level >= 1; --cur_level) {
    select_entry_point(cur_level, &entry_point, &dist, ctx);
  }

  const uint32_t capacity = std::max(ctx->topk(), ctx->ef());
  if (!ctx->filter().is_valid()) {
    ctx->search_heap().reset_pool(capacity, entity_.max_degree(0));
    const int ret = dispatch_search_neighbors(entry_point, dist, ctx);
    if (ailego_unlikely(ret != 0)) return ret;
  } else {
    auto &topk = ctx->search_heap().reset<TopkHeap>(capacity);
    search_neighbors(0, &entry_point, &dist, topk, ctx);
  }

  if (ctx->group_by_search()) {
    expand_neighbors_by_group(ctx);
  }

  return 0;
}

template <typename EntityType>
void HnswAlgorithm<EntityType>::select_entry_point(level_t level,
                                                   node_id_t *entry_point,
                                                   dist_t *dist,
                                                   HnswContext *ctx) const {
  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  HnswDistCalculator &dc = ctx->dist_calculator();
  const bool use_provider = dc.has_provider();
  const bool has_extra_values = ctx->has_extra_values();
  while (true) {
    const auto neighbors = entity.get_neighbors_typed(level, *entry_point);
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_neighbors())++;
    }
    uint32_t size = neighbors.size();
    if (size == 0) {
      break;
    }

    std::vector<MemBlockType> neighbor_vec_blocks;
    std::vector<IndexStorage::MemoryBlock> provider_vec_blocks;
    int ret;
    if (ailego_unlikely(use_provider)) {
      ret = dc.get_vector(&neighbors[0], size, provider_vec_blocks);
    } else {
      ret = entity.get_vector_typed(&neighbors[0], size, neighbor_vec_blocks);
    }
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_vector())++;
    }
    if (ailego_unlikely(ret != 0)) {
      break;
    }

    bool find_closer = false;

    std::vector<float> dists(size);
    std::vector<const void *> neighbor_vecs(size);
    std::vector<const void *> neighbor_extra_values(has_extra_values ? size
                                                                     : 0);
    if (ailego_unlikely(use_provider)) {
      for (uint32_t i = 0; i < size; ++i) {
        neighbor_vecs[i] = provider_vec_blocks[i].data();
        if (has_extra_values) {
          neighbor_extra_values[i] = ctx->get_extra_values(neighbor_vecs[i]);
        }
      }
    } else {
      for (uint32_t i = 0; i < size; ++i) {
        neighbor_vecs[i] = neighbor_vec_blocks[i].data();
        if (has_extra_values) {
          neighbor_extra_values[i] = ctx->get_extra_values(neighbor_vecs[i]);
        }
      }
    }

    dc.batch_dist(neighbor_vecs.data(), size, dists.data(),
                  has_extra_values ? neighbor_extra_values.data() : nullptr);

    for (uint32_t i = 0; i < size; ++i) {
      dist_t cur_dist = dists[i];

      if (cur_dist < *dist) {
        *entry_point = neighbors[i];
        *dist = cur_dist;
        find_closer = true;
      }
    }

    if (!find_closer) {
      break;
    }
  }

  return;
}

template <typename EntityType>
void HnswAlgorithm<EntityType>::add_neighbors(node_id_t id, level_t level,
                                              TopkHeap &topk_heap,
                                              HnswContext *ctx) {
  if (ailego_unlikely(topk_heap.size() == 0)) {
    return;
  }

  HnswDistCalculator &dc = ctx->dist_calculator();

  update_neighbors(dc, id, level, topk_heap, ctx);

  // reverse update neighbors
  for (size_t i = 0; i < topk_heap.size(); ++i) {
    reverse_update_neighbors(dc, topk_heap[i].first, level, id,
                             topk_heap[i].second, ctx->update_heap(), ctx);
  }

  return;
}

// Exact-distance scan policies feed the same SearchGraph loop as
// QuantizedGraph. Direct preserves mmap prefetching; Buffered pins pages
// through batch_dist; Fallback also supports external providers and
// construction/search statistics.
enum class HnswScanMode { Direct, Buffered, Fallback };

template <class EntityType, class MemBlockType, HnswScanMode Mode>
class HnswExactGraphScan {
 public:
  static constexpr bool kVisitOnExpansion = false;
  HnswExactGraphScan(const EntityType &entity, level_t level,
                     HnswDistCalculator &dc, HnswContext &ctx,
                     uint32_t prefetch_lines, uint32_t prefetch_offset)
      : entity_(entity),
        level_(level),
        dc_(dc),
        ctx_(ctx),
        prefetch_lines_(prefetch_lines),
        prefetch_offset_(prefetch_offset),
        has_extra_values_(ctx.has_extra_values()),
        use_provider_(Mode == HnswScanMode::Fallback && dc.has_provider()) {}

  int begin(node_id_t id) {
    // Release the previous adjacency page before asking the buffer pool for
    // another row, matching the lifetime of the former loop-local view.
    neighbors_ = {};
    neighbors_ = entity_.get_neighbors_typed(level_, id);
    ailego_prefetch(neighbors_.data);
    if constexpr (Mode == HnswScanMode::Fallback) {
      if (ctx_.debugging()) ++(*ctx_.mutable_stats_get_neighbors());
    }
    return 0;
  }
  size_t neighbor_count() const {
    return neighbors_.size();
  }
  size_t batch_size() const {
    return neighbors_.size();
  }
  node_id_t neighbor(size_t i) const {
    return neighbors_[i];
  }
  void prepare_batch(size_t capacity) {
    if (neighbor_vecs_.size() < capacity) neighbor_vecs_.resize(capacity);
    if (has_extra_values_ && neighbor_extra_values_.size() < capacity)
      neighbor_extra_values_.resize(capacity);
  }
  void on_duplicate() {
    if constexpr (Mode == HnswScanMode::Fallback) {
      if (ctx_.debugging()) ++(*ctx_.mutable_stats_visit_dup_cnt());
    }
  }
  void stage(node_id_t id, size_t position, size_t slot) {
    if constexpr (Mode == HnswScanMode::Direct) {
      const void *vector = entity_.get_vector_ptr(id);
      neighbor_vecs_[slot] = vector;
      // Match the direct path's prefix of the original adjacency, rather
      // than the compacted list of unvisited nodes.
      if (position < prefetch_offset_) prefetch(vector);
      if (has_extra_values_)
        neighbor_extra_values_[slot] = ctx_.get_extra_values(vector);
    }
  }
  int score(size_t, const node_id_t *ids, size_t count, float *distances) {
    if constexpr (Mode != HnswScanMode::Direct) {
      neighbor_blocks_.clear();
      provider_blocks_.clear();
      int ret;
      if (use_provider_) {
        ret =
            dc_.get_vector(ids, static_cast<uint32_t>(count), provider_blocks_);
      } else {
        ret = entity_.get_vector_typed(ids, static_cast<uint32_t>(count),
                                       neighbor_blocks_);
      }
      if constexpr (Mode == HnswScanMode::Fallback) {
        if (ctx_.debugging()) ++(*ctx_.mutable_stats_get_vector());
      }
      if (ret != 0) return ret;
      for (size_t i = 0; i < count; ++i) {
        neighbor_vecs_[i] = use_provider_ ? provider_blocks_[i].data()
                                          : neighbor_blocks_[i].data();
        if (has_extra_values_)
          neighbor_extra_values_[i] = ctx_.get_extra_values(neighbor_vecs_[i]);
      }
      for (size_t i = 0; i < std::min(size_t{prefetch_offset_}, count); ++i) {
        prefetch(neighbor_vecs_[i]);
      }
    }
    dc_.batch_dist(neighbor_vecs_.data(), static_cast<uint32_t>(count),
                   distances,
                   has_extra_values_ ? neighbor_extra_values_.data() : nullptr);
    return 0;
  }

 private:
  void prefetch(const void *vector) const {
    const auto *p = static_cast<const char *>(vector);
    for (uint32_t cl = 0; cl < prefetch_lines_; ++cl)
      ailego_prefetch(p + cl * 64);
  }
  using Neighbors =
      decltype(std::declval<const EntityType &>().get_neighbors_typed(
          level_t{}, node_id_t{}));
  const EntityType &entity_;
  level_t level_;
  HnswDistCalculator &dc_;
  HnswContext &ctx_;
  uint32_t prefetch_lines_, prefetch_offset_;
  bool has_extra_values_, use_provider_;
  Neighbors neighbors_;
  std::vector<const void *> neighbor_vecs_, neighbor_extra_values_;
  std::vector<MemBlockType> neighbor_blocks_;
  std::vector<IndexStorage::MemoryBlock> provider_blocks_;
};

template <HnswScanMode Mode, class EntityType, class HeapType, class Visit>
void fast_search_neighbors(const EntityType &entity, HeapType &pool,
                           Visit visit, HnswDistCalculator &dc,
                           HnswContext *ctx, node_id_t entry_point,
                           dist_t entry_dist, uint32_t prefetch_lines,
                           uint32_t prefetch_offset) {
  visit.set_visited(entry_point);
  pool.push_block(&entry_dist, &entry_point, 1);
  HnswExactGraphScan<EntityType, typename EntityType::MemoryBlock, Mode> scan(
      entity, 0, dc, *ctx, prefetch_lines, prefetch_offset);
  GraphSearchPoolFrontier<HeapType> frontier{pool};
  GraphSearchScratch scratch;
  // Keep the existing fast-path handling of failed buffer reads: end traversal
  // with the retained pool; the caller also checks its distance-calculator
  // error.
  (void)SearchGraph(scan, frontier, visit, scratch);
}

// The fallback retains its separate candidate and result heaps, filtering,
// exact-distance stopping rule, and best-entry update for construction.
template <typename EntityType, typename MemBlockType, typename FilterFn>
void dual_heap_search_neighbors(const EntityType &entity, level_t level,
                                node_id_t *entry_point, dist_t *dist,
                                TopkHeap &topk, HnswContext *ctx,
                                HnswDistCalculator &dc, FilterFn &&filter) {
  const uint32_t prefetch_lines =
      ctx->pl() > 0 ? ctx->pl() : (entity.vector_size() + 63) / 64;
  VisitFilter &visit = ctx->visit_filter();
  CandidateHeap &candidates = ctx->candidates();
  candidates.clear();
  visit.clear();
  visit.set_visited(*entry_point);
  if (!filter(*entry_point)) topk.emplace(*entry_point, *dist);
  candidates.emplace(*entry_point, *dist);

  struct Frontier {
    CandidateHeap &candidates;
    TopkHeap &topk;
    HnswContext &ctx;
    node_id_t *entry_point;
    dist_t *dist;
    FilterFn &filter;
    bool has_next() const {
      return !candidates.empty() && !ctx.reach_scan_limit() &&
             !(topk.full() && candidates.begin()->second > topk[0].second);
    }
    node_id_t pop() {
      const node_id_t id = candidates.begin()->first;
      candidates.pop();
      return id;
    }
    void push_batch(const node_id_t *ids, const float *distances,
                    size_t count) {
      for (size_t i = 0; i < count; ++i) {
        const node_id_t id = ids[i];
        const dist_t distance = distances[i];
        if (!topk.full() || distance < topk[0].second) {
          candidates.emplace(id, distance);
          if (distance < *dist) {
            *entry_point = id;
            *dist = distance;
          }
          if (!filter(id)) topk.emplace(id, distance);
        }
      }
    }
  } frontier{candidates, topk, *ctx, entry_point, dist, filter};
  HnswExactGraphScan<EntityType, MemBlockType, HnswScanMode::Fallback> scan(
      entity, level, dc, *ctx, prefetch_lines, ctx->po());
  GraphSearchScratch scratch;
  (void)SearchGraph(scan, frontier, visit, scratch);
}

// ============================================================================
// search_neighbors: fallback dual-heap path used for construction, filtered
// queries, and upper levels for every backend.
// ============================================================================
template <typename EntityType>
void HnswAlgorithm<EntityType>::search_neighbors(level_t level,
                                                 node_id_t *entry_point,
                                                 dist_t *dist, TopkHeap &topk,
                                                 HnswContext *ctx) const {
  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  HnswDistCalculator &dc = ctx->dist_calculator();

  auto run_with_filter = [&](auto &&filter) {
    dual_heap_search_neighbors<EntityType, MemBlockType>(
        entity, level, entry_point, dist, topk, ctx, dc,
        std::forward<decltype(filter)>(filter));
  };

  if (ctx->filter().is_valid()) {
    auto filter = [&](node_id_t id) {
      return ctx->filter()(entity.get_key_typed(id));
    };
    run_with_filter(filter);
  } else {
    run_with_filter([](node_id_t) { return false; });
  }
}

template <typename EntityType>
int HnswAlgorithm<EntityType>::dispatch_search_neighbors(
    node_id_t entry_point, dist_t entry_dist, HnswContext *ctx) const {
  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  const uint32_t prefetch_lines =
      ctx->pl() > 0 ? ctx->pl() : (entity.vector_size() + 63) / 64;
  const bool dispatched =
      dispatch_visit_filter(ctx->visit_filter(), [&](auto visit) {
        visit.clear();
        ctx->search_heap().dispatch([&](auto &pool) {
          using Heap = std::decay_t<decltype(pool)>;
          if constexpr (!std::is_same_v<Heap, TopkHeap>) {
            if constexpr (std::is_same_v<MemBlockType, MmapMemoryBlock>) {
              fast_search_neighbors<HnswScanMode::Direct>(
                  entity, pool, visit, ctx->dist_calculator(), ctx, entry_point,
                  entry_dist, prefetch_lines, ctx->po());
            } else {
              fast_search_neighbors<HnswScanMode::Buffered>(
                  entity, pool, visit, ctx->dist_calculator(), ctx, entry_point,
                  entry_dist, prefetch_lines, ctx->po());
            }
          }
        });
      });
  if (ailego_unlikely(!dispatched)) {
    LOG_ERROR("Failed to dispatch HNSW visit filter, mode %d",
              ctx->visit_filter().get_mode());
    return IndexError_Runtime;
  }
  return 0;
}

template <typename EntityType>
void HnswAlgorithm<EntityType>::expand_neighbors_by_group(
    HnswContext *ctx) const {
  if (!ctx->group_by().is_valid()) {
    return;
  }

  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  std::function<std::string(node_id_t)> group_by = [&](node_id_t id) {
    return ctx->group_by()(entity.get_key_typed(id));
  };

  // devide into groups
  std::map<std::string, TopkHeap> &group_topk_heaps = ctx->group_topk_heaps();
  ctx->search_heap().for_each([&](node_id_t id, dist_t score) {
    std::string group_id = group_by(id);

    auto &topk_heap = group_topk_heaps[group_id];
    if (topk_heap.empty()) {
      topk_heap.limit(ctx->group_topk());
    }
    topk_heap.emplace(id, score);
    return true;
  });

  // stage 2, expand to reach group num as possible
  if (group_topk_heaps.size() < ctx->group_num()) {
    VisitFilter &visit = ctx->visit_filter();
    CandidateHeap &candidates = ctx->candidates();
    HnswDistCalculator &dc = ctx->dist_calculator();

    std::function<bool(node_id_t)> filter = [](node_id_t) { return false; };
    if (ctx->filter().is_valid()) {
      filter = [&](node_id_t id) {
        return ctx->filter()(entity.get_key_typed(id));
      };
    }

    // refill to get enough groups
    candidates.clear();
    visit.clear();
    ctx->search_heap().for_each([&](node_id_t id, dist_t score) {
      visit.set_visited(id);
      candidates.emplace(id, score);
      return true;
    });

    // do expand
    while (!candidates.empty() && !ctx->reach_scan_limit()) {
      auto top = candidates.begin();
      node_id_t main_node = top->first;

      candidates.pop();
      const auto neighbors = entity.get_neighbors_typed(0, main_node);
      if (ailego_unlikely(ctx->debugging())) {
        (*ctx->mutable_stats_get_neighbors())++;
      }

      std::vector<node_id_t> neighbor_ids(neighbors.size());
      uint32_t size = 0;
      for (uint32_t i = 0; i < neighbors.size(); ++i) {
        node_id_t node = neighbors[i];
        if (visit.visited(node)) {
          if (ailego_unlikely(ctx->debugging())) {
            (*ctx->mutable_stats_visit_dup_cnt())++;
          }
          continue;
        }
        visit.set_visited(node);
        neighbor_ids[size++] = node;
      }
      if (size == 0) {
        continue;
      }

      std::vector<MemBlockType> neighbor_vec_blocks;
      int ret = entity.get_vector_typed(neighbor_ids.data(), size,
                                        neighbor_vec_blocks);
      if (ailego_unlikely(ctx->debugging())) {
        (*ctx->mutable_stats_get_vector())++;
      }
      if (ailego_unlikely(ret != 0)) {
        break;
      }

      std::vector<float> dists(size);
      std::vector<const void *> neighbor_vecs(size);
      const bool has_extra_values = ctx->has_extra_values();
      std::vector<const void *> neighbor_extra_values(has_extra_values ? size
                                                                       : 0);
      for (uint32_t i = 0; i < size; ++i) {
        neighbor_vecs[i] = neighbor_vec_blocks[i].data();
        if (has_extra_values) {
          neighbor_extra_values[i] = ctx->get_extra_values(neighbor_vecs[i]);
        }
      }
      dc.batch_dist(neighbor_vecs.data(), size, dists.data(),
                    has_extra_values ? neighbor_extra_values.data() : nullptr);

      for (uint32_t i = 0; i < size; ++i) {
        node_id_t node = neighbor_ids[i];
        dist_t cur_dist = dists[i];

        if (!filter(node)) {
          std::string group_id = group_by(node);

          auto &topk_heap = group_topk_heaps[group_id];
          if (topk_heap.empty()) {
            topk_heap.limit(ctx->group_topk());
          }
          topk_heap.emplace(node, cur_dist);

          if (group_topk_heaps.size() >= ctx->group_num()) {
            break;
          }
        }

        candidates.emplace(node, cur_dist);
      }
    }
  }
}

template <typename EntityType>
void HnswAlgorithm<EntityType>::update_neighbors(HnswDistCalculator &dc,
                                                 node_id_t id, level_t level,
                                                 TopkHeap &topk_heap,
                                                 HnswContext *ctx) {
  topk_heap.sort();

  uint32_t max_neighbor_cnt = entity_.neighbor_cnt(level);
  if (topk_heap.size() <= static_cast<size_t>(entity_.prune_cnt())) {
    if (topk_heap.size() <= static_cast<size_t>(max_neighbor_cnt)) {
      entity_.update_neighbors(level, id, topk_heap.container());
      return;
    }
  }

  uint32_t cur_size = 0;
  if constexpr (std::is_same_v<EntityType, HnswBufferPoolStreamerEntity>) {
    const auto &read_entity =
        static_cast<const EntityType &>(ctx->get_entity());
    auto &prune_ids = ctx->prune_ids();
    auto &prune_blocks = ctx->prune_blocks();
    auto &selected_indices = ctx->prune_selected_indices();

    prune_ids.clear();
    prune_ids.reserve(topk_heap.size());
    for (const auto &candidate : topk_heap) {
      prune_ids.emplace_back(candidate.first);
    }
    prune_blocks.clear();
    // The active build metric may describe the provider's original vectors,
    // not the representation stored in the BufferPool entity.
    const int ret =
        dc.has_provider()
            ? dc.get_vector(prune_ids.data(), prune_ids.size(), prune_blocks)
            : read_entity.get_vector_for_prune(prune_ids.data(),
                                               prune_ids.size(), prune_blocks);
    if (ailego_unlikely(ret != 0)) {
      dc.set_error();
      return;
    }
    selected_indices.clear();
    selected_indices.reserve(max_neighbor_cnt);

    for (size_t i = 0; i < topk_heap.size(); ++i) {
      node_id_t cur_node = topk_heap[i].first;
      dist_t cur_node_dist = topk_heap[i].second;
      bool good = true;
      for (const size_t selected : selected_indices) {
        const dist_t pair_distance =
            dc.dist(prune_blocks[i].data(), prune_blocks[selected].data());
        if (pair_distance <= cur_node_dist) {
          good = false;
          break;
        }
      }

      if (good) {
        topk_heap.mutable_at(cur_size).first = cur_node;
        topk_heap.mutable_at(cur_size).second = cur_node_dist;
        selected_indices.emplace_back(i);
        cur_size++;
        if (cur_size >= max_neighbor_cnt) {
          break;
        }
      }
    }
    prune_blocks.clear();
  } else {
    for (size_t i = 0; i < topk_heap.size(); ++i) {
      node_id_t cur_node = topk_heap[i].first;
      dist_t cur_node_dist = topk_heap[i].second;
      bool good = true;
      for (uint32_t j = 0; j < cur_size; ++j) {
        dist_t tmp_dist = dc.dist(cur_node, topk_heap[j].first);
        if (tmp_dist <= cur_node_dist) {
          good = false;
          break;
        }
      }

      if (good) {
        topk_heap.mutable_at(cur_size).first = cur_node;
        topk_heap.mutable_at(cur_size).second = cur_node_dist;
        cur_size++;
        if (cur_size >= max_neighbor_cnt) {
          break;
        }
      }
    }
  }

  // when after-prune neighbor count is too seldom,
  // we use this strategy to make-up enough edges
  // not only just make-up out-degrees
  // we also make-up enough in-degrees
  uint32_t min_neighbors = entity_.min_neighbor_cnt();
  for (size_t k = cur_size; cur_size < min_neighbors && k < topk_heap.size();
       ++k) {
    bool exist = false;
    for (size_t j = 0; j < cur_size; ++j) {
      if (topk_heap[j].first == topk_heap[k].first) {
        exist = true;
        break;
      }
    }
    if (!exist) {
      topk_heap.mutable_at(cur_size).first = topk_heap[k].first;
      topk_heap.mutable_at(cur_size).second = topk_heap[k].second;
      cur_size++;
    }
  }

  topk_heap.truncate(cur_size);
  entity_.update_neighbors(level, id, topk_heap.container());

  return;
}

template <typename EntityType>
void HnswAlgorithm<EntityType>::reverse_update_neighbors(
    HnswDistCalculator &dc, node_id_t id, level_t level, node_id_t link_id,
    dist_t dist, TopkHeap &update_heap, HnswContext *ctx) {
  const size_t max_neighbor_cnt = entity_.neighbor_cnt(level);

  const uint32_t lock_idx = id & kLockMask;
  std::unique_lock<std::mutex> node_lock(lock_pool_[lock_idx]);
  const Neighbors neighbors = entity_.get_neighbors(level, id);
  const size_t size = neighbors.size();
  ailego_assert_with(size <= max_neighbor_cnt, "invalid neighbor size");
  if (size < max_neighbor_cnt) {
    entity_.add_neighbor(level, id, size, link_id);
    return;
  }

  if constexpr (std::is_same_v<EntityType, HnswBufferPoolStreamerEntity>) {
    const auto &read_entity =
        static_cast<const EntityType &>(ctx->get_entity());
    // A wide level-0 list can require O(M^2) pairwise comparisons.  Resolving
    // each pair through HnswDistCalculator::dist(node_id, node_id) repeatedly
    // pins and releases the same BufferStorage pages and dominates high-M
    // construction.  Resolve the center and every candidate once, keep those
    // blocks alive for the prune, and preserve the existing scalar pruning
    // order exactly.
    auto &candidate_ids = ctx->prune_ids();
    auto &candidate_blocks = ctx->prune_blocks();

    candidate_ids.clear();
    candidate_ids.reserve(size + 2);
    candidate_ids.emplace_back(id);
    candidate_ids.emplace_back(link_id);
    for (size_t i = 0; i < size; ++i) {
      candidate_ids.emplace_back(neighbors[i]);
    }

    candidate_blocks.clear();
    const int ret =
        dc.has_provider()
            ? dc.get_vector(candidate_ids.data(), candidate_ids.size(),
                            candidate_blocks)
            : read_entity.get_vector_for_prune(
                  candidate_ids.data(), candidate_ids.size(), candidate_blocks);
    if (ailego_unlikely(ret != 0)) {
      dc.set_error();
      return;
    }

    update_heap.clear();
    // Use block indices until pruning is complete.  This keeps sorted heap
    // entries directly associated with their pinned vectors without a hash
    // map.
    update_heap.emplace(1, dist);
    for (size_t i = 0; i < size; ++i) {
      const dist_t candidate_distance =
          dc.dist(candidate_blocks[0].data(), candidate_blocks[i + 2].data());
      update_heap.emplace(static_cast<node_id_t>(i + 2), candidate_distance);
    }
    update_heap.sort();

    size_t selected_count = 0;
    for (size_t i = 0; i < update_heap.size(); ++i) {
      const node_id_t candidate_block_index = update_heap[i].first;
      const dist_t candidate_distance = update_heap[i].second;
      bool good = true;
      if (selected_count != 0) {
        for (size_t j = 0; j < selected_count; ++j) {
          const dist_t pair_distance =
              dc.dist(candidate_blocks[candidate_block_index].data(),
                      candidate_blocks[update_heap[j].first].data());
          if (pair_distance <= candidate_distance) {
            good = false;
            break;
          }
        }
      }

      if (good) {
        update_heap.mutable_at(selected_count) = {candidate_block_index,
                                                  candidate_distance};
        ++selected_count;
        if (selected_count >= max_neighbor_cnt) {
          break;
        }
      }
    }

    update_heap.truncate(selected_count);
    for (auto &selected : update_heap.mutable_container()) {
      selected.first = candidate_ids[selected.first];
    }
    entity_.update_neighbors(level, id, update_heap.container());

    node_lock.unlock();
    update_heap.clear();
    // Release BufferStorage pins before this worker consumes its next node.
    candidate_blocks.clear();
  } else {
    update_heap.emplace(link_id, dist);
    for (size_t i = 0; i < size; ++i) {
      node_id_t node = neighbors[i];
      dist_t cur_dist = dc.dist(id, node);
      update_heap.emplace(node, cur_dist);
    }

    update_heap.sort();
    size_t cur_size = 0;
    for (size_t i = 0; i < update_heap.size(); ++i) {
      node_id_t cur_node = update_heap[i].first;
      dist_t cur_node_dist = update_heap[i].second;
      bool good = true;
      for (size_t j = 0; j < cur_size; ++j) {
        dist_t tmp_dist = dc.dist(cur_node, update_heap[j].first);
        if (tmp_dist <= cur_node_dist) {
          good = false;
          break;
        }
      }

      if (good) {
        update_heap.mutable_at(cur_size).first = cur_node;
        update_heap.mutable_at(cur_size).second = cur_node_dist;
        cur_size++;
        if (cur_size >= max_neighbor_cnt) {
          break;
        }
      }
    }

    update_heap.truncate(cur_size);
    entity_.update_neighbors(level, id, update_heap.container());
    update_heap.clear();
  }

  return;
}

// Explicit template instantiation
template class HnswAlgorithm<HnswMmapStreamerEntity>;
template class HnswAlgorithm<HnswBufferPoolStreamerEntity>;
template class HnswAlgorithm<HnswContiguousStreamerEntity>;
template class HnswAlgorithm<HnswExternalStreamerEntity>;

}  // namespace core
}  // namespace zvec
