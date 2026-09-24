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
#include <thread>
#if RABITQ_SUPPORTED
#include <rabitqlib/quantization/rabitq.hpp>
#include "symphony_qg_utils.h"
#endif

namespace zvec::core {

void HnswSymphonyQG::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  blocks_.clear();
}

#if RABITQ_SUPPORTED

int HnswSymphonyQG::get_block(const HnswEntity &entity, node_id_t id,
                              std::shared_ptr<const Block> &block) const {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blocks_.find(id);
    if (it != blocks_.end()) {
      block = it->second;
      return 0;
    }
  }

  // Blocks dropped by clear() after a graph mutation are rebuilt lazily, so
  // search stays correct at the cost of the original per-node build.
  auto next = std::make_shared<Block>();
  IndexStorage::MemoryBlock raw;
  int ret = entity.get_vector(id, raw);
  if (ret != 0) return ret;
  if (raw.data() == nullptr) {
    return IndexError_Mismatch;
  }
  next->center = static_cast<const float *>(raw.data());
  const auto neighbors = entity.get_neighbors(0, id);
  next->neighbors.reserve(neighbors.size());
  for (uint32_t i = 0; i < neighbors.size(); ++i) {
    if (entity.get_key(neighbors[i]) != kInvalidKey) {
      next->neighbors.push_back(neighbors[i]);
    }
  }

  const size_t dim = rotation_.padded_dim();
  constexpr size_t batch_size = rabitqlib::fastscan::kBatchSize;
  const size_t batch_bytes = rabitqlib::QGBatchDataMap<float>::data_bytes(dim);
  auto codes = std::make_shared<std::vector<char>>(
      ((next->neighbors.size() + batch_size - 1) / batch_size) * batch_bytes,
      0);
  std::vector<float> rotated_center;
  rotation_.rotate(next->center, rotated_center);
  std::vector<float> rotated(batch_size * dim);
  std::vector<float> vector;
  for (size_t offset = 0; offset < next->neighbors.size();
       offset += batch_size) {
    const size_t count = std::min(batch_size, next->neighbors.size() - offset);
    for (size_t i = 0; i < count; ++i) {
      IndexStorage::MemoryBlock neighbor;
      ret = entity.get_vector(next->neighbors[offset + i], neighbor);
      if (ret != 0) return ret;
      if (neighbor.data() == nullptr) {
        return IndexError_Mismatch;
      }
      rotation_.rotate(neighbor.data(), vector);
      std::copy(vector.begin(), vector.end(), rotated.begin() + i * dim);
    }
    rabitqlib::quant::quantize_qg_batch(
        rotated.data(), rotated_center.data(), count, dim,
        codes->data() + (offset / batch_size) * batch_bytes,
        rabitqlib::METRIC_L2);
  }
  next->codes = std::shared_ptr<const char>(codes, codes->data());

  // Concurrent readers may have built the same block. Publish only one copy.
  std::lock_guard<std::mutex> lock(mutex_);
  block = blocks_.emplace(id, std::move(next)).first->second;
  return 0;
}

int HnswSymphonyQG::prebuild(const HnswEntity &entity, size_t doc_cnt,
                             size_t threads) {
  const size_t dim = rotation_.padded_dim();
  constexpr size_t batch_size = rabitqlib::fastscan::kBatchSize;
  const size_t batch_bytes = rabitqlib::QGBatchDataMap<float>::data_bytes(dim);

  std::vector<std::shared_ptr<Block>> blocks(doc_cnt);

  // Pass 1 (single thread): collect neighbor lists; codes space is sized
  // exactly. Center pointers reference the mmap-ed vector segment and stay
  // valid for the lifetime of the streamer.
  struct BlockPlan {
    const float *center;
    std::vector<node_id_t> neighbors;
  };
  std::vector<BlockPlan> plans(doc_cnt);
  size_t total_codes = 0;
  for (node_id_t id = 0; id < doc_cnt; ++id) {
    if (entity.get_key(id) == kInvalidKey) continue;
    IndexStorage::MemoryBlock raw;
    int ret = entity.get_vector(id, raw);
    if (ret != 0 || raw.data() == nullptr) continue;
    plans[id].center = static_cast<const float *>(raw.data());
    const auto neighbors = entity.get_neighbors(0, id);
    plans[id].neighbors.reserve(neighbors.size());
    for (uint32_t i = 0; i < neighbors.size(); ++i) {
      if (entity.get_key(neighbors[i]) != kInvalidKey) {
        plans[id].neighbors.push_back(neighbors[i]);
      }
    }
    total_codes +=
        ((plans[id].neighbors.size() + batch_size - 1) / batch_size) *
        batch_bytes;
  }

  auto codes_arena = std::make_shared<std::vector<char>>();
  codes_arena->resize(total_codes);

  // Pass 2 (parallel): rotate + quantize every neighbor batch.
  std::atomic<node_id_t> next_id{0};
  std::atomic<size_t> arena_cursor{0};
  std::atomic<int> first_error{0};

  auto worker = [&]() {
    std::vector<float> rotated_center;
    std::vector<float> rotated(batch_size * dim);
    std::vector<float> vector;
    while (first_error.load() == 0) {
      const node_id_t id = next_id.fetch_add(1);
      if (id >= doc_cnt) break;
      if (plans[id].center == nullptr || plans[id].neighbors.empty()) continue;

      auto block = std::make_shared<Block>();
      block->center = plans[id].center;
      block->neighbors = std::move(plans[id].neighbors);
      const size_t num_batches =
          (block->neighbors.size() + batch_size - 1) / batch_size;
      char *codes = codes_arena->data() +
                    arena_cursor.fetch_add(num_batches * batch_bytes);
      rotation_.rotate(block->center, rotated_center);
      for (size_t offset = 0; offset < block->neighbors.size();
           offset += batch_size) {
        const size_t count =
            std::min(batch_size, block->neighbors.size() - offset);
        for (size_t i = 0; i < count; ++i) {
          IndexStorage::MemoryBlock neighbor;
          const int ret =
              entity.get_vector(block->neighbors[offset + i], neighbor);
          if (ret != 0 || neighbor.data() == nullptr) {
            first_error.store(ret != 0 ? ret : IndexError_Mismatch);
            return;
          }
          rotation_.rotate(neighbor.data(), vector);
          std::copy(vector.begin(), vector.end(), rotated.begin() + i * dim);
        }
        rabitqlib::quant::quantize_qg_batch(
            rotated.data(), rotated_center.data(), count, dim,
            codes + (offset / batch_size) * batch_bytes, rabitqlib::METRIC_L2);
      }
      // Aliasing constructor keeps the arena alive for the block's lifetime.
      block->codes = std::shared_ptr<const char>(codes_arena, codes);
      blocks[id] = std::move(block);
    }
  };

  std::vector<std::thread> pool;
  const size_t nthreads =
      threads == 0 ? std::max(1u, std::thread::hardware_concurrency())
                   : threads;
  for (size_t i = 0; i < nthreads; ++i) pool.emplace_back(worker);
  for (auto &t : pool) t.join();
  if (first_error.load() != 0) {
    return first_error.load();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (node_id_t id = 0; id < doc_cnt; ++id) {
    if (blocks[id]) {
      blocks_[id] = std::move(blocks[id]);
    }
  }
  return 0;
}

int HnswSymphonyQG::search(node_id_t entry, HnswContext &ctx) const {
  const auto &entity = ctx.get_entity();
  auto &results =
      ctx.search_heap().reset<TopkHeap>(std::max(ctx.topk(), ctx.ef()));
  results.clear();
  auto &visited = ctx.visit_filter();
  visited.clear();
  SymphonyQGBeam beam(std::max(size_t{1}, results.limit()));
  beam.insert(entry, std::numeric_limits<float>::max());
  std::vector<float> rotated_query;
  rotation_.rotate(ctx.dist_calculator().query(), rotated_query);
  rabitqlib::BatchQuery<float> batch_query(rotated_query.data(),
                                           rotation_.padded_dim());
  constexpr size_t batch_size = rabitqlib::fastscan::kBatchSize;
  const size_t batch_bytes =
      rabitqlib::QGBatchDataMap<float>::data_bytes(rotation_.padded_dim());
  std::array<float, batch_size> distances;
  std::vector<node_id_t> expanded;
  while (beam.has_next() && !ctx.reach_scan_limit()) {
    const node_id_t id = beam.pop();
    // Local estimates for the same neighbor differ by center. Like SymphonyQG,
    // mark nodes on expansion, allowing a better estimate to re-enter the beam.
    if (visited.visited(id)) continue;
    visited.set_visited(id);
    expanded.push_back(id);
    std::shared_ptr<const Block> block;
    int ret = get_block(entity, id, block);
    if (ret != 0) return ret;
    const float exact = ctx.dist_calculator().dist(block->center);
    if (!ctx.filter().is_valid() || !ctx.filter()(entity.get_key(id))) {
      results.emplace(id, exact);
    }
    batch_query.set_g_add(exact);
    if (ctx.debugging()) ++(*ctx.mutable_stats_get_neighbors());
    const char *codes = block->codes.get();
    for (size_t offset = 0; offset < block->neighbors.size();
         offset += batch_size) {
      ScanSymphonyQGBatch(codes + (offset / batch_size) * batch_bytes,
                          batch_query, rotation_.padded_dim(),
                          distances.data());
      const size_t count =
          std::min(batch_size, block->neighbors.size() - offset);
      for (size_t i = 0; i < count; ++i) {
        const node_id_t neighbor = block->neighbors[offset + i];
        if (!visited.visited(neighbor) && std::isfinite(distances[i])) {
          beam.insert(neighbor, distances[i]);
        }
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
      if (ctx.filter().is_valid() && !ctx.filter()(key)) continue;
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
int HnswSymphonyQG::search(node_id_t, HnswContext &) const {
  return IndexError_Unsupported;
}
#endif
}  // namespace zvec::core
