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
#include <cmath>
#include <limits>
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

  auto next = std::make_shared<Block>();
  IndexStorage::MemoryBlock raw;
  int ret = entity.get_vector(id, raw);
  if (ret != 0) return ret;
  if (raw.data() == nullptr) {
    return IndexError_Mismatch;
  }
  const auto *center = static_cast<const float *>(raw.data());
  next->center.assign(center, center + dimension_);
  std::vector<float> rotated_center;
  rotation_.rotate(next->center.data(), rotated_center);

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
  next->codes.resize(
      ((next->neighbors.size() + batch_size - 1) / batch_size) * batch_bytes,
      0);
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
        next->codes.data() + (offset / batch_size) * batch_bytes,
        rabitqlib::METRIC_L2);
  }

  // Concurrent readers may have built the same block. Publish only one copy.
  std::lock_guard<std::mutex> lock(mutex_);
  block = blocks_.emplace(id, std::move(next)).first->second;
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
    const float exact = ctx.dist_calculator().dist(block->center.data());
    if (!ctx.filter().is_valid() || !ctx.filter()(entity.get_key(id))) {
      results.emplace(id, exact);
    }
    batch_query.set_g_add(exact);
    if (ctx.debugging()) ++(*ctx.mutable_stats_get_neighbors());
    for (size_t offset = 0; offset < block->neighbors.size();
         offset += batch_size) {
      ScanSymphonyQGBatch(
          block->codes.data() + (offset / batch_size) * batch_bytes,
          batch_query, rotation_.padded_dim(), distances.data());
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
int HnswSymphonyQG::search(node_id_t, HnswContext &) const {
  return IndexError_Unsupported;
}
#endif
}  // namespace zvec::core
