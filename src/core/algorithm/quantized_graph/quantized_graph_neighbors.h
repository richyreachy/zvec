// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>
#include <zvec/core/framework/index_error.h>

namespace zvec::core {
// Shared by codec training and block construction. Selection depends only on
// the source graph and stored coordinates, never on a query or learned codec.
template <class Graph>
int SelectQuantizedGraphNeighbors(const Graph &entity, uint32_t id,
                                  const float *center_ptr, size_t dimension,
                                  size_t max_neighbors,
                                  std::vector<uint32_t> &neighbors_out,
                                  std::vector<float> &selection_cache) {
  using NodeId = uint32_t;
  constexpr NodeId kInvalidNodeId = UINT32_MAX;
  neighbors_out.clear();
  const auto &neighbors = entity.neighbors(id);
  neighbors_out.reserve(neighbors.size());
  for (uint32_t i = 0; i < neighbors.size(); ++i) {
    if (entity.valid(neighbors[i])) {
      neighbors_out.push_back(neighbors[i]);
    }
  }
  const size_t original_neighbor_count = neighbors_out.size();
  // Offline two-hop candidate repair. Fixed row capacity and the exact same
  // query kernel are retained. No query or GT is used during construction.
  if (neighbors_out.size() < max_neighbors) {
    const size_t original = neighbors_out.size();
    std::array<NodeId, 512> seen;
    seen.fill(kInvalidNodeId);
    auto add = [&](NodeId v) {
      if (v == id || !entity.valid(v)) return;
      size_t h = (uint64_t(v) * 2654435761U) & 511U;
      while (seen[h] != kInvalidNodeId && seen[h] != v) h = (h + 1) & 511U;
      if (seen[h] == v) return;
      seen[h] = v;
      neighbors_out.push_back(v);
    };
    for (NodeId v : neighbors_out) {
      size_t h = (uint64_t(v) * 2654435761U) & 511U;
      while (seen[h] != kInvalidNodeId && seen[h] != v) h = (h + 1) & 511U;
      seen[h] = v;
    }
    const size_t seeds = std::min(original, size_t{8});
    for (size_t j = 0; j < seeds && neighbors_out.size() < 256; ++j) {
      const NodeId seed = neighbors_out[j * original / seeds];
      const auto children = entity.neighbors(seed);
      const size_t samples = std::min(size_t(children.size()), size_t{16});
      for (size_t k = 0; k < samples && neighbors_out.size() < 256; ++k)
        add(children[k * children.size() / samples]);
    }
  }
  // Trim to the block's degree bound like upstream SymphonyQG's fixed
  // degree_bound_, but with the HNSW diversity heuristic so the truncated
  // graph keeps short and long-range links and stays navigable: plain
  // truncation loses several recall points because HNSW's link order is
  // newest-first. Distances use squared L2 in coordinate space; for cosine
  // these are the normalized coordinates, with norm metadata excluded.
  const size_t kMaxBlockNeighbors = max_neighbors;
  if (neighbors_out.size() > kMaxBlockNeighbors) {
    const size_t cnt = neighbors_out.size();
    const auto *center = center_ptr;
    selection_cache.assign(cnt * dimension, 0.0f);
    std::vector<float> dists(cnt);
    for (size_t i = 0; i < cnt; ++i) {
      typename Graph::Vector raw;
      const int ret = entity.get_vector(neighbors_out[i], raw);
      if (ret != 0) {
        return ret;
      }
      if (raw.data() == nullptr) {
        return IndexError_Mismatch;
      }
      const auto *v = static_cast<const float *>(raw.data());
      std::copy(v, v + dimension, selection_cache.begin() + i * dimension);
      float d = 0.0f;
      for (size_t k = 0; k < dimension; ++k) {
        const float diff = v[k] - center[k];
        d += diff * diff;
      }
      dists[i] = d;
    }
    std::vector<uint32_t> order(cnt);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return dists[a] < dists[b]; });
    const float *cache = selection_cache.data();
    auto sq_dist = [&](uint32_t a, uint32_t b) {
      const float *x = cache + a * dimension;
      const float *y = cache + b * dimension;
      float d = 0.0f;
      for (size_t k = 0; k < dimension; ++k) {
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
    for (uint32_t idx : selected) pruned.push_back(neighbors_out[idx]);
    neighbors_out = std::move(pruned);
  }
  return 0;
}
}  // namespace zvec::core
