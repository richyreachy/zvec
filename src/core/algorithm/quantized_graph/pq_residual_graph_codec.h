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
#include <random>
#include <utility>
#include <vector>
#include <turbo/quantizer/pq_residual_block_quantizer/pq_residual_block_quantizer.h>

namespace zvec::core {

// Sampling and vector ownership remain graph responsibilities; all residual
// arithmetic, training, packing and scanning are implemented in Turbo.
class PQResidualGraphCodec : public turbo::PQResidualBlockQuantizer {
 public:
  static constexpr size_t kMaxTrainVectors = 4096;
  using turbo::PQResidualBlockQuantizer::PQResidualBlockQuantizer;

  template <class Graph>
  int train(const Graph &graph, size_t count, size_t threads) {
    reset_training();
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    std::mt19937 rng(42);
    size_t seen = 0;
    bool has_live = false;
    for (size_t id = 0; id < count; ++id) {
      if (!graph.valid(static_cast<uint32_t>(id))) continue;
      has_live = true;
      const auto &neighbors = graph.neighbors(static_cast<uint32_t>(id));
      for (size_t i = 0; i < neighbors.size(); ++i) {
        const uint32_t neighbor = neighbors[i];
        if (neighbor >= count || !graph.valid(neighbor)) continue;
        ++seen;
        if (edges.size() < kMaxTrainVectors) {
          edges.emplace_back(static_cast<uint32_t>(id), neighbor);
        } else {
          const size_t slot =
              std::uniform_int_distribution<size_t>(0, seen - 1)(rng);
          if (slot < edges.size())
            edges[slot] = {static_cast<uint32_t>(id), neighbor};
        }
      }
    }
    if (!has_live) return 0;
    auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(
        encoded_dim());
    std::vector<float> residual(encoded_dim(), 0);
    // An isolated live node still needs exact search. A zero-residual codebook
    // handles its empty blocks and remains usable if edges are inserted later.
    const size_t samples = std::max(size_t{16}, edges.size());
    for (size_t i = 0; i < samples; ++i) {
      if (!edges.empty()) {
        const auto &edge = edges[i % edges.size()];
        typename Graph::Vector center, neighbor;
        int ret = graph.get_vector(edge.first, center);
        if (ret != 0) return ret;
        ret = graph.get_vector(edge.second, neighbor);
        if (ret != 0) return ret;
        if (!center.data() || !neighbor.data()) return IndexError_Mismatch;
        ret = transform_residual(center.data(), neighbor.data(), residual);
        if (ret != 0) return ret;
      }
      ailego::NumericalVector<float> vector(encoded_dim());
      std::copy(residual.begin(), residual.end(), vector.data());
      holder->emplace(i, std::move(vector));
    }
    return turbo::PQResidualBlockQuantizer::train(holder, threads);
  }
};
}  // namespace zvec::core
