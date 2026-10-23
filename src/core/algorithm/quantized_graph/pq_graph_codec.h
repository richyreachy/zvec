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
#include <cmath>
#include <random>
#include <vector>
#include <turbo/quantizer/pq_block_quantizer/pq_block_quantizer.h>
#include <zvec/core/framework/index_error.h>

namespace zvec::core {

// Graph-specific sampling and loading; the block format, training and distance
// computations belong to Turbo. The trained model remains a derived cache.
class PQGraphCodec : public turbo::PQBlockQuantizer {
 public:
  static constexpr size_t kMaxTrainVectors = 4096;
  using turbo::PQBlockQuantizer::PQBlockQuantizer;

  template <class Graph>
  int train(const Graph &graph, size_t count, size_t threads) {
    reset_training();
    // Reservoir sample IDs before loading vectors: memory stays bounded even
    // for large indexes and deleted slots never participate in training.
    std::vector<uint32_t> ids;
    std::mt19937 rng(42);
    size_t live = 0;
    for (size_t id = 0; id < count; ++id) {
      if (!graph.valid(static_cast<uint32_t>(id))) continue;
      ++live;
      if (ids.size() < kMaxTrainVectors) {
        ids.push_back(static_cast<uint32_t>(id));
      } else {
        const size_t slot =
            std::uniform_int_distribution<size_t>(0, live - 1)(rng);
        if (slot < ids.size()) ids[slot] = static_cast<uint32_t>(id);
      }
    }
    if (ids.empty()) return 0;
    auto holder =
        std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
            encoded_dim());
    // KMeans needs at least 16 samples. Repeat small corpora rather than add
    // artificial vectors which would distort their codebook.
    const size_t samples = std::max(size_t{16}, ids.size());
    std::vector<float> transformed;
    for (size_t i = 0; i < samples; ++i) {
      typename Graph::Vector raw;
      const int ret = graph.get_vector(ids[i % ids.size()], raw);
      if (ret != 0) return ret;
      if (!raw.data()) return IndexError_Mismatch;
      transform(raw.data(), transformed);
      ailego::NumericalVector<float> vector(encoded_dim());
      for (size_t d = 0; d < encoded_dim(); ++d) {
        if (!std::isfinite(transformed[d])) return IndexError_Mismatch;
        vector[d] = transformed[d];
      }
      holder->emplace(i, std::move(vector));
    }
    return turbo::PQBlockQuantizer::train(holder, threads);
  }
};
}  // namespace zvec::core
