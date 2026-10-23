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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace zvec::core {

// Portable symmetric SQ8, calibrated independently per neighbor. Each lane
// stores a float scale followed by dimension signed bytes. No training model
// or ISA-specific kernel is needed. Cosine inputs must already be normalized,
// just as for exact HNSW scoring; zero vectors remain zero.
class SQ8GraphCodec {
 public:
  static constexpr size_t kBatchSize = 16;
  using Query = std::vector<float>;

  explicit SQ8GraphCodec(size_t dimension, bool cosine = false)
      : dimension_(dimension), cosine_(cosine) {}
  size_t encoded_dim() const {
    return dimension_;
  }
  size_t batch_bytes() const {
    return kBatchSize * (sizeof(float) + dimension_);
  }
  void transform(const void *vector, std::vector<float> &out) const {
    const auto *values = static_cast<const float *>(vector);
    out.assign(values, values + dimension_);
  }
  void encode(const float *, const float *vectors, size_t count,
              char *codes) const {
    // Initialize inactive lanes too: scan always processes a full batch.
    std::memset(codes, 0, batch_bytes());
    for (size_t i = 0; i < count; ++i) {
      const float *v = vectors + i * dimension_;
      float max_abs = 0;
      for (size_t d = 0; d < dimension_; ++d) {
        max_abs = std::max(max_abs, std::abs(v[d]));
      }
      const float scale =
          std::max(max_abs / 127.0f, std::numeric_limits<float>::min());
      char *lane = codes + i * (sizeof(float) + dimension_);
      std::memcpy(lane, &scale, sizeof(scale));
      for (size_t d = 0; d < dimension_; ++d) {
        const float rounded = std::round(v[d] / scale);
        const int8_t value =
            static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, rounded)));
        std::memcpy(lane + sizeof(float) + d, &value, sizeof(value));
      }
    }
  }
  void prepare_query(const void *vector, Query &query) const {
    transform(vector, query);
  }
  void scan(const char *codes, Query &query, float, float *distances) const {
    for (size_t i = 0; i < kBatchSize; ++i) {
      const char *lane = codes + i * (sizeof(float) + dimension_);
      float scale;
      std::memcpy(&scale, lane, sizeof(scale));
      float score = 0;
      for (size_t d = 0; d < dimension_; ++d) {
        int8_t value;
        std::memcpy(&value, lane + sizeof(float) + d, sizeof(value));
        const float reconstructed = scale * value;
        if (cosine_) {
          score += query[d] * reconstructed;
        } else {
          const float diff = query[d] - reconstructed;
          score += diff * diff;
        }
      }
      distances[i] = cosine_ ? 1.0f - score : score;
    }
  }

 private:
  size_t dimension_;
  bool cosine_;
};
}  // namespace zvec::core
