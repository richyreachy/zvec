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
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace zvec::core {
// Approximate ordering with a conservative bucket score for stopping.
// The range is scaled once by the estimated entry distance. Extreme bins retain
// every candidate; no item is dropped. Normal-bin ordering error <= width.
class QuantizedGraphBucket {
 public:
  static constexpr size_t N = 2048;
  static_assert(N >= 64 && N <= 4096 && N % 64 == 0);
  void reset(size_t) {
    for (size_t b : touched_) bins_[b].clear();
    touched_.clear();
    bits_.fill(0);
    summary_ = 0;
    first_ = N;
    width_ = 1.0f;
    inverse_width_ = 1.0f;
    finite_inverse_ = true;
  }
  void set_scale(float distance, float coefficient = .016f) {
    width_ = std::max(distance, 1e-6f) * coefficient;
    inverse_width_ = 1.0f / width_;
    finite_inverse_ = std::isfinite(inverse_width_);
  }
  float width() const {
    return width_;
  }
  size_t locate(float d) const {
    size_t b = 0;
    if (d >= 0) {
      // Extremely small widths can have an infinite reciprocal. Preserve
      // direct division there, including zero distances.
      const float x = finite_inverse_ ? d * inverse_width_ : d / width_;
      b = x >= float(N - 2) ? N - 1 : size_t(x) + 1;
    }
    return b;
  }
  // Scores stay in their original lane order; only bin arithmetic is batched.
  void locate_batch(const float *distances, uint32_t *bins,
                    size_t count) const {
    if (!finite_inverse_) {
      for (size_t i = 0; i < count; ++i) bins[i] = locate(distances[i]);
      return;
    }
    for (size_t i = 0; i < count; ++i) {
      const float x = distances[i] * inverse_width_;
      const float bounded = std::max(0.f, std::min(float(N - 2), x));
      bins[i] = distances[i] >= 0 ? uint32_t(bounded) + 1 : 0;
    }
  }
  float lower_bound(float d) const {
    size_t b = locate(d);
    return b == 0 ? -std::numeric_limits<float>::infinity()
                  : float(b - 1) * width_;
  }
  void insert(uint32_t id, float d) {
    insert_at(id, locate(d));
  }
  void insert_at(uint32_t id, size_t b) {
    if (bins_[b].empty()) touched_.push_back(b);
    bins_[b].push_back(id);
    bits_[b / 64] |= uint64_t{1} << (b % 64);
    summary_ |= uint64_t{1} << (b / 64);
    first_ = std::min(first_, b);
  }
  bool has_next() const {
    return summary_ != 0;
  }
  size_t first() const {
    return first_;
  }
  uint32_t next_id_at(size_t) const {
    return bins_[first()].back();
  }
  float next_distance() const {
    size_t b = first();
    return b == 0 ? -std::numeric_limits<float>::infinity()
                  : float(b - 1) * width_;
  }
  uint32_t pop() {
    size_t b = first();
    uint32_t id = bins_[b].back();
    bins_[b].pop_back();
    if (bins_[b].empty()) {
      bits_[b / 64] &= ~(uint64_t{1} << (b % 64));
      if (bits_[b / 64] == 0) summary_ &= ~(uint64_t{1} << (b / 64));
      if (summary_) {
        const size_t word = __builtin_ctzll(summary_);
        first_ = 64 * word + __builtin_ctzll(bits_[word]);
      } else {
        first_ = N;
      }
    }
    return id;
  }
  bool is_full(float) const {
    return false;
  }

 private:
  std::array<std::vector<uint32_t>, N> bins_;
  std::array<uint64_t, N / 64> bits_{};
  uint64_t summary_ = 0;
  size_t first_ = N;
  std::vector<size_t> touched_;
  float width_ = 1;
  float inverse_width_ = 1;
  bool finite_inverse_ = true;
};
}  // namespace zvec::core
