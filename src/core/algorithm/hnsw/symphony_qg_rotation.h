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
#include <cstdint>
#include <random>
#include <vector>
#include "symphony_qg_kernels.h"

namespace zvec::core {
// Randomized normalized Hadamard passes form an orthogonal rotation. A single
// pass matches upstream SymphonyQG's rotator; more passes improve the
// Gaussianity of rotated coordinates at proportional query-time cost.
// Fixed signs make derived codes reproducible after reopen without changing
// the HNSW storage format. Zero padding preserves original L2 distances.
// The 1/sqrt(padded) normalization is folded into the signs so the hot path
// is one element-wise multiply plus the transform.
class SymphonyQGRotation {
 public:
  static constexpr size_t kPasses = 1;
  explicit SymphonyQGRotation(size_t dimension) : dimension_(dimension) {
    padded_ = 64;
    while (padded_ < dimension) padded_ *= 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(padded_));
    std::mt19937 rng(0x535147U);
    signs_.resize(kPasses * padded_);
    for (auto &sign : signs_) sign = ((rng() & 1U) ? 1.0f : -1.0f) * scale;
  }
  size_t padded_dim() const {
    return padded_;
  }
  void rotate(const void *data, std::vector<float> &out) const {
    const auto *values = static_cast<const float *>(data);
    out.assign(padded_, 0.0f);
    std::copy(values, values + dimension_, out.begin());
    for (size_t pass = 0; pass < kPasses; ++pass) {
      const float *signs = signs_.data() + pass * padded_;
      if (!symqg_mul(signs, out.data(), padded_)) {
        for (size_t i = 0; i < padded_; ++i) out[i] *= signs[i];
      }
      if (!symqg_hadamard(out.data(), padded_)) {
        for (size_t stride = 1; stride < padded_; stride *= 2) {
          for (size_t offset = 0; offset < padded_; offset += 2 * stride) {
            for (size_t i = 0; i < stride; ++i) {
              float a = out[offset + i], b = out[offset + i + stride];
              out[offset + i] = a + b;
              out[offset + i + stride] = a - b;
            }
          }
        }
      }
    }
  }

 private:
  size_t dimension_, padded_;
  std::vector<float> signs_;
};
}  // namespace zvec::core
