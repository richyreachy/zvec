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
#include <random>
#include <stdexcept>
#include <vector>
#include <turbo/distance/common/symqg_kernels.h>

namespace zvec::turbo {
// One view preserves the historical randomized orthogonal transform exactly.
// Multiple independently signed views form an isometric frame when each view
// is scaled by 1/sqrt(views). This adds quantization directions, not source
// dimensions. Views are explicit model metadata, never process environment.
class SymQGRotation {
 public:
  static constexpr size_t kPasses = 1;
  static bool valid_views(size_t views) {
    return views == 1 || views == 2 || views == 4;
  }
  explicit SymQGRotation(size_t dimension, size_t views = 1)
      : dimension_(dimension), views_(views) {
    if (!dimension || dimension > 4096 || !valid_views(views))
      throw std::invalid_argument("invalid SymQG rotation dimension or views");
    padded_ = 64;
    while (padded_ < dimension) padded_ *= 2;
    if (dimension == 768) padded_ = 768;
    base_padded_ = padded_;
    padded_ *= views_;
    const float scale = 1.0f / std::sqrt(static_cast<float>(padded_));
    std::mt19937 rng(0x535147U);
    signs_.resize(kPasses * padded_);
    for (auto &sign : signs_) sign = ((rng() & 1U) ? 1.0f : -1.0f) * scale;
  }
  size_t padded_dim() const {
    return padded_;
  }
  size_t views() const {
    return views_;
  }
  void rotate(const void *data, std::vector<float> &out,
              bool fast_query = false) const {
    const auto *values = static_cast<const float *>(data);
    out.assign(padded_, 0.0f);
    for (size_t view = 0; view < views_; ++view) {
      float *part = out.data() + view * base_padded_;
      std::copy_n(values, dimension_, part);
      for (size_t pass = 0; pass < kPasses; ++pass) {
        const float *signs =
            signs_.data() + pass * padded_ + view * base_padded_;
        if (!symqg_mul(signs, part, base_padded_)) {
          for (size_t i = 0; i < base_padded_; ++i) part[i] *= signs[i];
        }
        if (base_padded_ == 768) {
          // Query reassociation may change rounding. Keep the canonical path
          // for database/model transforms and byte-exact model validation.
          if (fast_query && symqg_hadamard768(part)) continue;
          // Paley H12 tensor Sylvester H64: H H^T = 768 I.
          // The normalization is already included in signs_. No zero padding.
          for (size_t block = 0; block < 12; ++block) {
            float *v = part + block * 64;
            if (!symqg_hadamard(v, 64)) {
              for (size_t stride = 1; stride < 64; stride *= 2)
                for (size_t offset = 0; offset < 64; offset += 2 * stride)
                  for (size_t k = 0; k < stride; ++k) {
                    const float a = v[offset + k], b = v[offset + k + stride];
                    v[offset + k] = a + b;
                    v[offset + k + stride] = a - b;
                  }
            }
          }
          static constexpr int8_t h[12][12] = {
              {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
              {1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1},
              {1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1},
              {1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1},
              {1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1},
              {1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1},
              {1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1},
              {1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1},
              {1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1},
              {1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1},
              {1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1},
              {1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1}};
          std::array<float, 768> mixed{};
          for (size_t i = 0; i < 12; ++i)
            for (size_t j = 0; j < 12; ++j) {
              if (h[i][j] > 0)
                for (size_t k = 0; k < 64; ++k)
                  mixed[i * 64 + k] += part[j * 64 + k];
              else
                for (size_t k = 0; k < 64; ++k)
                  mixed[i * 64 + k] -= part[j * 64 + k];
            }
          std::copy(mixed.begin(), mixed.end(), part);
          continue;
        }
        if (!symqg_hadamard(part, base_padded_)) {
          for (size_t stride = 1; stride < base_padded_; stride *= 2) {
            for (size_t offset = 0; offset < base_padded_;
                 offset += 2 * stride) {
              for (size_t i = 0; i < stride; ++i) {
                float a = part[offset + i], b = part[offset + i + stride];
                part[offset + i] = a + b;
                part[offset + i + stride] = a - b;
              }
            }
          }
        }
      }
    }
  }

 private:
  size_t dimension_, padded_, base_padded_, views_;
  std::vector<float> signs_;
};

}  // namespace zvec::turbo
