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

#include <cstdint>
#include <vector>
#include "quantizer/rotator/rotator.h"

namespace zvec {
namespace turbo {

//! Fast Walsh-Hadamard rotation (RaBitQ-style). The input is zero-padded to
//! the next power of two, sign-flipped by a seed-derived Rademacher vector, and
//! transformed by an in-place FWHT. out_dim() is the padded power-of-two size,
//! so it equals in_dim() only when in_dim() is already a power of two.
class FhtRotator : public Rotator {
 public:
  explicit FhtRotator(int in_dim = 0, uint64_t seed = 0x9E3779B97F4A7C15ull);
  ~FhtRotator() override = default;

  RotatorType type() const override {
    return RotatorType::kFht;
  }

  int in_dim() const override {
    return in_dim_;
  }

  int out_dim() const override {
    return out_dim_;
  }

  void apply(const float *in, float *out) const override;

  void apply_inverse(const float *in, float *out) const override;

  void train(const void *data, size_t num, size_t stride) override;

  int serialize(std::string *out) const override;

  int deserialize(const void *data, size_t len) override;

 private:
  //! Regenerate signs_ (length out_dim_) from in_dim_ and seed_.
  void rebuild();

  int in_dim_{0};
  int out_dim_{0};
  uint64_t seed_{0};
  //! Rademacher sign vector of length out_dim_ (+1 / -1).
  std::vector<float> signs_{};
};

}  // namespace turbo
}  // namespace zvec
