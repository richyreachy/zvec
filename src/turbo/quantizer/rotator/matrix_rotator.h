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

#include <vector>
#include "quantizer/rotator/rotator.h"

namespace zvec {
namespace turbo {

//! Dense orthogonal rotation backed by an explicit dim x dim matrix
//! (OPQ-style). Dimension preserving: out_dim() == in_dim() == dim().
class MatrixRotator : public Rotator {
 public:
  explicit MatrixRotator(int dim = 0) : dim_(dim) {}
  ~MatrixRotator() override = default;

  RotatorType type() const override {
    return RotatorType::kMatrix;
  }

  int in_dim() const override {
    return dim_;
  }

  int out_dim() const override {
    return dim_;
  }

  void apply(const float *in, float *out) const override;

  void apply_inverse(const float *in, float *out) const override;

  void train(const void *data, size_t num, size_t stride) override;

  int serialize(std::string *out) const override;

  int deserialize(const void *data, size_t len) override;

 private:
  int dim_{0};
  //! Row-major dim_ x dim_ orthonormal matrix.
  std::vector<float> mat_{};
};

}  // namespace turbo
}  // namespace zvec
