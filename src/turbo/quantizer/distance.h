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

#include <cstddef>
#include <string>
#include <utility>
#include <zvec/turbo/turbo.h>

namespace zvec {
namespace turbo {

//! A callable distance handle bound to a quantized query vector.
//!
//! DistanceImpl owns the quantized query bytes and a dispatched
//! DistanceFunc. Invoking `operator()(candidate)` computes the distance
//! between the stored query and the given candidate vector, which is
//! expected to already be in the same quantized layout.
class DistanceImpl {
 public:
  DistanceImpl() = default;

  DistanceImpl(DistanceFunc func, std::string quantized_query, size_t dim)
      : func_(std::move(func)),
        query_storage_(std::move(quantized_query)),
        dim_(dim) {}

  //! Whether the handle is ready to compute distances.
  bool valid() const {
    return static_cast<bool>(func_);
  }

  //! Compute the distance between the stored query and `candidate`.
  float operator()(const void *candidate) const {
    float d = 0.0f;
    func_(candidate, query_storage_.data(), dim_, &d);
    return d;
  }

 private:
  DistanceFunc func_{};
  std::string query_storage_{};
  size_t dim_{0};
};

}  // namespace turbo
}  // namespace zvec
