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

namespace zvec::turbo::scalar::internal {

static __attribute__((always_inline)) void inner_product_int8_scalar(
    const void *a, const void *b, size_t dim, float *distance) {
  const int8_t *m = reinterpret_cast<const int8_t *>(a);
  const int8_t *q = reinterpret_cast<const int8_t *>(b);

  float sum = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    sum += static_cast<float>(m[i] * q[i]);
  }

  *distance = -sum;
}

}  // namespace zvec::turbo::scalar::internal
