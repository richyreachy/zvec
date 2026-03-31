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

#include "scalar/record_quantized_int8/inner_product.h"
#include <cstdint>
#include "scalar/record_quantized_int8/common.h"

namespace zvec::turbo::scalar {

// Compute squared Euclidean distance between a single quantized int8
// vector pair.
void inner_product_int8_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
  const size_t original_dim = dim - 20;

  if (original_dim <= 0) {
    return;
  }

  internal::inner_product_int8_scalar(a, b, original_dim, distance);

  const float *a_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const uint8_t *>(a) + original_dim);
  const float *b_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const uint8_t *>(b) + original_dim);

  float qa = a_tail[0];
  float qb = a_tail[1];
  float qs = a_tail[2];

  float ma = b_tail[0];
  float mb = b_tail[1];
  float ms = b_tail[2];

  *distance = -(ma * qa * *distance + mb * qa * qs + qb * ma * ms +
                original_dim * qb * mb);
}

// Batch version of inner_product_int8_distance.
void inner_product_int8_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances) {
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
}

}  // namespace zvec::turbo::scalar