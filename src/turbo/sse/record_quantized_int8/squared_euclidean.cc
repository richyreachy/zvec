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

#include "sse/record_quantized_int8/squared_euclidean.h"
#include "sse/record_quantized_int8/common.h"
#if defined(__SSE__)
#include <immintrin.h>
#endif

namespace zvec::turbo::sse {

void squared_euclidean_int8_distance(const void *a, const void *b, size_t dim,
                                     float *distance) {
#if defined(__SSE__)
  const int original_dim = dim - 20;
  if (original_dim <= 0) {
    return;
  }
  internal::inner_product_int8_sse(a, b, original_dim, distance);

  const float *a_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(a) + original_dim);
  const float *b_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(b) + original_dim);

  float ma = a_tail[0];
  float mb = a_tail[1];
  float ms = a_tail[2];
  float ms2 = a_tail[3];

  float qa = b_tail[0];
  float qb = b_tail[1];
  float qs = b_tail[2];
  float qs2 = b_tail[3];

  const float sum = qa * qs;
  const float sum2 = qa * qa * qs2;

  *distance = ma * ma * ms2 + sum2 - 2 * ma * qa * *distance +
              (mb - qb) * (mb - qb) * original_dim +
              2 * (mb - qb) * (ms * ma - sum);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void squared_euclidean_int8_batch_distance(const void *const *vectors,
                                           const void *query, size_t n,
                                           size_t dim, float *distances) {
#if defined(__SSE__)

#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

}  // namespace zvec::turbo::sse
