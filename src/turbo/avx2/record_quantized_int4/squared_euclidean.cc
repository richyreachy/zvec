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

#include "avx2/record_quantized_int4/squared_euclidean.h"
#include "avx2/record_quantized_int4/inner_product_common.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace zvec::turbo::avx2 {

void squared_euclidean_int4_distance(const void *a, const void *b, size_t dim,
                                     float *distance) {
#if defined(__AVX2__)
  const int d = dim - 32;
  const size_t original_dim = d >> 1;

  if (original_dim <= 0) {
    return;
  }

  internal::inner_product_int4_avx2(a, b, original_dim, distance);

  const float *a_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const uint8_t *>(a) + original_dim);
  const float *b_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const uint8_t *>(b) + original_dim);

  float qa = a_tail[0];
  float qb = a_tail[1];
  float qs = a_tail[2];
  float qs2 = a_tail[3];

  const float sum = qa * qs;
  const float sum2 = qa * qa * qs2;

  float ma = b_tail[0];
  float mb = b_tail[1];
  float ms = b_tail[2];
  float ms2 = b_tail[3];

  *distance = ma * ma * ms2 + sum2 - 2 * ma * qa * *distance +
              (mb - qb) * (mb - qb) * d + 2 * (mb - qb) * (ms * ma - sum);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif  // __AVX2__
}

void squared_euclidean_int4_batch_distance(const void *const *vectors,
                                           const void *query, size_t n,
                                           size_t dim, float *distances) {
#if defined(__AVX2__)
  const int d = dim - 32;
  const size_t original_dim = d >> 1;

  if (original_dim <= 0) {
    return;
  }

  internal::inner_product_int4_batch_avx2(vectors, query, n, original_dim,
                                          distances);

  const float *q_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(query) + original_dim);

  float qa = q_tail[0];
  float qb = q_tail[1];
  float qs = q_tail[2];
  float qs2 = q_tail[3];

  const float sum = qa * qs;
  const float sum2 = qa * qa * qs2;

  for (int i = 0; i < n; ++i) {
    const float *m_tail = reinterpret_cast<const float *>(
        reinterpret_cast<const int8_t *>(vectors[i]) + original_dim);

    float ma = m_tail[0];
    float mb = m_tail[1];
    float ms = m_tail[2];
    float ms2 = m_tail[3];

    float &result = distances[i];
    result = ma * ma * ms2 + sum2 - 2 * ma * qa * result +
             (mb - qb) * (mb - qb) * d + 2 * (mb - qb) * (ms * ma - sum);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif  //__AVX2__
}

}  // namespace zvec::turbo::avx2