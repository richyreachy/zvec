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

#include "avx/float32/inner_product.h"
#include "avx/float32/common.h"

#if defined(__AVX__)
#include <immintrin.h>
#include <cstdint>
#endif

namespace zvec::turbo::avx {

// Compute inner product distance between a single quantized FP32
// vector pair.
void inner_product_fp32_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
#if defined(__AVX__)
  const float *lhs = reinterpret_cast<const float *>(a);
  const float *rhs = reinterpret_cast<const float *>(b);

  const float *last = lhs + dim;
  const float *last_aligned = lhs + ((dim >> 4) << 4);

  __m256 ymm_sum_0 = _mm256_setzero_ps();
  __m256 ymm_sum_1 = _mm256_setzero_ps();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_lhs_0 = _mm256_load_ps(lhs + 0);
      __m256 ymm_lhs_1 = _mm256_load_ps(lhs + 8);
      __m256 ymm_rhs_0 = _mm256_load_ps(rhs + 0);
      __m256 ymm_rhs_1 = _mm256_load_ps(rhs + 8);
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      ymm_sum_0 =
          _mm256_fmadd_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs), ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_lhs_0 = _mm256_loadu_ps(lhs + 0);
      __m256 ymm_lhs_1 = _mm256_loadu_ps(lhs + 8);
      __m256 ymm_rhs_0 = _mm256_loadu_ps(rhs + 0);
      __m256 ymm_rhs_1 = _mm256_loadu_ps(rhs + 8);
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      ymm_sum_0 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs),
                                  ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  }
  float result = HorizontalAdd_FP32_V256(_mm256_add_ps(ymm_sum_0, ymm_sum_1));

  switch (last - lhs) {
    case 7:
      FMA_FP32_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      FMA_FP32_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      FMA_FP32_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      FMA_FP32_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      FMA_FP32_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_FP32_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_FP32_GENERAL(lhs[0], rhs[0], result)
  }
  *distance = -1 * result;
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif  // __AVX__
}

// Batch version of inner_product_fp32_distance.
void inner_product_fp32_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances) {
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
}

}  // namespace zvec::turbo::avx