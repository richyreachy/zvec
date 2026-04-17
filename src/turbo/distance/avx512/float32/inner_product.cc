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

#include "avx512/float32/inner_product.h"
#include "avx512/float32/common.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace zvec::turbo::avx512 {

// Compute squared Euclidean distance between a single quantized FP32
// vector pair.
void inner_product_fp32_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
#if defined(__AVX512F__)
  const float *lhs = reinterpret_cast<const float *>(a);
  const float *rhs = reinterpret_cast<const float *>(b);

  const float *last = lhs + dim;
  const float *last_aligned = lhs + ((dim >> 5) << 5);

  __m512 zmm_sum_0 = _mm512_setzero_ps();
  __m512 zmm_sum_1 = _mm512_setzero_ps();

  if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      FMA_FP32_AVX512(_mm512_load_ps(lhs + 0), _mm512_load_ps(rhs + 0),
                      zmm_sum_0)

      FMA_FP32_AVX512(_mm512_load_ps(lhs + 16), _mm512_load_ps(rhs + 16),
                      zmm_sum_1)
    }

    if (last >= last_aligned + 16) {
      FMA_FP32_AVX512(_mm512_load_ps(lhs), _mm512_load_ps(rhs), zmm_sum_0)
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      FMA_FP32_AVX512(_mm512_loadu_ps(lhs + 0), _mm512_loadu_ps(rhs + 0),
                      zmm_sum_0)

      FMA_FP32_AVX512(_mm512_loadu_ps(lhs + 16), _mm512_loadu_ps(rhs + 16),
                      zmm_sum_1)
    }

    if (last >= last_aligned + 16) {
      FMA_FP32_AVX512(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs), zmm_sum_0)
      lhs += 16;
      rhs += 16;
    }
  }

  zmm_sum_0 = _mm512_add_ps(zmm_sum_0, zmm_sum_1);
  if (lhs != last) {
    __mmask16 mask = (__mmask16)((1 << (last - lhs)) - 1);
    __m512 zmm_undefined = _mm512_undefined_ps();
    zmm_sum_0 = _mm512_mask3_fmadd_ps(
        _mm512_mask_loadu_ps(zmm_undefined, mask, lhs),
        _mm512_mask_loadu_ps(zmm_undefined, mask, rhs), zmm_sum_0, mask);
  }

  *distance = -1 * HorizontalAdd_FP32_V512(zmm_sum_0);

#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif  //__AVX512F__
}

// Batch version of inner_product_fp32_distance.
void inner_product_fp32_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances) {
#if defined(__AVX512F__)
  for (size_t i = 0; i < n; ++i) {
    inner_product_fp32_distance(vectors[i], query, dim, &distances[i]);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif  //__AVX512F__
}

}  // namespace zvec::turbo::avx512