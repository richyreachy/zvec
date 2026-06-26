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

#include <cstddef>

#if defined(__AVX512FP16__)
#include <immintrin.h>
#include <zvec/ailego/utility/float_helper.h>
#include "avx512_fp16/half_float/inner_product.h"
#include "avx512_fp16/half_float/inner_product_common.h"

using namespace zvec::ailego;

using namespace zvec::turbo::avx512_fp16::internal;

#endif

namespace zvec::turbo::avx512_fp16 {

// Compute squared Euclidean distance between a single quantized FP16
// vector pair.
void inner_product_fp16_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
#if defined(__AVX512FP16__)
  const Float16 *lhs = reinterpret_cast<const Float16 *>(a);
  const Float16 *rhs = reinterpret_cast<const Float16 *>(b);

  const Float16 *last = lhs + dim;
  const Float16 *last_aligned = lhs + ((dim >> 6) << 6);

  __m512h zmm_sum_0 = _mm512_setzero_ph();
  __m512h zmm_sum_1 = _mm512_setzero_ph();

  if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      FMA_FP16_AVX512FP16(_mm512_load_ph(lhs + 0), _mm512_load_ph(rhs + 0),
                          zmm_sum_0)

      FMA_FP16_AVX512FP16(_mm512_load_ph(lhs + 32), _mm512_load_ph(rhs + 32),
                          zmm_sum_1)
    }

    if (last >= last_aligned + 32) {
      FMA_FP16_AVX512FP16(_mm512_load_ph(lhs), _mm512_load_ph(rhs), zmm_sum_0)
      lhs += 32;
      rhs += 32;
    }
  } else {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      FMA_FP16_AVX512FP16(_mm512_loadu_ph(lhs + 0), _mm512_loadu_ph(rhs + 0),
                          zmm_sum_0)

      FMA_FP16_AVX512FP16(_mm512_loadu_ph(lhs + 32), _mm512_loadu_ph(rhs + 32),
                          zmm_sum_1)
    }

    if (last >= last_aligned + 32) {
      FMA_FP16_AVX512FP16(_mm512_loadu_ph(lhs), _mm512_loadu_ph(rhs), zmm_sum_0)
      lhs += 32;
      rhs += 32;
    }
  }

  zmm_sum_0 = _mm512_add_ph(zmm_sum_0, zmm_sum_1);

  if (lhs != last) {
    __mmask32 mask = (__mmask32)((1 << (last - lhs)) - 1);
    __m512i zmm_undefined = _mm512_undefined_epi32();
    zmm_sum_0 = _mm512_mask3_fmadd_ph(
        _mm512_castsi512_ph(_mm512_mask_loadu_epi16(zmm_undefined, mask, lhs)),
        _mm512_castsi512_ph(_mm512_mask_loadu_epi16(zmm_undefined, mask, rhs)),
        zmm_sum_0, mask);
  }

  *distance = -1 * HorizontalAdd_FP16_V512(zmm_sum_0);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

// Batch version of inner_product_fp16_distance.
void inner_product_fp16_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances) {
#if defined(__AVX512FP16__)
  for (size_t i = 0; i < n; ++i) {
    inner_product_fp16_distance(vectors[i], query, dim, &distances[i]);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif  // __AVX512FP16__
}

}  // namespace zvec::turbo::avx512_fp16