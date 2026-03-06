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

#include <ailego/internal/cpu_features.h>
#include "distance_matrix_accum_fp32.i"
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_FP32_STEP_SSE FMA_FP32_SSE
#define ACCUM_FP32_STEP_AVX FMA_FP32_AVX
#define ACCUM_FP32_STEP_AVX512 FMA_FP32_AVX512

#if defined(__AVX512F__) && !defined(__AVX512DQ__)
#define _mm512_xor_ps(a, b) \
  _mm512_castsi512_ps(      \
      _mm512_xor_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b)))
#endif  // __AVX512DQ__

#if defined(__SSE__)
#define NEGZEROS_FP32_SSE _mm_set1_ps(-0.0f)
#endif  // __SSE__

#if defined(__AVX__)
#define NEGZEROS_FP32_AVX _mm256_set1_ps(-0.0f)
#endif  // __AVX__

#if defined(__AVX512F__)
#define NEGZEROS_FP32_AVX512 _mm512_set1_ps(-0.0f)
#endif  // __AVX512F__

//! Reverse sign of value (SSE)
#define NEGATE_FP32_SSE(v, ...) _mm_xor_ps(v, NEGZEROS_FP32_SSE)

//! Reverse sign of value (AVX)
#define NEGATE_FP32_AVX(v, ...) _mm256_xor_ps(v, NEGZEROS_FP32_AVX)

//! Reverse sign of value (AVX512)
#define NEGATE_FP32_AVX512(v, ...) _mm512_xor_ps(v, NEGZEROS_FP32_AVX512)

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_FP32_GENERAL(m, q, sum) sum += (m * q);

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_FP32_SSE(xmm_m, xmm_q, xmm_sum) \
  xmm_sum = _mm_fmadd_ps(xmm_m, xmm_q, xmm_sum);

//! Calculate Fused-Multiply-Add (AVX)
#define FMA_FP32_AVX(ymm_m, ymm_q, ymm_sum) \
  ymm_sum = _mm256_fmadd_ps(ymm_m, ymm_q, ymm_sum);

//! Calculate Fused-Multiply-Add (AVX512)
#define FMA_FP32_AVX512(zmm_m, zmm_q, zmm_sum) \
  zmm_sum = _mm512_fmadd_ps(zmm_m, zmm_q, zmm_sum);

#if defined(__AVX512F__)
//! Inner Product
float InnerProductAVX512(const float *lhs, const float *rhs, size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 5) << 5);

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
  return HorizontalAdd_FP32_V512(zmm_sum_0);
}

#endif

}  // namespace ailego
}  // namespace zvec
