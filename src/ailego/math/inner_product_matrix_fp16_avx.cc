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
#include "distance_matrix_accum_fp16.i"
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_FP32_STEP_SSE FMA_FP32_SSE
#define ACCUM_FP32_STEP_AVX FMA_FP32_AVX
#define ACCUM_FP32_STEP_AVX512 FMA_FP32_AVX512
#define ACCUM_FP32_STEP_NEON FMA_FP32_NEON
#define ACCUM_FP16_STEP_GENERAL FMA_FP16_GENERAL
#define ACCUM_FP16_STEP_NEON FMA_FP16_NEON

#if defined(__AVX512F__) && !defined(__AVX512DQ__)
#define _mm512_xor_ps(a, b) \
  _mm512_castsi512_ps(      \
      _mm512_xor_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b)))
#endif  // __AVX512DQ__

#if defined(__SSE__)
static const __m128 NEGZEROS_FP32_SSE = _mm_set1_ps(-0.0f);
#endif  // __SSE__

#if defined(__AVX__)
static const __m256 NEGZEROS_FP32_AVX = _mm256_set1_ps(-0.0f);
#endif  // __AVX__

//! Reverse sign of value (GENERAL)
#define NEGATE_FP32_GENERAL(v) -(v)

#define NEGATE_FP32_SSE(v, ...) _mm_xor_ps(v, NEGZEROS_FP32_SSE)

//! Reverse sign of value (AVX)
#define NEGATE_FP32_AVX(v, ...) _mm256_xor_ps(v, NEGZEROS_FP32_AVX)

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_FP32_SSE(xmm_m, xmm_q, xmm_sum) \
  xmm_sum = _mm_fmadd_ps(xmm_m, xmm_q, xmm_sum);

//! Calculate Fused-Multiply-Add (AVX)
#define FMA_FP32_AVX(ymm_m, ymm_q, ymm_sum) \
  ymm_sum = _mm256_fmadd_ps(ymm_m, ymm_q, ymm_sum);

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_FP16_GENERAL(m, q, sum) sum += (m * q);

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))

#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

// sparse
#if defined(__AVX__)
const static __m128i SHUFFLE_MASK256[256] = {
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, -127, -127),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 5,
                 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 7,
                 6, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 7,
                 6, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 7,
                 6, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 7, 6, 5, 4, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 9, 8, 5, 4, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 9, 8, 7, 6, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 9, 8, 7, 6, 5,
                 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 9, 8, 7, 6, 5,
                 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 11, 10),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 5, 4,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 7, 6,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 7, 6,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 7, 6,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 7, 6, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 9, 8, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 13, 12),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 5, 4,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 7, 6,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 7, 6,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 7, 6,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 7, 6, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 9, 8, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 9, 8, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 9, 8, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 9, 8, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 11, 10),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6,
                 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 15, 14),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 5, 4,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 7, 6,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 7, 6,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 7, 6,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 7, 6, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 9, 8, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 9, 8, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 9, 8, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 9, 8, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 11, 10),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6,
                 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 13, 12),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6,
                 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 11, 10),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 5,
                 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 5,
                 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7,
                 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7,
                 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7, 6, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7,
                 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7, 6, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7, 6, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9,
                 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9,
                 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9,
                 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9,
                 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
                 4),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2),
    _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
};

constexpr uint32_t MAX_SPARSE_BUFFER_LENGTH = 65536;

float InnerProductSparseInSegmentAVX(uint32_t m_sparse_count,
                                     const uint16_t *m_sparse_index,
                                     const Float16 *m_sparse_value,
                                     uint32_t q_sparse_count,
                                     const uint16_t *q_sparse_index,
                                     const Float16 *q_sparse_value) {
  float sum = 0.0f;

  // handle if the first dim is zero
  bool m_zero = false;
  Float16 m_zero_value{0.0f};
  if (m_sparse_count > 0 && m_sparse_index[0] == 0) {
    m_sparse_count--;
    m_sparse_index++;
    m_zero_value = *m_sparse_value++;
    m_zero = true;
  }

  bool q_zero = false;
  Float16 q_zero_value{0.0f};
  if (q_sparse_count > 0 && q_sparse_index[0] == 0) {
    q_sparse_count--;
    q_sparse_index++;
    q_zero_value = *q_sparse_value++;
    q_zero = true;
  }

  if (m_zero && q_zero) {
    sum = m_zero_value * q_zero_value;
  }

  size_t i1 = 0, i2 = 0;
  size_t end1 = m_sparse_count / 8 * 8;
  size_t end2 = q_sparse_count / 8 * 8;

  uint16_t fixed_buffer_1[MAX_SPARSE_BUFFER_LENGTH];
  uint16_t fixed_buffer_2[MAX_SPARSE_BUFFER_LENGTH];

  Float16 *val_start_1 = reinterpret_cast<Float16 *>(fixed_buffer_1);
  Float16 *val_start_2 = reinterpret_cast<Float16 *>(fixed_buffer_2);

  Float16 *val_1 = val_start_1;
  Float16 *val_2 = val_start_2;

  if (i1 < end1 && i2 < end2) {
    while (m_sparse_index[i1 + 7] < q_sparse_index[i2]) {
      i1 += 8;
      if (i1 >= end1) goto do_scalar;
    }

    while (q_sparse_index[i2 + 7] < m_sparse_index[i1]) {
      i2 += 8;
      if (i2 >= end2) goto do_scalar;
    }

    __m128i mm_index_m =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_sparse_index[i1]));
    __m128i mm_index_q =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(&q_sparse_index[i2]));

    while (true) {
#ifdef DEBUG_PRINT
      std::cout << "index 1: " << std::endl;
      print_data16(&mm_index_m);

      std::cout << "index 2: " << std::endl;
      print_data16(&mm_index_q);
#endif

      __m128i mm_cmp_res =
          _mm_cmpistrm(mm_index_q, mm_index_m,
                       _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);

#ifdef DEBUG_PRINT
      std::cout << "cmp res: " << std::endl;
      print_data16(&mm_cmp_res);
#endif

      int r = _mm_extract_epi32(mm_cmp_res, 0);

      if (r) {
        int r1 = r;

        __m128i v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&m_sparse_value[i1]));
        __m128i vs = _mm_shuffle_epi8(v, SHUFFLE_MASK256[r1]);

        _mm_storeu_si128(reinterpret_cast<__m128i *>(val_1), vs);
        val_1 += _mm_popcnt_u32(r1);

        mm_cmp_res = _mm_cmpistrm(
            mm_index_m, mm_index_q,
            _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);
        r = _mm_extract_epi32(mm_cmp_res, 0);

        r1 = r;

        v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&q_sparse_value[i2]));
        vs = _mm_shuffle_epi8(v, SHUFFLE_MASK256[r1]);

        _mm_storeu_si128(reinterpret_cast<__m128i *>(val_2), vs);
        val_2 += _mm_popcnt_u32(r1);
      }

      const uint16_t id1_max = m_sparse_index[i1 + 7];

      if (id1_max <= q_sparse_index[i2 + 7]) {
        i1 += 8;
        if (i1 >= end1) goto do_scalar;
        mm_index_m = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&m_sparse_index[i1]));
      }

      if (id1_max >= q_sparse_index[i2 + 7]) {
        i2 += 8;
        if (i2 >= end2) goto do_scalar;
        mm_index_q = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&q_sparse_index[i2]));
      }
    }
  }

do_scalar:
  while (i1 < m_sparse_count && i2 < q_sparse_count) {
    if (m_sparse_index[i1] == q_sparse_index[i2]) {
      *val_1++ = m_sparse_value[i1];
      *val_2++ = q_sparse_value[i2];

      ++i1;
      ++i2;
    } else if (m_sparse_index[i1] < q_sparse_index[i2]) {
      ++i1;
    } else {
      ++i2;
    }
  }

  size_t res_num = val_1 - val_start_1;

  size_t res_num8 = res_num / 8 * 8;

  if (res_num8) {
    __m256 sum256 = _mm256_setzero_ps();

    for (size_t k = 0; k < res_num8; k += 8) {
      __m256 ymm_1 =
          _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(val_start_1 + k)));
      __m256 ymm_2 =
          _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(val_start_2 + k)));
      ACCUM_FP32_STEP_AVX(ymm_1, ymm_2, sum256);
    }

    sum += HorizontalAdd_FP32_V256(sum256);
  }

  for (size_t k = res_num8; k < res_num; ++k)
    sum += val_start_1[k] * val_start_2[k];

  return sum;
}

#endif  // __AVX__


#if defined(__AVX__)
float InnerProductAVX(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_1X1_AVX(lhs, rhs, size, out, 0ull, )
}

float InnerProductAVX_2X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_2X1_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_2X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_2X2_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_4X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_4X1_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_4X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_4X2_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_4X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_4X4_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_8X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_8X1_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_8X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_8X2_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_8X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_8X4_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_8X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_8X8_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_16X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X1_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_16X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X2_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_16X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X4_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_16X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X8_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_16X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X16_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_32X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X1_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_32X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X2_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_32X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X4_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_32X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X8_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_32X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X16_AVX(lhs, rhs, size, out, )
}

float InnerProductAVX_32X32(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X32_AVX(lhs, rhs, size, out, )
}

float MinusInnerProductAVX(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_1X1_AVX(lhs, rhs, size, out, 0ull, NEGATE_FP32_GENERAL)
}

void MinusInnerProductAVX_2X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_2X1_AVX(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductAVX_2X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_2X2_AVX(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductAVX_4X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_4X1_AVX(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductAVX_4X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_4X2_AVX(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductAVX_4X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_4X4_AVX(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductAVX_8X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_8X1_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_8X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_8X2_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_8X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_8X4_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_8X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_8X8_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_16X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X1_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_16X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X2_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_16X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X4_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_16X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X8_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_16X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X16_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_32X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X1_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_32X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X2_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_32X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X4_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_32X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X8_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_32X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X16_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX_32X32(const Float16 *lhs, const Float16 *rhs, size_t size, float *out)
{
  ACCUM_FP16_32X32_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

#endif
}  // namespace ailego
}  // namespace zvec