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

#include "distance_matrix_accum_int8.i"
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_INT8_STEP_SSE FMA_INT8_SSE

#if defined(__SSE__)
static const __m128 NEGZEROS_FP32_SSE = _mm_set1_ps(-0.0f);
#endif  // __SSE__

#if defined(__SSE4_1__)
static const __m128i ONES_INT16_SSE = _mm_set1_epi32(0x00010001);
#endif  // __SSE4_1__

//! Reverse sign of value (SSE)
#define NEGATE_FP32_SSE(v, ...) \
  _mm_xor_ps(_mm_cvtepi32_ps(v), NEGZEROS_FP32_SSE)

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_INT8_GENERAL(m, q, sum) sum += static_cast<float>(m * q);

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_INT8_SSE(xmm_m, xmm_q, xmm_sum)                                    \
  xmm_sum = _mm_add_epi32(                                                     \
      _mm_madd_epi16(                                                          \
          _mm_maddubs_epi16(_mm_abs_epi8(xmm_q), _mm_sign_epi8(xmm_m, xmm_q)), \
          ONES_INT16_SSE),                                                     \
      xmm_sum);

#if defined(__SSE4_1__)
//! Inner Product
static inline float InnerProductSSE(const int8_t *lhs, const int8_t *rhs,
                                    size_t size) {
  const int8_t *last = lhs + size;
  const int8_t *last_aligned = lhs + ((size >> 5) << 5);

  __m128i xmm_sum_0 = _mm_setzero_si128();
  __m128i xmm_sum_1 = _mm_setzero_si128();

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m128i xmm_lhs_0 = _mm_load_si128((const __m128i *)(lhs + 0));
      __m128i xmm_lhs_1 = _mm_load_si128((const __m128i *)(lhs + 16));
      __m128i xmm_rhs_0 = _mm_load_si128((const __m128i *)(rhs + 0));
      __m128i xmm_rhs_1 = _mm_load_si128((const __m128i *)(rhs + 16));

      xmm_lhs_0 = _mm_sign_epi8(xmm_lhs_0, xmm_rhs_0);
      xmm_lhs_1 = _mm_sign_epi8(xmm_lhs_1, xmm_rhs_1);
      xmm_rhs_0 = _mm_abs_epi8(xmm_rhs_0);
      xmm_rhs_1 = _mm_abs_epi8(xmm_rhs_1);
      xmm_sum_0 =
          _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs_0, xmm_lhs_0),
                                       ONES_INT16_SSE),
                        xmm_sum_0);
      xmm_sum_1 =
          _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs_1, xmm_lhs_1),
                                       ONES_INT16_SSE),
                        xmm_sum_1);
    }

    if (last >= last_aligned + 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)rhs);

      xmm_lhs = _mm_sign_epi8(xmm_lhs, xmm_rhs);
      xmm_rhs = _mm_abs_epi8(xmm_rhs);
      xmm_sum_0 = _mm_add_epi32(
          _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs, xmm_lhs), ONES_INT16_SSE),
          xmm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m128i xmm_lhs_0 = _mm_loadu_si128((const __m128i *)(lhs + 0));
      __m128i xmm_lhs_1 = _mm_loadu_si128((const __m128i *)(lhs + 16));
      __m128i xmm_rhs_0 = _mm_loadu_si128((const __m128i *)(rhs + 0));
      __m128i xmm_rhs_1 = _mm_loadu_si128((const __m128i *)(rhs + 16));

      xmm_lhs_0 = _mm_sign_epi8(xmm_lhs_0, xmm_rhs_0);
      xmm_lhs_1 = _mm_sign_epi8(xmm_lhs_1, xmm_rhs_1);
      xmm_rhs_0 = _mm_abs_epi8(xmm_rhs_0);
      xmm_rhs_1 = _mm_abs_epi8(xmm_rhs_1);
      xmm_sum_0 =
          _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs_0, xmm_lhs_0),
                                       ONES_INT16_SSE),
                        xmm_sum_0);
      xmm_sum_1 =
          _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs_1, xmm_lhs_1),
                                       ONES_INT16_SSE),
                        xmm_sum_1);
    }

    if (last >= last_aligned + 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);

      xmm_lhs = _mm_sign_epi8(xmm_lhs, xmm_rhs);
      xmm_rhs = _mm_abs_epi8(xmm_rhs);
      xmm_sum_0 = _mm_add_epi32(
          _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs, xmm_lhs), ONES_INT16_SSE),
          xmm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  }
  float result = static_cast<float>(
      HorizontalAdd_INT32_V128(_mm_add_epi32(xmm_sum_0, xmm_sum_1)));

  switch (last - lhs) {
    case 15:
      FMA_INT8_GENERAL(lhs[14], rhs[14], result)
      /* FALLTHRU */
    case 14:
      FMA_INT8_GENERAL(lhs[13], rhs[13], result)
      /* FALLTHRU */
    case 13:
      FMA_INT8_GENERAL(lhs[12], rhs[12], result)
      /* FALLTHRU */
    case 12:
      FMA_INT8_GENERAL(lhs[11], rhs[11], result)
      /* FALLTHRU */
    case 11:
      FMA_INT8_GENERAL(lhs[10], rhs[10], result)
      /* FALLTHRU */
    case 10:
      FMA_INT8_GENERAL(lhs[9], rhs[9], result)
      /* FALLTHRU */
    case 9:
      FMA_INT8_GENERAL(lhs[8], rhs[8], result)
      /* FALLTHRU */
    case 8:
      FMA_INT8_GENERAL(lhs[7], rhs[7], result)
      /* FALLTHRU */
    case 7:
      FMA_INT8_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      FMA_INT8_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      FMA_INT8_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      FMA_INT8_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      FMA_INT8_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_INT8_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_INT8_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}
#endif  // __SSE4_1__

#if defined(__SSE4_1__)

void InnerProductSSE_2X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_2X1_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_2X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_2X2_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_4X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_4X1_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_4X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_4X2_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_4X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_4X4_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_8X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X1_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_8X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X2_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_8X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X4_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_8X8(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X8_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_16X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X1_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_16X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X2_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_16X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X4_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_16X8(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X8_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_16X16(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X16_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_32X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X1_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_32X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X2_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_32X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X4_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_32X8(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X8_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_32X16(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X16_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductSSE_32X32(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X32_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float MinusInnerProductSSE(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  return -InnerProductSSE(lhs, rhs, size);
}

void MinusInnerProductSSE_2X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_2X1_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_2X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_2X2_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_4X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_4X1_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_4X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_4X2_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_4X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_4X4_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_8X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X1_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_8X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X2_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_8X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X4_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_8X8(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X8_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_16X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X1_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_16X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X2_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_16X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X4_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_16X8(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X8_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_16X16(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X16_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_32X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X1_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_32X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X2_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_32X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X4_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_32X8(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X8_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_32X16(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X16_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_32X32(const int8_t *lhs, const int8_t *rhs, size_t size, float *out)
{
  ACCUM_INT8_32X32_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

#endif  // __SSE4_1__

}  // namespace ailego
}  // namespace zvec