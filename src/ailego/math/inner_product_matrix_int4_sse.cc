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
#include "distance_matrix_accum_int4.i"
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_INT4_STEP_SSE FMA_INT4_SSE

#if defined(__SSE4_1__)
//! Four-bits Convert Table
static const AILEGO_ALIGNED(32) int8_t Int4ConvertTable[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};
#endif  // __SSE4_1__

#if defined(__SSE4_1__)
static const __m128 NEGZEROS_FP32_SSE = _mm_set1_ps(-0.0f);
static const __m128i MASK_INT4_SSE = _mm_set1_epi32(0x0f0f0f0f);
static const __m128i ONES_INT16_SSE = _mm_set1_epi32(0x00010001);
static const __m128i INT4_LOOKUP_SSE =
    _mm_load_si128((const __m128i *)Int4ConvertTable);
#endif  // __SSE4_1__

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_INT4_GENERAL(m, q, sum)                               \
  sum += Int4MulTable[(((m) << 4) & 0xf0) | (((q) >> 0) & 0xf)] + \
         Int4MulTable[(((m) >> 0) & 0xf0) | (((q) >> 4) & 0xf)];

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_INT4_SSE(xmm_m, xmm_q, xmm_sum)                                    \
  {                                                                            \
    __m128i xmm_lhs = _mm_shuffle_epi8(INT4_LOOKUP_SSE,                        \
                                       _mm_and_si128((xmm_m), MASK_INT4_SSE)); \
    __m128i xmm_rhs = _mm_shuffle_epi8(INT4_LOOKUP_SSE,                        \
                                       _mm_and_si128((xmm_q), MASK_INT4_SSE)); \
    xmm_sum = _mm_add_epi32(                                                   \
        _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_rhs),                \
                                         _mm_sign_epi8(xmm_lhs, xmm_rhs)),     \
                       ONES_INT16_SSE),                                        \
        xmm_sum);                                                              \
    xmm_lhs = _mm_shuffle_epi8(                                                \
        INT4_LOOKUP_SSE,                                                       \
        _mm_and_si128(_mm_srli_epi32((xmm_m), 4), MASK_INT4_SSE));             \
    xmm_rhs = _mm_shuffle_epi8(                                                \
        INT4_LOOKUP_SSE,                                                       \
        _mm_and_si128(_mm_srli_epi32((xmm_q), 4), MASK_INT4_SSE));             \
    xmm_sum = _mm_add_epi32(                                                   \
        _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_rhs),                \
                                         _mm_sign_epi8(xmm_lhs, xmm_rhs)),     \
                       ONES_INT16_SSE),                                        \
        xmm_sum);                                                              \
  }

//! Compute the distance between matrix and query
#define FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)                       \
  {                                                                        \
    __m128i xmm_lhs_0 = _mm_shuffle_epi8(                                  \
        INT4_LOOKUP_SSE, _mm_and_si128((xmm_lhs), MASK_INT4_SSE));         \
    __m128i xmm_rhs_0 = _mm_shuffle_epi8(                                  \
        INT4_LOOKUP_SSE, _mm_and_si128((xmm_rhs), MASK_INT4_SSE));         \
    __m128i xmm_lhs_1 = _mm_shuffle_epi8(                                  \
        INT4_LOOKUP_SSE,                                                   \
        _mm_and_si128(_mm_srli_epi32((xmm_lhs), 4), MASK_INT4_SSE));       \
    __m128i xmm_rhs_1 = _mm_shuffle_epi8(                                  \
        INT4_LOOKUP_SSE,                                                   \
        _mm_and_si128(_mm_srli_epi32((xmm_rhs), 4), MASK_INT4_SSE));       \
    xmm_lhs_0 = _mm_sign_epi8(xmm_lhs_0, xmm_rhs_0);                       \
    xmm_lhs_1 = _mm_sign_epi8(xmm_lhs_1, xmm_rhs_1);                       \
    xmm_rhs_0 = _mm_abs_epi8(xmm_rhs_0);                                   \
    xmm_rhs_1 = _mm_abs_epi8(xmm_rhs_1);                                   \
    xmm_lhs_0 = _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs_0, xmm_lhs_0),    \
                               ONES_INT16_SSE);                            \
    xmm_lhs_1 = _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs_1, xmm_lhs_1),    \
                               ONES_INT16_SSE);                            \
    xmm_sum = _mm_add_epi32(_mm_add_epi32(xmm_lhs_0, xmm_lhs_1), xmm_sum); \
  }

//! Reverse sign of value (SSE)
#define NEGATE_FP32_SSE(v, ...) \
  _mm_xor_ps(_mm_cvtepi32_ps(v), NEGZEROS_FP32_SSE)

#if defined(__SSE4_1__)
//! Inner Product
static inline float InnerProductSSE(const uint8_t *lhs, const uint8_t *rhs,
                                    size_t size) {
  const uint8_t *last = lhs + size;
  const uint8_t *last_aligned = lhs + ((size >> 4) << 4);
  __m128i xmm_sum = _mm_setzero_si128();

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)(lhs));
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)(rhs));
      FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)(lhs));
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)(rhs));
      FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)
    }
  }
  float result = static_cast<float>(HorizontalAdd_INT32_V128(xmm_sum));

  switch (last - lhs) {
    case 15:
      FMA_INT4_GENERAL(lhs[14], rhs[14], result)
      /* FALLTHRU */
    case 14:
      FMA_INT4_GENERAL(lhs[13], rhs[13], result)
      /* FALLTHRU */
    case 13:
      FMA_INT4_GENERAL(lhs[12], rhs[12], result)
      /* FALLTHRU */
    case 12:
      FMA_INT4_GENERAL(lhs[11], rhs[11], result)
      /* FALLTHRU */
    case 11:
      FMA_INT4_GENERAL(lhs[10], rhs[10], result)
      /* FALLTHRU */
    case 10:
      FMA_INT4_GENERAL(lhs[9], rhs[9], result)
      /* FALLTHRU */
    case 9:
      FMA_INT4_GENERAL(lhs[8], rhs[8], result)
      /* FALLTHRU */
    case 8:
      FMA_INT4_GENERAL(lhs[7], rhs[7], result)
      /* FALLTHRU */
    case 7:
      FMA_INT4_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      FMA_INT4_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      FMA_INT4_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      FMA_INT4_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      FMA_INT4_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_INT4_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_INT4_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}
#endif  // __SSE4_1__

#if defined(__SSE4_1__)

float InnerProductSSE_2X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_2X1_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_2X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_2X2_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_4X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_4X1_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_4X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_4X2_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_4X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_4X4_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_8X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_8X1_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_8X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_8X2_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_8X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_8X4_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_8X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_8X8_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_16X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_16X1_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_16X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_16X2_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_16X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_16X4_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_16X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_16X8_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_16X16(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_16X16_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_32X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_32X1_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_32X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_32X2_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_32X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_32X4_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_32X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_32X8_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_32X16(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_32X16_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float InnerProductSSE_32X32(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_32X32_SSE(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

float MinusInnerProductSSE(const uint8_t *lhs, const uint8_t *rhs, size_t size){
  return -InnerProductSSE(lhs, rhs, size);
}

void MinusInnerProductSSE_2X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_2X1_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_2X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_2X2_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_4X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_4X1_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_4X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_4X2_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_4X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_4X4_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_8X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_8X1_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_8X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_8X2_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_8X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_8X4_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_8X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_8X8_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_16X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_16X1_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_16X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_16X2_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_16X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_16X4_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_16X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_16X8_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_16X16(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_16X16_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_32X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_32X1_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_32X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_32X2_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_32X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_32X4_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_32X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_32X8_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_32X16(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out){
  ACCUM_INT4_32X16_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductSSE_32X32(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out)
{
  ACCUM_INT4_32X32_SSE(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

#endif  // __SSE4_1__

}  // namespace ailego
}  // namespace zvec