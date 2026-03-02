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
#define ACCUM_INT8_STEP_AVX FMA_INT8_AVX

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

#if defined(__AVX512F__)
static const __m512 NEGZEROS_FP32_AVX512 = _mm512_set1_ps(-0.0f);
#endif  // __AVX512F__

#if defined(__SSE4_1__)
static const __m128i ONES_INT16_SSE = _mm_set1_epi32(0x00010001);
#endif  // __SSE4_1__

#if defined(__AVX2__)
static const __m256i ONES_INT16_AVX = _mm256_set1_epi32(0x00010001);
#endif  // __AVX2__

//! Reverse sign of value (SSE)
#define NEGATE_FP32_SSE(v, ...) \
  _mm_xor_ps(_mm_cvtepi32_ps(v), NEGZEROS_FP32_SSE)

//! Reverse sign of value (AVX)
#define NEGATE_FP32_AVX(v, ...) \
  _mm256_xor_ps(_mm256_cvtepi32_ps(v), NEGZEROS_FP32_AVX)

//! Reverse sign of value (AVX512)
#define NEGATE_FP32_AVX512(v, ...) \
  _mm512_xor_ps(_mm512_cvtepi32_ps(v), NEGZEROS_FP32_AVX512)

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_INT8_GENERAL(m, q, sum) sum += static_cast<float>(m * q);

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_INT8_SSE(xmm_m, xmm_q, xmm_sum)                                    \
  xmm_sum = _mm_add_epi32(                                                     \
      _mm_madd_epi16(                                                          \
          _mm_maddubs_epi16(_mm_abs_epi8(xmm_q), _mm_sign_epi8(xmm_m, xmm_q)), \
          ONES_INT16_SSE),                                                     \
      xmm_sum);

//! Calculate Fused-Multiply-Add (AVX)
#define FMA_INT8_AVX(ymm_m, ymm_q, ymm_sum)                                   \
  ymm_sum = _mm256_add_epi32(                                                 \
      _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(ymm_q),          \
                                             _mm256_sign_epi8(ymm_m, ymm_q)), \
                        ONES_INT16_AVX),                                      \
      ymm_sum);

#if defined(__AVX2__)
//! Inner Product
float InnerProductAVX2(const int8_t *lhs, const int8_t *rhs, size_t size) {
  const int8_t *last = lhs + size;
  const int8_t *last_aligned = lhs + ((size >> 6) << 6);
  float result = 0.0;

  __m256i ymm_sum_0 = _mm256_setzero_si256();
  __m256i ymm_sum_1 = _mm256_setzero_si256();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_load_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_load_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_load_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_load_si256((const __m256i *)(rhs + 32));

      ymm_lhs_0 = _mm256_sign_epi8(ymm_lhs_0, ymm_rhs_0);
      ymm_lhs_1 = _mm256_sign_epi8(ymm_lhs_1, ymm_rhs_1);
      ymm_rhs_0 = _mm256_abs_epi8(ymm_rhs_0);
      ymm_rhs_1 = _mm256_abs_epi8(ymm_rhs_1);

      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_lhs_0),
                            ONES_INT16_AVX),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_1, ymm_lhs_1),
                            ONES_INT16_AVX),
          ymm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_load_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_load_si256((const __m256i *)rhs);
      ymm_lhs = _mm256_sign_epi8(ymm_lhs, ymm_rhs);
      ymm_rhs = _mm256_abs_epi8(ymm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs, ymm_lhs),
                            ONES_INT16_AVX),
          ymm_sum_0);
      lhs += 32;
      rhs += 32;
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)rhs);
      xmm_lhs = _mm_sign_epi8(xmm_lhs, xmm_rhs);
      xmm_rhs = _mm_abs_epi8(xmm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(),
                           _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs, xmm_lhs),
                                          ONES_INT16_SSE)),
          ymm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_loadu_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_loadu_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_loadu_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_loadu_si256((const __m256i *)(rhs + 32));

      ymm_lhs_0 = _mm256_sign_epi8(ymm_lhs_0, ymm_rhs_0);
      ymm_lhs_1 = _mm256_sign_epi8(ymm_lhs_1, ymm_rhs_1);
      ymm_rhs_0 = _mm256_abs_epi8(ymm_rhs_0);
      ymm_rhs_1 = _mm256_abs_epi8(ymm_rhs_1);

      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_lhs_0),
                            ONES_INT16_AVX),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_1, ymm_lhs_1),
                            ONES_INT16_AVX),
          ymm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_loadu_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_loadu_si256((const __m256i *)rhs);
      ymm_lhs = _mm256_sign_epi8(ymm_lhs, ymm_rhs);
      ymm_rhs = _mm256_abs_epi8(ymm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs, ymm_lhs),
                            ONES_INT16_AVX),
          ymm_sum_0);
      lhs += 32;
      rhs += 32;
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);
      xmm_lhs = _mm_sign_epi8(xmm_lhs, xmm_rhs);
      xmm_rhs = _mm_abs_epi8(xmm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(),
                           _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs, xmm_lhs),
                                          ONES_INT16_SSE)),
          ymm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  }
  result = static_cast<float>(
      HorizontalAdd_INT32_V256(_mm256_add_epi32(ymm_sum_0, ymm_sum_1)));

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

void InnerProductAVX2_2X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_2X1_AVX(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductAVX2_2X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_2X2_AVX(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductAVX2_4X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_4X1_AVX(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductAVX2_4X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_4X2_AVX(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductAVX2_4X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_4X4_AVX(lhs, rhs, size, out, _mm_cvtepi32_ps)
}

void InnerProductAVX2_8X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X1_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_8X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X2_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_8X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X4_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_8X8(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X8_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_16X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X1_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_16X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X2_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_16X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X4_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_16X8(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X8_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_16X16(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X16_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_32X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X1_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_32X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X2_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_32X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X4_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_32X8(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X8_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_32X16(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X16_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

void InnerProductAVX2_32X32(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X32_AVX(lhs, rhs, size, out, _mm256_cvtepi32_ps)
}

float MinusInnerProductAVX2(const int8_t *lhs, const int8_t *rhs, size_t size){
  return -InnerProductAVX2(lhs, rhs, size);
}

void MinusInnerProductAVX2_2X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_2X1_AVX(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductAVX2_2X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_2X2_AVX(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductAVX2_4X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_4X1_AVX(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductAVX2_4X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_4X2_AVX(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductAVX2_4X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_4X4_AVX(lhs, rhs, size, out, NEGATE_FP32_SSE)
}

void MinusInnerProductAVX2_8X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X1_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_8X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X2_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_8X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X4_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_8X8(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_8X8_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_16X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X1_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_16X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X2_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_16X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X4_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_16X8(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X8_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_16X16(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_16X16_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_32X1(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X1_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_32X2(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X2_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_32X4(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X4_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_32X8(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X8_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_32X16(const int8_t *lhs, const int8_t *rhs, size_t size, float *out){
  ACCUM_INT8_32X16_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}

void MinusInnerProductAVX2_32X32(const int8_t *lhs, const int8_t *rhs, size_t size, float *out)
{
  ACCUM_INT8_32X32_AVX(lhs, rhs, size, out, NEGATE_FP32_AVX)
}
#endif  // __AVX2__


}  // namespace ailego
}  // namespace zvec