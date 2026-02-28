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
#define ACCUM_FP32_STEP_NEON FMA_FP32_NEON

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
#define NEGZEROS_FP32_AVX512 _mm512_set1_ps(-0.0f)
#endif  // __AVX512F__

//! Reverse sign of value (SSE)
#define NEGATE_FP32_SSE(v, ...) _mm_xor_ps(v, NEGZEROS_FP32_SSE)

//! Reverse sign of value (AVX)
#define NEGATE_FP32_AVX(v, ...) _mm256_xor_ps(v, NEGZEROS_FP32_AVX)

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_FP32_GENERAL(m, q, sum) sum += (m * q);

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_FP32_SSE(xmm_m, xmm_q, xmm_sum) \
  xmm_sum = _mm_fmadd_ps(xmm_m, xmm_q, xmm_sum);

//! Calculate Fused-Multiply-Add (AVX)
#define FMA_FP32_AVX(ymm_m, ymm_q, ymm_sum) \
  ymm_sum = _mm256_fmadd_ps(ymm_m, ymm_q, ymm_sum);

//! Calculate Fused-Multiply-Add (NEON)
#define FMA_FP32_NEON(v_m, v_q, v_sum) v_sum = vfmaq_f32(v_sum, v_m, v_q);

#if defined(__AVX__)
//! Inner Product
float InnerProductAVX(const float *lhs, const float *rhs,
                                    size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 4) << 4);

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
  return result;
}

//! InnerProduct
float InnerProductAVX_2X1(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_2X1_AVX(lhs, rhs, size, &score, )

  return score;
}

float InnerProductAVX_2X2(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_2X2_AVX(lhs, rhs, size, &score, )
  
  return score;
}
float InnerProductAVX_4X1(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_4X1_AVX(lhs, rhs, size, &score, )
  
  return score;
}
float InnerProductAVX_4X2(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_4X2_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_4X4(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_4X4_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_8X1(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_8X1_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_8X2(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_8X2_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_8X4(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_8X4_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_8X8(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_8X8_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_16X1(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_16X1_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_16X2(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_16X2_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_16X4(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_16X4_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_16X8(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_16X8_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_16X16(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_16X16_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_32X1(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_32X1_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_32X2(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_32X2_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_32X4(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_32X4_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_32X8(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_32X8_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_32X16(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_32X16_AVX(lhs, rhs, size, &score, )
  
  return score;
}

float InnerProductAVX_32X32(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_32X32_AVX(lhs, rhs, size, &score, )
  
  return score;
}

//! MinusInnerProduct
float MinusInnerProductAVX_2X1(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_2X1_AVX(lhs, rhs, size, &score, NEGATE_FP32_SSE)

  return score;
}

float MinusInnerProductAVX_2X2(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_2X2_AVX(lhs, rhs, size, &score, NEGATE_FP32_SSE)
  
  return score;
}
float MinusInnerProductAVX_4X1(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_4X1_AVX(lhs, rhs, size, &score, NEGATE_FP32_SSE)
  
  return score;
}
float MinusInnerProductAVX_4X2(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_4X2_AVX(lhs, rhs, size, &score, NEGATE_FP32_SSE)
  
  return score;
}

float MinusInnerProductAVX_4X4(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_4X4_AVX(lhs, rhs, size, &score, NEGATE_FP32_SSE)
  
  return score;
}

float MinusInnerProductAVX_8X1(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_8X1_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_8X2(const float *lhs, const float *rhs, size_t size) {
  float score;
  ACCUM_FP32_8X2_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_8X4(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_8X4_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_8X8(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_8X8_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_16X1(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_16X1_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_16X2(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_16X2_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_16X4(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_16X4_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_16X8(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_16X8_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_16X16(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_16X16_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_32X1(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_32X1_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_32X2(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_32X2_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_32X4(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_32X4_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_32X8(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_32X8_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_32X16(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_32X16_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

float MinusInnerProductAVX_32X32(const float *lhs, const float *rhs, size_t size){
  float score;
  ACCUM_FP32_32X32_AVX(lhs, rhs, size, &score, NEGATE_FP32_AVX)
  
  return score;
}

#endif  // __AVX__

}  // namespace ailego
}  // namespace zvec
