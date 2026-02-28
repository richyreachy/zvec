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

#include "distance_matrix_accum_int4.i"
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_INT4_STEP_SSE SSD_INT4_SSE
#define ACCUM_INT4_STEP_AVX SSD_INT4_AVX

#if defined(__SSE4_1__)
static const __m128i MASK_INT4_SSE = _mm_set1_epi32(0xf0f0f0f0);
static const __m128i ONES_INT16_SSE = _mm_set1_epi32(0x00010001);
#endif  // __SSE4_1__

#if defined(__AVX2__)
static const __m256i MASK_INT4_AVX = _mm256_set1_epi32(0xf0f0f0f0);
static const __m256i ONES_INT16_AVX = _mm256_set1_epi32(0x00010001);
#endif  // __AVX2__

//! Calculate sum of squared difference (GENERAL)
#define SSD_INT4_GENERAL(m, q, sum)                                       \
  sum += Int4SquaredDiffTable[(((m) << 4) & 0xf0) | (((q) >> 0) & 0xf)] + \
         Int4SquaredDiffTable[(((m) >> 0) & 0xf0) | (((q) >> 4) & 0xf)];

//! Calculate sum of squared difference (SSE)
#define SSD_INT4_SSE(xmm_m, xmm_q, xmm_sum)                                  \
  {                                                                          \
    __m128i xmm_lhs =                                                        \
        _mm_and_si128(_mm_slli_epi32((xmm_m), 4), MASK_INT4_SSE);            \
    __m128i xmm_rhs =                                                        \
        _mm_and_si128(_mm_slli_epi32((xmm_q), 4), MASK_INT4_SSE);            \
    xmm_lhs = _mm_srli_epi32(_mm_sub_epi8(_mm_max_epi8(xmm_lhs, xmm_rhs),    \
                                          _mm_min_epi8(xmm_lhs, xmm_rhs)),   \
                             4);                                             \
    xmm_sum = _mm_add_epi32(                                                 \
        _mm_madd_epi16(_mm_maddubs_epi16(xmm_lhs, xmm_lhs), ONES_INT16_SSE), \
        xmm_sum);                                                            \
    xmm_lhs = _mm_and_si128((xmm_m), MASK_INT4_SSE);                         \
    xmm_rhs = _mm_and_si128((xmm_q), MASK_INT4_SSE);                         \
    xmm_lhs = _mm_srli_epi32(_mm_sub_epi8(_mm_max_epi8(xmm_lhs, xmm_rhs),    \
                                          _mm_min_epi8(xmm_lhs, xmm_rhs)),   \
                             4);                                             \
    xmm_sum = _mm_add_epi32(                                                 \
        _mm_madd_epi16(_mm_maddubs_epi16(xmm_lhs, xmm_lhs), ONES_INT16_SSE), \
        xmm_sum);                                                            \
  }

//! Calculate sum of squared difference (AVX)
#define SSD_INT4_AVX(ymm_m, ymm_q, ymm_sum)                                   \
  {                                                                           \
    __m256i ymm_lhs =                                                         \
        _mm256_and_si256(_mm256_slli_epi32((ymm_m), 4), MASK_INT4_AVX);       \
    __m256i ymm_rhs =                                                         \
        _mm256_and_si256(_mm256_slli_epi32((ymm_q), 4), MASK_INT4_AVX);       \
    ymm_lhs =                                                                 \
        _mm256_srli_epi32(_mm256_sub_epi8(_mm256_max_epi8(ymm_lhs, ymm_rhs),  \
                                          _mm256_min_epi8(ymm_lhs, ymm_rhs)), \
                          4);                                                 \
    ymm_sum = _mm256_add_epi32(                                               \
        _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_lhs, ymm_lhs),             \
                          ONES_INT16_AVX),                                    \
        ymm_sum);                                                             \
    ymm_lhs = _mm256_and_si256((ymm_m), MASK_INT4_AVX);                       \
    ymm_rhs = _mm256_and_si256((ymm_q), MASK_INT4_AVX);                       \
    ymm_lhs =                                                                 \
        _mm256_srli_epi32(_mm256_sub_epi8(_mm256_max_epi8(ymm_lhs, ymm_rhs),  \
                                          _mm256_min_epi8(ymm_lhs, ymm_rhs)), \
                          4);                                                 \
    ymm_sum = _mm256_add_epi32(                                               \
        _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_lhs, ymm_lhs),             \
                          ONES_INT16_AVX),                                    \
        ymm_sum);                                                             \
  }

//! Compute the distance between matrix and query
#define SSD_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)                       \
  {                                                                        \
    __m128i xmm_lhs_0 =                                                    \
        _mm_and_si128(_mm_slli_epi32((xmm_lhs), 4), MASK_INT4_SSE);        \
    __m128i xmm_rhs_0 =                                                    \
        _mm_and_si128(_mm_slli_epi32((xmm_rhs), 4), MASK_INT4_SSE);        \
    __m128i xmm_lhs_1 = _mm_and_si128((xmm_lhs), MASK_INT4_SSE);           \
    __m128i xmm_rhs_1 = _mm_and_si128((xmm_rhs), MASK_INT4_SSE);           \
    xmm_lhs_0 =                                                            \
        _mm_srli_epi32(_mm_sub_epi8(_mm_max_epi8(xmm_lhs_0, xmm_rhs_0),    \
                                    _mm_min_epi8(xmm_lhs_0, xmm_rhs_0)),   \
                       4);                                                 \
    xmm_rhs_0 =                                                            \
        _mm_srli_epi32(_mm_sub_epi8(_mm_max_epi8(xmm_lhs_1, xmm_rhs_1),    \
                                    _mm_min_epi8(xmm_lhs_1, xmm_rhs_1)),   \
                       4);                                                 \
    xmm_lhs_0 = _mm_madd_epi16(_mm_maddubs_epi16(xmm_lhs_0, xmm_lhs_0),    \
                               ONES_INT16_SSE);                            \
    xmm_rhs_0 = _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs_0, xmm_rhs_0),    \
                               ONES_INT16_SSE);                            \
    xmm_sum = _mm_add_epi32(_mm_add_epi32(xmm_lhs_0, xmm_rhs_0), xmm_sum); \
  }

//! Compute the distance between matrix and query
#define SSD_INT4_ITER_AVX(ymm_lhs, ymm_rhs, ymm_sum)                          \
  {                                                                           \
    __m256i ymm_lhs_0 =                                                       \
        _mm256_and_si256(_mm256_slli_epi32((ymm_lhs), 4), MASK_INT4_AVX);     \
    __m256i ymm_rhs_0 =                                                       \
        _mm256_and_si256(_mm256_slli_epi32((ymm_rhs), 4), MASK_INT4_AVX);     \
    __m256i ymm_lhs_1 = _mm256_and_si256((ymm_lhs), MASK_INT4_AVX);           \
    __m256i ymm_rhs_1 = _mm256_and_si256((ymm_rhs), MASK_INT4_AVX);           \
    ymm_lhs_0 = _mm256_srli_epi32(                                            \
        _mm256_sub_epi8(_mm256_max_epi8(ymm_lhs_0, ymm_rhs_0),                \
                        _mm256_min_epi8(ymm_lhs_0, ymm_rhs_0)),               \
        4);                                                                   \
    ymm_rhs_0 = _mm256_srli_epi32(                                            \
        _mm256_sub_epi8(_mm256_max_epi8(ymm_lhs_1, ymm_rhs_1),                \
                        _mm256_min_epi8(ymm_lhs_1, ymm_rhs_1)),               \
        4);                                                                   \
    ymm_lhs_0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_lhs_0, ymm_lhs_0), \
                                  ONES_INT16_AVX);                            \
    ymm_rhs_0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_rhs_0), \
                                  ONES_INT16_AVX);                            \
    ymm_sum =                                                                 \
        _mm256_add_epi32(_mm256_add_epi32(ymm_lhs_0, ymm_rhs_0), ymm_sum);    \
  }

//! Compute the square root of value (SSE)
#define SQRT_FP32_SSE(v, ...) _mm_sqrt_ps(_mm_cvtepi32_ps(v))

//! Compute the square root of value (AVX)
#define SQRT_FP32_AVX(v, ...) _mm256_sqrt_ps(_mm256_cvtepi32_ps(v))

//! Compute the square root of value (AVX512)
#define SQRT_FP32_AVX512(v, ...) _mm512_sqrt_ps(_mm512_cvtepi32_ps(v))

#if defined(__AVX2__)
//! Squared Euclidean Distance
static inline float SquaredEuclideanDistanceAVX(const uint8_t *lhs,
                                                const uint8_t *rhs,
                                                size_t size) {
  const uint8_t *last = lhs + size;
  const uint8_t *last_aligned = lhs + ((size >> 5) << 5);

  __m256i ymm_sum = _mm256_setzero_si256();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m256i ymm_lhs = _mm256_load_si256((const __m256i *)(lhs));
      __m256i ymm_rhs = _mm256_load_si256((const __m256i *)(rhs));
      SSD_INT4_ITER_AVX(ymm_lhs, ymm_rhs, ymm_sum)
    }
    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)rhs);
      __m128i xmm_sum = _mm_setzero_si128();
      SSD_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)
      ymm_sum = _mm256_add_epi32(_mm256_set_m128i(_mm_setzero_si128(), xmm_sum),
                                 ymm_sum);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m256i ymm_lhs = _mm256_loadu_si256((const __m256i *)(lhs));
      __m256i ymm_rhs = _mm256_loadu_si256((const __m256i *)(rhs));
      SSD_INT4_ITER_AVX(ymm_lhs, ymm_rhs, ymm_sum)
    }
    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);
      __m128i xmm_sum = _mm_setzero_si128();
      SSD_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)
      ymm_sum = _mm256_add_epi32(_mm256_set_m128i(_mm_setzero_si128(), xmm_sum),
                                 ymm_sum);
      lhs += 16;
      rhs += 16;
    }
  }
  float result = static_cast<float>(HorizontalAdd_INT32_V256(ymm_sum));

  switch (last - lhs) {
    case 15:
      SSD_INT4_GENERAL(lhs[14], rhs[14], result)
      /* FALLTHRU */
    case 14:
      SSD_INT4_GENERAL(lhs[13], rhs[13], result)
      /* FALLTHRU */
    case 13:
      SSD_INT4_GENERAL(lhs[12], rhs[12], result)
      /* FALLTHRU */
    case 12:
      SSD_INT4_GENERAL(lhs[11], rhs[11], result)
      /* FALLTHRU */
    case 11:
      SSD_INT4_GENERAL(lhs[10], rhs[10], result)
      /* FALLTHRU */
    case 10:
      SSD_INT4_GENERAL(lhs[9], rhs[9], result)
      /* FALLTHRU */
    case 9:
      SSD_INT4_GENERAL(lhs[8], rhs[8], result)
      /* FALLTHRU */
    case 8:
      SSD_INT4_GENERAL(lhs[7], rhs[7], result)
      /* FALLTHRU */
    case 7:
      SSD_INT4_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      SSD_INT4_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      SSD_INT4_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      SSD_INT4_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      SSD_INT4_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      SSD_INT4_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      SSD_INT4_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}


//! SquaredEuclideanDistance
float SquaredEuclideanDistanceAVX2_2X1(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_2X1_AVX(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceAVX2_2X2(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_2X1_AVX(lhs, rhs, size, &score, _mm_cvtepi32_ps)
  
  return score;
}
float SquaredEuclideanDistanceAVX2_4X1(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_4X1_AVX(lhs, rhs, size, &score, _mm_cvtepi32_ps)
  
  return score;
}
float SquaredEuclideanDistanceAVX2_4X2(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_4X2_AVX(lhs, rhs, size, &score, _mm_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_4X4(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_4X4_AVX(lhs, rhs, size, &score, _mm_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_8X1(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_8X1_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_8X2(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_8X2_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_8X4(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_8X4_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_8X8(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_8X8_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_16X1(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_16X1_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_16X2(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_16X2_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_16X4(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_16X4_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_16X8(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_16X8_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_16X16(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_16X16_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_32X1(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_32X1_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_32X2(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_32X2_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_32X4(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_32X4_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_32X8(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_32X8_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_32X16(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_32X16_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

float SquaredEuclideanDistanceAVX2_32X32(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_32X32_AVX(lhs, rhs, size, &score, _mm256_cvtepi32_ps)
  
  return score;
}

//! EuclideanDistance
float EuclideanDistanceAVX2_2X1(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_2X1_AVX(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceAVX2_2X2(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_2X2_AVX(lhs, rhs, size, &score, SQRT_FP32_SSE)
  
  return score;
}
float EuclideanDistanceAVX2_4X1(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_4X1_AVX(lhs, rhs, size, &score, SQRT_FP32_SSE)
  
  return score;
}
float EuclideanDistanceAVX2_4X2(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_4X2_AVX(lhs, rhs, size, &score, SQRT_FP32_SSE)
  
  return score;
}

float EuclideanDistanceAVX2_4X4(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_4X4_AVX(lhs, rhs, size, &score, SQRT_FP32_SSE)
  
  return score;
}

float EuclideanDistanceAVX2_8X1(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_8X1_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_8X2(const int8_t *lhs, const int8_t *rhs, size_t size) {
  float score;
  ACCUM_INT4_8X2_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_8X4(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_8X4_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_8X8(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_8X8_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_16X1(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_16X1_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_16X2(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_16X2_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_16X4(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_16X4_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_16X8(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_16X8_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_16X16(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_16X16_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_32X1(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_32X1_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_32X2(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_32X2_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_32X4(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_32X4_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_32X8(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_32X8_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_32X16(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_32X16_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}

float EuclideanDistanceAVX2_32X32(const int8_t *lhs, const int8_t *rhs, size_t size){
  float score;
  ACCUM_INT4_32X32_AVX(lhs, rhs, size, &score, SQRT_FP32_AVX)
  
  return score;
}
#endif  // __AVX2__

}  // namespace ailego
}  // namespace zvec