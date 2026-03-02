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
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_FP32_STEP_SSE SSD_FP32_SSE
#define ACCUM_FP32_STEP_AVX SSD_FP32_AVX

//! Calculate sum of squared difference (GENERAL)
#define SSD_FP32_GENERAL(m, q, sum) \
  {                                 \
    float x = m - q;                \
    sum += (x * x);                 \
  }

//! Calculate sum of squared difference (SSE)
#define SSD_FP32_SSE(xmm_m, xmm_q, xmm_sum)        \
  {                                                \
    __m128 xmm_d = _mm_sub_ps(xmm_m, xmm_q);       \
    xmm_sum = _mm_fmadd_ps(xmm_d, xmm_d, xmm_sum); \
  }

//! Calculate sum of squared difference (AVX)
#define SSD_FP32_AVX(ymm_m, ymm_q, ymm_sum)           \
  {                                                   \
    __m256 ymm_d = _mm256_sub_ps(ymm_m, ymm_q);       \
    ymm_sum = _mm256_fmadd_ps(ymm_d, ymm_d, ymm_sum); \
  }

#if defined(__AVX__)
float SquaredEuclideanDistanceAVX(const float *lhs, const float *rhs, size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 4) << 4);

  __m256 ymm_sum_0 = _mm256_setzero_ps();
  __m256 ymm_sum_1 = _mm256_setzero_ps();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_d_0 =
          _mm256_sub_ps(_mm256_load_ps(lhs + 0), _mm256_load_ps(rhs + 0));
      __m256 ymm_d_1 =
          _mm256_sub_ps(_mm256_load_ps(lhs + 8), _mm256_load_ps(rhs + 8));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d_0, ymm_d_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_d_1, ymm_d_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      __m256 ymm_d = _mm256_sub_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d, ymm_d, ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_d_0 =
          _mm256_sub_ps(_mm256_loadu_ps(lhs + 0), _mm256_loadu_ps(rhs + 0));
      __m256 ymm_d_1 =
          _mm256_sub_ps(_mm256_loadu_ps(lhs + 8), _mm256_loadu_ps(rhs + 8));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d_0, ymm_d_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_d_1, ymm_d_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      __m256 ymm_d = _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d, ymm_d, ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  }
  float result = HorizontalAdd_FP32_V256(_mm256_add_ps(ymm_sum_0, ymm_sum_1));

  switch (last - lhs) {
    case 7:
      SSD_FP32_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      SSD_FP32_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      SSD_FP32_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      SSD_FP32_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      SSD_FP32_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      SSD_FP32_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      SSD_FP32_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}

//! SquaredEuclideanDistance
void SquaredEuclideanDistanceAVX_2X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_2X1_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_2X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_2X2_AVX(lhs, rhs, size, out, )
}
void SquaredEuclideanDistanceAVX_4X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X1_AVX(lhs, rhs, size, out, )
}
void SquaredEuclideanDistanceAVX_4X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X2_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_4X4(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X4_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_8X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_8X1_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_8X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_8X2_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_8X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_8X4_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_8X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_8X8_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_16X1(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X1_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_16X2(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X2_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_16X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X4_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_16X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X8_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_16X16(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X16_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_32X1(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X1_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_32X2(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X2_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_32X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X4_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_32X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X8_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_32X16(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X16_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_32X32(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X32_AVX(lhs, rhs, size, out, )
}

//! EuclideanDistance
void EuclideanDistanceAVX_2X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_2X1_AVX(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceAVX_2X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_2X2_AVX(lhs, rhs, size, out, _mm_sqrt_ps)
}
void EuclideanDistanceAVX_4X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X1_AVX(lhs, rhs, size, out, _mm_sqrt_ps)
}
void EuclideanDistanceAVX_4X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X2_AVX(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceAVX_4X4(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X4_AVX(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceAVX_8X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_8X1_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_8X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_8X2_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_8X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_8X4_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_8X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_8X8_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_16X1(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X1_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_16X2(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X2_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_16X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X4_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_16X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X8_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_16X16(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X16_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_32X1(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X1_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_32X2(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X2_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_32X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X4_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_32X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X8_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_32X16(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X16_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_32X32(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X32_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

#endif  // __AVX__

}  // namespace ailego
}  // namespace zvec