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

#if defined(__SSE__)
float SquaredEuclideanDistanceSSE(const float *lhs, const float *rhs, size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 3) << 3);

  __m128 xmm_sum_0 = _mm_setzero_ps();
  __m128 xmm_sum_1 = _mm_setzero_ps();

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m128 xmm_d_0 = _mm_sub_ps(_mm_load_ps(lhs + 0), _mm_load_ps(rhs + 0));
      __m128 xmm_d_1 = _mm_sub_ps(_mm_load_ps(lhs + 4), _mm_load_ps(rhs + 4));
      xmm_sum_0 = _mm_fmadd_ps(xmm_d_0, xmm_d_0, xmm_sum_0);
      xmm_sum_1 = _mm_fmadd_ps(xmm_d_1, xmm_d_1, xmm_sum_1);
    }

    if (last >= last_aligned + 4) {
      __m128 xmm_d = _mm_sub_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
      xmm_sum_0 = _mm_fmadd_ps(xmm_d, xmm_d, xmm_sum_0);
      lhs += 4;
      rhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m128 xmm_d_0 = _mm_sub_ps(_mm_loadu_ps(lhs + 0), _mm_loadu_ps(rhs + 0));
      __m128 xmm_d_1 = _mm_sub_ps(_mm_loadu_ps(lhs + 4), _mm_loadu_ps(rhs + 4));
      xmm_sum_0 = _mm_fmadd_ps(xmm_d_0, xmm_d_0, xmm_sum_0);
      xmm_sum_1 = _mm_fmadd_ps(xmm_d_1, xmm_d_1, xmm_sum_1);
    }

    if (last >= last_aligned + 4) {
      __m128 xmm_d = _mm_sub_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
      xmm_sum_0 = _mm_fmadd_ps(xmm_d, xmm_d, xmm_sum_0);
      lhs += 4;
      rhs += 4;
    }
  }
  float result = HorizontalAdd_FP32_V128(_mm_add_ps(xmm_sum_0, xmm_sum_1));

  switch (last - lhs) {
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
void SquaredEuclideanDistanceSSE_2X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_2X1_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_2X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_2X2_SSE(lhs, rhs, size, out, )
}
void SquaredEuclideanDistanceSSE_4X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X1_SSE(lhs, rhs, size, out, )
}
void SquaredEuclideanDistanceSSE_4X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X2_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_4X4(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X4_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_8X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_8X1_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_8X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_8X2_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_8X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_8X4_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_8X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_8X8_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_16X1(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X1_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_16X2(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X2_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_16X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X4_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_16X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X8_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_16X16(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X16_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_32X1(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X1_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_32X2(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X2_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_32X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X4_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_32X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X8_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_32X16(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X16_SSE(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceSSE_32X32(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X32_SSE(lhs, rhs, size, out, )
}

//! EuclideanDistance
void EuclideanDistanceSSE_2X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_2X1_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_2X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_2X2_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}
void EuclideanDistanceSSE_4X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X1_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}
void EuclideanDistanceSSE_4X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X2_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_4X4(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X4_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_8X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_8X1_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_8X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_8X2_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_8X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_8X4_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_8X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_8X8_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_16X1(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X1_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_16X2(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X2_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_16X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X4_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_16X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X8_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_16X16(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X16_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_32X1(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X1_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_32X2(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X2_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_32X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X4_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_32X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X8_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_32X16(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X16_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceSSE_32X32(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X32_SSE(lhs, rhs, size, out, _mm_sqrt_ps)
}

#endif  // __SSE__

}  // namespace ailego
}  // namespace zvec