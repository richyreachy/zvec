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
#define ACCUM_FP32_STEP_AVX512 SSD_FP32_AVX512
#define ACCUM_FP32_STEP_NEON SSD_FP32_NEON

//! Calculate sum of squared difference (GENERAL)
#define SSD_FP32_GENERAL(m, q, sum) \
  {                                 \
    float x = m - q;                \
    sum += (x * x);                 \
  }

//! Calculate sum of squared difference (NEON)
#define SSD_FP32_NEON(v_m, v_q, v_sum)     \
  {                                        \
    float32x4_t v_d = vsubq_f32(v_m, v_q); \
    v_sum = vfmaq_f32(v_sum, v_d, v_d);    \
  }

#if defined(__ARM_NEON)
//! Squared Euclidean Distance
void SquaredEuclideanDistanceNEON(const float *lhs, const float *rhs, size_t size, float *out) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 3) << 3);

  float32x4_t v_sum_0 = vdupq_n_f32(0);
  float32x4_t v_sum_1 = vdupq_n_f32(0);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    float32x4_t v_d_0 = vsubq_f32(vld1q_f32(lhs + 0), vld1q_f32(rhs + 0));
    float32x4_t v_d_1 = vsubq_f32(vld1q_f32(lhs + 4), vld1q_f32(rhs + 4));
    v_sum_0 = vfmaq_f32(v_sum_0, v_d_0, v_d_0);
    v_sum_1 = vfmaq_f32(v_sum_1, v_d_1, v_d_1);
  }
  if (last >= last_aligned + 4) {
    float32x4_t v_d = vsubq_f32(vld1q_f32(lhs), vld1q_f32(rhs));
    v_sum_0 = vfmaq_f32(v_sum_0, v_d, v_d);
    lhs += 4;
    rhs += 4;
  }

  float result = vaddvq_f32(vaddq_f32(v_sum_0, v_sum_1));
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
void SquaredEuclideanDistanceNEON_2X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_2X1_NEON(lhs, rhs, size, &score, )

  return score;
}

void SquaredEuclideanDistanceNEON_2X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_2X2_NEON(lhs, rhs, size, &score, )
}
void SquaredEuclideanDistanceNEON_4X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X1_NEON(lhs, rhs, size, &score, )
}
void SquaredEuclideanDistanceNEON_4X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X2_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_4X4(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X4_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_8X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_8X1_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_8X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_8X2_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_8X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_8X4_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_8X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_8X8_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_16X1(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X1_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_16X2(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X2_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_16X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X4_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_16X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X8_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_16X16(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X16_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_32X1(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X1_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_32X2(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X2_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_32X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X4_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_32X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X8_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_32X16(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X16_NEON(lhs, rhs, size, &score, )
}

void SquaredEuclideanDistanceNEON_32X32(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X32_NEON(lhs, rhs, size, &score, )
}

//! EuclideanDistance
void EuclideanDistanceNEON_2X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_2X1_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_2X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_2X2_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}
void EuclideanDistanceNEON_4X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X1_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}
void EuclideanDistanceNEON_4X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X2_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_4X4(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_4X4_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_8X1(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_8X1_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_8X2(const float *lhs, const float *rhs, size_t size, float *out) {
  ACCUM_FP32_8X2_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_8X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_8X4_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_8X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_8X8_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_16X1(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X1_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_16X2(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X2_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_16X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X4_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_16X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X8_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_16X16(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_16X16_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_32X1(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X1_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_32X2(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X2_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_32X4(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X4_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_32X8(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X8_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_32X16(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X16_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

void EuclideanDistanceNEON_32X32(const float *lhs, const float *rhs, size_t size, float *out){
  ACCUM_FP32_32X32_NEON(lhs, rhs, size, &score, vsqrtq_f32)
}

#endif  // __ARM_NEON

}  // namespace ailego
}  // namespace zvec