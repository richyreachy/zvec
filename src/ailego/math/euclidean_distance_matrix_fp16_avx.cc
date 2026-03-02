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
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {


#define ACCUM_FP32_STEP_SSE SSD_FP32_SSE
#define ACCUM_FP32_STEP_AVX SSD_FP32_AVX
#define ACCUM_FP16_STEP_GENERAL SSD_FP16_GENERAL

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


//! Calculate sum of squared difference (GENERAL)
#define SSD_FP16_GENERAL(m, q, sum) \
  {                                 \
    float x = m - q;                \
    sum += (x * x);                 \
  }

#if defined(__AVX__)

void SquaredEuclideanDistanceAVX(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_1X1_AVX(lhs, rhs, size, out, 0ull, )                                          
}

//! SquaredEuclideanDistance
void SquaredEuclideanDistanceAVX_2X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_2X1_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_2X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_2X2_AVX(lhs, rhs, size, out, )
}
void SquaredEuclideanDistanceAVX_4X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_4X1_AVX(lhs, rhs, size, out, )
}
void SquaredEuclideanDistanceAVX_4X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_4X2_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_4X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_4X4_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_8X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_8X1_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_8X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_8X2_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_8X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_8X4_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_8X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_8X8_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_16X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X1_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_16X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X2_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_16X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X4_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_16X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X8_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_16X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X16_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_32X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X1_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_32X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X2_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_32X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X4_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_32X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X8_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_32X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X16_AVX(lhs, rhs, size, out, )
}

void SquaredEuclideanDistanceAVX_32X32(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X32_AVX(lhs, rhs, size, out, )
}

//! EuclideanDistance
void EuclideanDistanceAVX_1X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_1X1_AVX(lhs, rhs, size, out, 0ull, std::sqrt)
}

void EuclideanDistanceAVX_2X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_2X1_AVX(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceAVX_2X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_2X2_AVX(lhs, rhs, size, out, _mm_sqrt_ps)
}
void EuclideanDistanceAVX_4X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_4X1_AVX(lhs, rhs, size, out, _mm_sqrt_ps)
}
void EuclideanDistanceAVX_4X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_4X2_AVX(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceAVX_4X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_4X4_AVX(lhs, rhs, size, out, _mm_sqrt_ps)
}

void EuclideanDistanceAVX_8X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_8X1_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_8X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_8X2_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_8X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_8X4_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_8X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_8X8_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_16X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X1_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_16X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X2_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_16X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X4_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_16X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X8_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_16X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_16X16_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_32X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X1_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_32X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X2_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_32X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X4_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_32X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X8_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_32X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X16_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

void EuclideanDistanceAVX_32X32(const Float16 *lhs, const Float16 *rhs, size_t size, float *out){
  ACCUM_FP16_32X32_AVX(lhs, rhs, size, out, _mm256_sqrt_ps)
}

#endif  // __AVX__

}  // namespace ailego
}  // namespace zvec