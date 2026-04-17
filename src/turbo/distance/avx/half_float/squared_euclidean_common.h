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

#pragma once

#if defined(__AVX__)

#include <immintrin.h>
#include <zvec/ailego/utility/float_helper.h>

using namespace zvec::ailego;

namespace zvec::turbo::avx {

//! Mask process of computing distance (FP16)
#define MATRIX_FP16_MASK_AVX(lhs, rhs, cnt, _MASK, _RES, _PROC)              \
  switch (cnt) {                                                             \
    case 7: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(_mm_set_epi16(                        \
          (short)(_MASK), *((const short *)(lhs) + 6),                       \
          *((const short *)(lhs) + 5), *((const short *)(lhs) + 4),          \
          *((const short *)(lhs) + 3), *((const short *)(lhs) + 2),          \
          *((const short *)(lhs) + 1), *((const short *)(lhs))));            \
      __m256 ymm_rhs = _mm256_cvtph_ps(_mm_set_epi16(                        \
          (short)(_MASK), *((const short *)(rhs) + 6),                       \
          *((const short *)(rhs) + 5), *((const short *)(rhs) + 4),          \
          *((const short *)(rhs) + 3), *((const short *)(rhs) + 2),          \
          *((const short *)(rhs) + 1), *((const short *)(rhs))));            \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
    case 6: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(                                      \
          _mm_set_epi32((int)(_MASK), *((const int *)(lhs) + 2),             \
                        *((const int *)(lhs) + 1), *((const int *)(lhs))));  \
      __m256 ymm_rhs = _mm256_cvtph_ps(                                      \
          _mm_set_epi32((int)(_MASK), *((const int *)(rhs) + 2),             \
                        *((const int *)(rhs) + 1), *((const int *)(rhs))));  \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
    case 5: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(_mm_set_epi16(                        \
          (short)(_MASK), (short)(_MASK), (short)(_MASK),                    \
          *((const short *)(lhs) + 4), *((const short *)(lhs) + 3),          \
          *((const short *)(lhs) + 2), *((const short *)(lhs) + 1),          \
          *((const short *)(lhs))));                                         \
      __m256 ymm_rhs = _mm256_cvtph_ps(_mm_set_epi16(                        \
          (short)(_MASK), (short)(_MASK), (short)(_MASK),                    \
          *((const short *)(rhs) + 4), *((const short *)(rhs) + 3),          \
          *((const short *)(rhs) + 2), *((const short *)(rhs) + 1),          \
          *((const short *)(rhs))));                                         \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
    case 4: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(                                      \
          _mm_set_epi64((__m64)(_MASK), *((const __m64 *)(lhs))));           \
      __m256 ymm_rhs = _mm256_cvtph_ps(                                      \
          _mm_set_epi64((__m64)(_MASK), *((const __m64 *)(rhs))));           \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
    case 3: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(_mm_set_epi16(                        \
          (short)(_MASK), (short)(_MASK), (short)(_MASK), (short)(_MASK),    \
          (short)(_MASK), *((const short *)(lhs) + 2),                       \
          *((const short *)(lhs) + 1), *((const short *)(lhs))));            \
      __m256 ymm_rhs = _mm256_cvtph_ps(_mm_set_epi16(                        \
          (short)(_MASK), (short)(_MASK), (short)(_MASK), (short)(_MASK),    \
          (short)(_MASK), *((const short *)(rhs) + 2),                       \
          *((const short *)(rhs) + 1), *((const short *)(rhs))));            \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
    case 2: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(_mm_set_epi32(                        \
          (int)(_MASK), (int)(_MASK), (int)(_MASK), *((const int *)(lhs)))); \
      __m256 ymm_rhs = _mm256_cvtph_ps(_mm_set_epi32(                        \
          (int)(_MASK), (int)(_MASK), (int)(_MASK), *((const int *)(rhs)))); \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
    case 1: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(                                      \
          _mm_set_epi16(*((const short *)(lhs)), (short)(_MASK),             \
                        (short)(_MASK), (short)(_MASK), (short)(_MASK),      \
                        (short)(_MASK), (short)(_MASK), (short)(_MASK)));    \
      __m256 ymm_rhs = _mm256_cvtph_ps(                                      \
          _mm_set_epi16(*((const short *)(rhs)), (short)(_MASK),             \
                        (short)(_MASK), (short)(_MASK), (short)(_MASK),      \
                        (short)(_MASK), (short)(_MASK), (short)(_MASK)));    \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
  }

static inline float HorizontalAdd_FP32_V256(__m256 v) {
  __m256 x1 = _mm256_hadd_ps(v, v);
  __m256 x2 = _mm256_hadd_ps(x1, x1);
  __m128 x3 = _mm256_extractf128_ps(x2, 1);
  __m128 x4 = _mm_add_ss(_mm256_castps256_ps128(x2), x3);
  return _mm_cvtss_f32(x4);
}

//! Calculate sum of squared difference (AVX)
#define SSD_FP32_AVX(ymm_m, ymm_q, ymm_sum)           \
  {                                                   \
    __m256 ymm_d = _mm256_sub_ps(ymm_m, ymm_q);       \
    ymm_sum = _mm256_fmadd_ps(ymm_d, ymm_d, ymm_sum); \
  }

#define ACCUM_FP32_STEP_AVX SSD_FP32_AVX

#define MATRIX_VAR_INIT_1X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  _VAR_TYPE _VAR_NAME##_0_0 = (_VAR_INIT);

#define MATRIX_VAR_INIT(_M, _N, _VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_##_M##X##_N(_VAR_TYPE, _VAR_NAME, _VAR_INIT)

//! Iterative process of computing distance (FP16, M=1, N=1)
#define MATRIX_FP16_ITER_1X1_AVX(m, q, _RES, _LOAD, _PROC)          \
  {                                                                 \
    __m256i ymm_mi = _LOAD((const __m256i *)m);                     \
    __m256i ymm_qi = _LOAD((const __m256i *)q);                     \
    __m256 ymm_m = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi)); \
    __m256 ymm_q = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_qi)); \
    _PROC(ymm_m, ymm_q, _RES##_0_0);                                \
    ymm_m = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1));   \
    ymm_q = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_qi, 1));   \
    _PROC(ymm_m, ymm_q, _RES##_0_0);                                \
  }

//! Compute the distance between matrix and query (FP16, M=1, N=1)
#define ACCUM_FP16_1X1_AVX(m, q, dim, out, _MASK, _NORM)                    \
  MATRIX_VAR_INIT(1, 1, __m256, ymm_sum, _mm256_setzero_ps())               \
  const Float16 *qe = q + dim;                                              \
  const Float16 *qe_aligned = q + ((dim >> 4) << 4);                        \
  if (((uintptr_t)m & 0x1f) == 0 && ((uintptr_t)q & 0x1f) == 0) {           \
    for (; q != qe_aligned; m += 16, q += 16) {                             \
      MATRIX_FP16_ITER_1X1_AVX(m, q, ymm_sum, _mm256_load_si256,            \
                               ACCUM_FP32_STEP_AVX)                         \
    }                                                                       \
    if (qe >= qe_aligned + 8) {                                             \
      __m256 ymm_m = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)m));   \
      __m256 ymm_q = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)q));   \
      ACCUM_FP32_STEP_AVX(ymm_m, ymm_q, ymm_sum_0_0)                        \
      m += 8;                                                               \
      q += 8;                                                               \
    }                                                                       \
  } else {                                                                  \
    for (; q != qe_aligned; m += 16, q += 16) {                             \
      MATRIX_FP16_ITER_1X1_AVX(m, q, ymm_sum, _mm256_loadu_si256,           \
                               ACCUM_FP32_STEP_AVX)                         \
    }                                                                       \
    if (qe >= qe_aligned + 8) {                                             \
      __m256 ymm_m = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)m));  \
      __m256 ymm_q = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q));  \
      ACCUM_FP32_STEP_AVX(ymm_m, ymm_q, ymm_sum_0_0)                        \
      m += 8;                                                               \
      q += 8;                                                               \
    }                                                                       \
  }                                                                         \
  MATRIX_FP16_MASK_AVX(m, q, (qe - q), _MASK, ymm_sum, ACCUM_FP32_STEP_AVX) \
  *out = _NORM(HorizontalAdd_FP32_V256(ymm_sum_0_0));

}  // namespace zvec::turbo::avx

#endif