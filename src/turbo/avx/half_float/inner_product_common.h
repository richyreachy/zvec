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

// Shared AVX512-VNNI inner product kernels for record_quantized_int8 distance
// implementations (cosine, l2, mips_l2, etc.).
//
// All functions are marked always_inline so that when this header is included
// from a per-file-march .cc translation unit, the compiler can fully inline
// and optimize them under the correct -march flag without any cross-TU call
// overhead.

#pragma once

#if defined(__AVX__)

#include <zvec/ailego/utility/float_helper.h>

using namespace zvec::ailego;

//! Calculate Fused-Multiply-Add (AVX)
#define FMA_FP32_AVX(ymm_m, ymm_q, ymm_sum) \
  ymm_sum = _mm256_fmadd_ps(ymm_m, ymm_q, ymm_sum);

#define ACCUM_FP32_STEP_AVX FMA_FP32_AVX

#define MATRIX_VAR_INIT_1X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  _VAR_TYPE _VAR_NAME##_0_0 = (_VAR_INIT);


#define MATRIX_VAR_INIT(_M, _N, _VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_##_M##X##_N(_VAR_TYPE, _VAR_NAME, _VAR_INIT)

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

#endif