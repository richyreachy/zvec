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
#define ACCUM_FP32_STEP_AVX512 SSD_FP32_AVX512
#define ACCUM_FP32_STEP_NEON SSD_FP32_NEON
#define ACCUM_FP16_STEP_GENERAL SSD_FP16_GENERAL
#define ACCUM_FP16_STEP_NEON SSD_FP16_NEON

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

//! Calculate sum of squared difference (AVX512)
#define SSD_FP32_AVX512(zmm_m, zmm_q, zmm_sum)        \
  {                                                   \
    __m512 zmm_d = _mm512_sub_ps(zmm_m, zmm_q);       \
    zmm_sum = _mm512_fmadd_ps(zmm_d, zmm_d, zmm_sum); \
  }

//! Calculate sum of squared difference (GENERAL)
#define SSD_FP16_GENERAL(m, q, sum) \
  {                                 \
    float x = m - q;                \
    sum += (x * x);                 \
  }

//! Calculate sum of squared difference (NEON)
#define SSD_FP16_NEON(v_m, v_q, v_sum)     \
  {                                        \
    float16x8_t v_d = vsubq_f16(v_m, v_q); \
    v_sum = vfmaq_f16(v_sum, v_d, v_d);    \
  }

//! Calculate sum of squared difference (NEON)
#define SSD_FP32_NEON(v_m, v_q, v_sum)     \
  {                                        \
    float32x4_t v_d = vsubq_f32(v_m, v_q); \
    v_sum = vfmaq_f32(v_sum, v_d, v_d);    \
  }

}  // namespace ailego
}  // namespace zvec