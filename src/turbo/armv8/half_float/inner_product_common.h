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

#if defined(__ARM_NEON)
#include <array>
#include <cstdint>
#include <arm_neon.h>
#include <zvec/ailego/utility/float_helper.h>

using namespace zvec::ailego;

namespace zvec::turbo::armv8::internal {

#define MATRIX_VAR_INIT_1X1(_VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  _VAR_TYPE _VAR_NAME##_0_0 = (_VAR_INIT);

#define MATRIX_VAR_INIT(_M, _N, _VAR_TYPE, _VAR_NAME, _VAR_INIT) \
  MATRIX_VAR_INIT_##_M##X##_N(_VAR_TYPE, _VAR_NAME, _VAR_INIT)

//! Scalar fused multiply-add for inner product (FP16 general)
#define ACCUM_FP16_STEP_GENERAL(m, q, sum) sum += (m * q);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

//! NEON fused multiply-add for inner product (FP16)
#define ACCUM_FP16_STEP_NEON(v_m, v_q, v_sum) v_sum = vfmaq_f16(v_sum, v_m, v_q);

//! Iterative process of computing distance (FP16, M=1, N=1)
#define MATRIX_FP16_ITER_1X1_NEON(m, q, _RES, _PROC)   \
  {                                                    \
    float16x8_t v_m = vld1q_f16((const float16_t *)m); \
    float16x8_t v_q = vld1q_f16((const float16_t *)q); \
    _PROC(v_m, v_q, _RES##_0_0)                        \
  }

//! Compute the distance between matrix and query (FP16, M=1, N=1)
#define ACCUM_FP16_1X1_NEON(m, q, dim, out, _MASK, _NORM)                    \
  MATRIX_VAR_INIT(1, 1, float16x8_t, v_sum, vdupq_n_f16(0))                  \
  const Float16 *qe = q + dim;                                               \
  const Float16 *qe_aligned = q + ((dim >> 3) << 3);                         \
  for (; q != qe_aligned; m += 8, q += 8) {                                  \
    MATRIX_FP16_ITER_1X1_NEON(m, q, v_sum, ACCUM_FP16_STEP_NEON)             \
  }                                                                          \
  if (qe >= qe_aligned + 4) {                                                \
    float16x8_t v_m =                                                        \
        vcombine_f16(vld1_f16((const float16_t *)m),                         \
                     vreinterpret_f16_u64(vdup_n_u64((uint64_t)(_MASK))));   \
    float16x8_t v_q =                                                        \
        vcombine_f16(vld1_f16((const float16_t *)q),                         \
                     vreinterpret_f16_u64(vdup_n_u64((uint64_t)(_MASK))));   \
    ACCUM_FP16_STEP_NEON(v_m, v_q, v_sum_0_0)                                \
    m += 4;                                                                  \
    q += 4;                                                                  \
  }                                                                          \
  float result = vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(v_sum_0_0)), \
                                      vcvt_high_f32_f16(v_sum_0_0)));        \
  switch (qe - q) {                                                          \
    case 3:                                                                  \
      ACCUM_FP16_STEP_GENERAL(m[2], q[2], result)                            \
      /* FALLTHRU */                                                         \
    case 2:                                                                  \
      ACCUM_FP16_STEP_GENERAL(m[1], q[1], result)                            \
      /* FALLTHRU */                                                         \
    case 1:                                                                  \
      ACCUM_FP16_STEP_GENERAL(m[0], q[0], result)                            \
  }                                                                          \
  *out = _NORM(result);

#else

//! NEON fused multiply-add for inner product (FP32)
#define ACCUM_FP32_STEP_NEON(v_m, v_q, v_sum) v_sum = vfmaq_f32(v_sum, v_m, v_q);

//! Iterative process of computing distance (FP16, M=1, N=1)
#define MATRIX_FP16_ITER_1X1_NEON(m, q, _RES, _PROC)     \
  {                                                      \
    float16x8_t v_m = vld1q_f16((const float16_t *)m);   \
    float16x8_t v_q = vld1q_f16((const float16_t *)q);   \
    float32x4_t v_m_0 = vcvt_f32_f16(vget_low_f16(v_m)); \
    float32x4_t v_q_0 = vcvt_f32_f16(vget_low_f16(v_q)); \
    _PROC(v_m_0, v_q_0, _RES##_0_0)                      \
    v_m_0 = vcvt_high_f32_f16(v_m);                      \
    v_q_0 = vcvt_high_f32_f16(v_q);                      \
    _PROC(v_m_0, v_q_0, _RES##_0_0)                      \
  }

//! Compute the distance between matrix and query (FP16, M=1, N=1)
#define ACCUM_FP16_1X1_NEON(m, q, dim, out, _MASK, _NORM)           \
  MATRIX_VAR_INIT(1, 1, float32x4_t, v_sum, vdupq_n_f32(0))         \
  const Float16 *qe = q + dim;                                      \
  const Float16 *qe_aligned = q + ((dim >> 3) << 3);                \
  for (; q != qe_aligned; m += 8, q += 8) {                         \
    MATRIX_FP16_ITER_1X1_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                 \
  if (qe >= qe_aligned + 4) {                                       \
    float32x4_t v_m = vcvt_f32_f16(vld1_f16((const float16_t *)m)); \
    float32x4_t v_q = vcvt_f32_f16(vld1_f16((const float16_t *)q)); \
    ACCUM_FP32_STEP_NEON(v_m, v_q, v_sum_0_0)                       \
    m += 4;                                                         \
    q += 4;                                                         \
  }                                                                 \
  float result = vaddvq_f32(v_sum_0_0);                             \
  switch (qe - q) {                                                 \
    case 3:                                                         \
      ACCUM_FP16_STEP_GENERAL(m[2], q[2], result)                   \
      /* FALLTHRU */                                                \
    case 2:                                                         \
      ACCUM_FP16_STEP_GENERAL(m[1], q[1], result)                   \
      /* FALLTHRU */                                                \
    case 1:                                                         \
      ACCUM_FP16_STEP_GENERAL(m[0], q[0], result)                   \
  }                                                                 \
  *out = _NORM(result);

#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

}  // namespace zvec::turbo::armv8::internal

#endif  // defined(__ARM_NEON)
