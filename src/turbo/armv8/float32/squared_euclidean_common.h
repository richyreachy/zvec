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

//! Calculate Sum-of-Squared-Differences (GENERAL)
#define SSD_FP32_GENERAL(m, q, sum) \
  {                                 \
    float x = m - q;                \
    sum += (x * x);                 \
  }

namespace zvec::turbo::armv8::internal {

static __attribute__((always_inline)) void squared_euclidean_fp32_armv8(
    const void *a, const void *b, size_t size, float *distance) {
  const float *lhs = reinterpret_cast<const float *>(a);
  const float *rhs = reinterpret_cast<const float *>(b);

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
  *distance = result;
}

}  // namespace zvec::turbo::armv8::internal

#endif  // defined(__ARM_NEON)
