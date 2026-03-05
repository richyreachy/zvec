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

#define ACCUM_FP32_STEP_NEON SSD_FP32_NEON
#define ACCUM_FP16_STEP_GENERAL SSD_FP16_GENERAL
#define ACCUM_FP16_STEP_NEON SSD_FP16_NEON

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

#if defined(__ARM_NEON)
void SquaredEuclideanDistanceNEON(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_1X1_NEON(lhs, rhs, size, out, 0ull, )                                        
}

void EuclideanDistanceNEON(const Float16 *lhs, const Float16 *rhs, size_t size, float *out) {
  ACCUM_FP16_1X1_NEON(lhs, rhs, size, out, 0ull, std::sqrt)                                        
}
#endif

}  // namespace ailego
}  // namespace zvec