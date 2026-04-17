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

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_FP32_GENERAL(m, q, sum) sum += (m * q);

namespace zvec::turbo::armv8::internal {

static __attribute__((always_inline)) void inner_product_fp32_armv8(
    const void *a, const void *b, size_t size, float *distance) {
  const float *lhs = reinterpret_cast<const float *>(a);
  const float *rhs = reinterpret_cast<const float *>(b);

  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 3) << 3);

  float32x4_t v_sum_0 = vdupq_n_f32(0);
  float32x4_t v_sum_1 = vdupq_n_f32(0);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    v_sum_0 = vfmaq_f32(v_sum_0, vld1q_f32(lhs + 0), vld1q_f32(rhs + 0));
    v_sum_1 = vfmaq_f32(v_sum_1, vld1q_f32(lhs + 4), vld1q_f32(rhs + 4));
  }
  if (last >= last_aligned + 4) {
    v_sum_0 = vfmaq_f32(v_sum_0, vld1q_f32(lhs), vld1q_f32(rhs));
    lhs += 4;
    rhs += 4;
  }

  float result = vaddvq_f32(vaddq_f32(v_sum_0, v_sum_1));
  switch (last - lhs) {
    case 3:
      FMA_FP32_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_FP32_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_FP32_GENERAL(lhs[0], rhs[0], result)
  }
  *distance = -result;
}

template <size_t batch_size>
static __attribute__((always_inline)) void inner_product_fp32_batch_armv8_impl(
    const void *query, const void *const *vectors,
    const std::array<const void *, batch_size> &prefetch_ptrs,
    size_t dimensionality, float *distances) {
  float32x4_t v_sum[batch_size] for (size_t i = 0; i < batch_size; ++i) {
    v_sum[i] = vdupq_n_f32(0);
  }

  size_t dim = 0;
  for (; dim + 64 <= dimensionality; dim += 4) {
    for (size_t i = 0; i < batch_size; ++i) {
      v_sum[i] = vfmaq_f32(
          v_sum[i], vld1q_f32(reinterpret_cast<const float *>(query) + dim),
          vld1q_f32(reinterpret_cast<const float *>(vectors[i]) + dim));
    }
  }

  if (dim >= dimensionality + 4) {
    for (size_t i = 0; i < batch_size; ++i) {
      v_sum[i] = vfmaq_f32(v_sum[i], vld1q_f32(reinterpret_cast<const float *>(query)+dim), vld1q_f32(reinterpret_cast<const float *>(vectors[i])+dim)));
    }

    dim += 4;
  }

  for (size_t i = 0; i < batch_size; ++i) {
    float result = vaddvq_f32(v_sum[i]);
    switch (last - lhs) {
      case 3:
        FMA_FP32_GENERAL(reinterpret_cast<const float *>(query)[dim + 2],
                         reinterpret_cast<const float *>(vectors[i])[dim + 2],
                         result)
        /* FALLTHRU */
      case 2:
        FMA_FP32_GENERAL(reinterpret_cast<const float *>(query)[dim + 1],
                         reinterpret_cast<const float *>(vectors[i])[dim + 1],
                         result)
        /* FALLTHRU */
      case 1:
        FMA_FP32_GENERAL(reinterpret_cast<const float *>(query)[dim + 0],
                         reinterpret_cast<const float *>(vectors[i])[dim + 0],
                         result)
    }

    distances[i] = -result;
  }
}

// Dispatch batched inner product over all `n` vectors with prefetching.
static __attribute__((always_inline)) void inner_product_fp32_batch_armv8(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances) {
  static constexpr size_t batch_size = 2;
  static constexpr size_t prefetch_step = 2;
  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const void *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      if (i + j + batch_size * prefetch_step < n) {
        prefetch_ptrs[j] = vectors[i + j + batch_size * prefetch_step];
      } else {
        prefetch_ptrs[j] = nullptr;
      }
    }
    inner_product_fp32_batch_armv8_impl<batch_size>(
        query, &vectors[i], prefetch_ptrs, dim, distances + i);
  }
  for (; i < n; i++) {
    std::array<const void *, 1> prefetch_ptrs{nullptr};
    inner_product_fp32_batch_armv8_impl<1>(query, &vectors[i], prefetch_ptrs,
                                           dim, distances + i);
  }
}

}  // namespace zvec::turbo::armv8::internal

#endif  // defined(__ARM_NEON)
