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
#include <array>
#include <type_traits>
#include <zvec/ailego/internal/platform.h>

#define SSD_FP32_GENERAL(m, q, sum) \
  {                                 \
    float x = m - q;                \
    sum += (x * x);                 \
  }

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_FP32_GENERAL(m, q, sum) sum += (m * q);

static inline float HorizontalAdd_FP32_V256(__m256 v) {
  __m256 x1 = _mm256_hadd_ps(v, v);
  __m256 x2 = _mm256_hadd_ps(x1, x1);
  __m128 x3 = _mm256_extractf128_ps(x2, 1);
  __m128 x4 = _mm_add_ss(_mm256_castps256_ps128(x2), x3);
  return _mm_cvtss_f32(x4);
}

static inline float sum4(__m128 v) {
  v = _mm_add_ps(v, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), 8)));
  return _mm_cvtss_f32(v) + _mm_cvtss_f32(_mm_shuffle_ps(v, v, 1));
}

static inline __m128 sum_top_bottom_avx(__m256 v) {
  const __m128 high = _mm256_extractf128_ps(v, 1);
  const __m128 low = _mm256_castps256_ps128(v);
  return _mm_add_ps(high, low);
}


template <typename ValueType, size_t dp_batch>
static std::enable_if_t<std::is_same_v<ValueType, float>, void>
inner_product_fp32_batch_avx_impl(
    const ValueType *query, const ValueType *const *ptrs,
    std::array<const ValueType *, dp_batch> &prefetch_ptrs,
    size_t dimensionality, float *results) {
  __m256 accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm256_setzero_ps();
  }
  size_t dim = 0;
  for (; dim + 8 <= dimensionality; dim += 8) {
    __m256 q = _mm256_loadu_ps(query + dim);

    __m256 data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm256_loadu_ps(ptrs[i] + dim);
    }
    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }
    for (size_t i = 0; i < dp_batch; ++i) {
      accs[i] = _mm256_fnmadd_ps(q, data_regs[i], accs[i]);
    }
  }

  __m128 sum128_regs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    sum128_regs[i] = sum_top_bottom_avx(accs[i]);
  }
  if (dim + 4 <= dimensionality) {
    __m128 q = _mm_loadu_ps(query + dim);

    __m128 data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm_loadu_ps(ptrs[i] + dim);
    }
    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }
    for (size_t i = 0; i < dp_batch; ++i) {
      sum128_regs[i] = _mm_fnmadd_ps(q, data_regs[i], sum128_regs[i]);
    }
    dim += 4;
  }
  if (dim + 2 <= dimensionality) {
    __m128 q = _mm_setzero_ps();

    __m128 data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm_setzero_ps();
    }

    q = _mm_loadh_pi(q, (const __m64 *)(query + dim));
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm_loadh_pi(data_regs[i], (const __m64 *)(ptrs[i] + dim));
    }
    for (size_t i = 0; i < dp_batch; ++i) {
      sum128_regs[i] = _mm_fnmadd_ps(q, data_regs[i], sum128_regs[i]);
    }
    dim += 2;
  }

  float res[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    res[i] = sum4(sum128_regs[i]);
  }
  if (dim < dimensionality) {
    float q = query[dim];
    for (size_t i = 0; i < dp_batch; ++i) {
      res[i] -= q * ptrs[i][dim];
    }
  }
  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = -res[i];
  }
}

// Dispatch batched inner product over all `n` vectors with prefetching.
static __attribute__((always_inline)) void inner_product_fp32_batch_avx(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances) {
  static constexpr size_t batch_size = 2;
  static constexpr size_t prefetch_step = 2;
  const float *typed_query = reinterpret_cast<const float *>(query);
  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const float *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      if (i + j + batch_size * prefetch_step < n) {
        prefetch_ptrs[j] = reinterpret_cast<const float *>(
            vectors[i + j + batch_size * prefetch_step]);
      } else {
        prefetch_ptrs[j] = nullptr;
      }
    }
    inner_product_fp32_batch_avx_impl<float, batch_size>(
        typed_query, reinterpret_cast<const float *const *>(&vectors[i]),
        prefetch_ptrs, dim, distances + i);
  }
  for (; i < n; i++) {
    std::array<const float *, 1> prefetch_ptrs{nullptr};
    inner_product_fp32_batch_avx_impl<float, 1>(
        typed_query, reinterpret_cast<const float *const *>(&vectors[i]),
        prefetch_ptrs, dim, distances + i);
  }
}


#endif