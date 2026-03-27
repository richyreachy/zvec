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

#if defined(__AVX2__)
#include <immintrin.h>
#include <array>
#include <cstdint>
#include <zvec/ailego/internal/platform.h>

namespace zvec::turbo::avx2::internal {

#define ONES_INT16_SSE _mm_set1_epi32(0x00010001)
#define ONES_INT16_AVX _mm256_set1_epi32(0x00010001)

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_INT8_GENERAL(m, q, sum) sum += static_cast<float>(m * q);

static inline int32_t HorizontalAdd_INT32_V256(__m256i v) {
  __m256i x1 = _mm256_hadd_epi32(v, v);
  __m256i x2 = _mm256_hadd_epi32(x1, x1);
  __m128i x3 = _mm256_extractf128_si256(x2, 1);
  __m128i x4 = _mm_add_epi32(_mm256_castsi256_si128(x2), x3);
  return _mm_cvtsi128_si32(x4);
}

static __attribute__((always_inline)) void inner_product_int8_avx2(
    const void *a, const void *b, size_t size, float *distance) {
  const int8_t *lhs = reinterpret_cast<const int8_t *>(a);
  const int8_t *rhs = reinterpret_cast<const int8_t *>(b);

  const int8_t *last = lhs + size;
  const int8_t *last_aligned = lhs + ((size >> 6) << 6);
  float result = 0.0;

  __m256i ymm_sum_0 = _mm256_setzero_si256();
  __m256i ymm_sum_1 = _mm256_setzero_si256();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_load_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_load_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_load_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_load_si256((const __m256i *)(rhs + 32));

      ymm_lhs_0 = _mm256_sign_epi8(ymm_lhs_0, ymm_rhs_0);
      ymm_lhs_1 = _mm256_sign_epi8(ymm_lhs_1, ymm_rhs_1);
      ymm_rhs_0 = _mm256_abs_epi8(ymm_rhs_0);
      ymm_rhs_1 = _mm256_abs_epi8(ymm_rhs_1);

      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_lhs_0),
                            ONES_INT16_AVX),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_1, ymm_lhs_1),
                            ONES_INT16_AVX),
          ymm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_load_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_load_si256((const __m256i *)rhs);
      ymm_lhs = _mm256_sign_epi8(ymm_lhs, ymm_rhs);
      ymm_rhs = _mm256_abs_epi8(ymm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs, ymm_lhs),
                            ONES_INT16_AVX),
          ymm_sum_0);
      lhs += 32;
      rhs += 32;
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)rhs);
      xmm_lhs = _mm_sign_epi8(xmm_lhs, xmm_rhs);
      xmm_rhs = _mm_abs_epi8(xmm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(),
                           _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs, xmm_lhs),
                                          ONES_INT16_SSE)),
          ymm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_loadu_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_loadu_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_loadu_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_loadu_si256((const __m256i *)(rhs + 32));

      ymm_lhs_0 = _mm256_sign_epi8(ymm_lhs_0, ymm_rhs_0);
      ymm_lhs_1 = _mm256_sign_epi8(ymm_lhs_1, ymm_rhs_1);
      ymm_rhs_0 = _mm256_abs_epi8(ymm_rhs_0);
      ymm_rhs_1 = _mm256_abs_epi8(ymm_rhs_1);

      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_lhs_0),
                            ONES_INT16_AVX),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_1, ymm_lhs_1),
                            ONES_INT16_AVX),
          ymm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_loadu_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_loadu_si256((const __m256i *)rhs);
      ymm_lhs = _mm256_sign_epi8(ymm_lhs, ymm_rhs);
      ymm_rhs = _mm256_abs_epi8(ymm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs, ymm_lhs),
                            ONES_INT16_AVX),
          ymm_sum_0);
      lhs += 32;
      rhs += 32;
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);
      xmm_lhs = _mm_sign_epi8(xmm_lhs, xmm_rhs);
      xmm_rhs = _mm_abs_epi8(xmm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(),
                           _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs, xmm_lhs),
                                          ONES_INT16_SSE)),
          ymm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  }
  result = static_cast<float>(
      HorizontalAdd_INT32_V256(_mm256_add_epi32(ymm_sum_0, ymm_sum_1)));

  switch (last - lhs) {
    case 15:
      FMA_INT8_GENERAL(lhs[14], rhs[14], result)
      /* FALLTHRU */
    case 14:
      FMA_INT8_GENERAL(lhs[13], rhs[13], result)
      /* FALLTHRU */
    case 13:
      FMA_INT8_GENERAL(lhs[12], rhs[12], result)
      /* FALLTHRU */
    case 12:
      FMA_INT8_GENERAL(lhs[11], rhs[11], result)
      /* FALLTHRU */
    case 11:
      FMA_INT8_GENERAL(lhs[10], rhs[10], result)
      /* FALLTHRU */
    case 10:
      FMA_INT8_GENERAL(lhs[9], rhs[9], result)
      /* FALLTHRU */
    case 9:
      FMA_INT8_GENERAL(lhs[8], rhs[8], result)
      /* FALLTHRU */
    case 8:
      FMA_INT8_GENERAL(lhs[7], rhs[7], result)
      /* FALLTHRU */
    case 7:
      FMA_INT8_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      FMA_INT8_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      FMA_INT8_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      FMA_INT8_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      FMA_INT8_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_INT8_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_INT8_GENERAL(lhs[0], rhs[0], result)
  }

  *distance = result;
}

template <size_t batch_size>
__attribute__((always_inline)) void inner_product_int8_batch_avx2_impl(
    const void *query, const void *const *vectors,
    const std::array<const void *, batch_size> &prefetch_ptrs,
    size_t dimensionality, float *distances) {
  // TBD
}

static __attribute__((always_inline)) void inner_product_int8_batch_avx2(
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
    inner_product_int8_batch_avx2_impl<batch_size>(
        query, &vectors[i], prefetch_ptrs, dim, distances + i);
  }
  for (; i < n; i++) {
    std::array<const void *, 1> prefetch_ptrs{nullptr};
    inner_product_int8_batch_avx2_impl<1>(query, &vectors[i], prefetch_ptrs,
                                          dim, distances + i);
  }
}

}  // namespace zvec::turbo::avx2::internal

#endif  // defined(__AVX2__)
