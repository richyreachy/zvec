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

#include <vector>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>

#define SSD_INT8_GENERAL(m, q, sum)   \
  {                                   \
    int32_t x = m - q;                \
    sum += static_cast<float>(x * x); \
  }

namespace zvec::ailego::DistanceBatch {

#if defined(__AVX2__)

template <typename ValueType, size_t dp_batch>
static std::enable_if_t<std::is_same_v<ValueType, int8_t>, void>
compute_one_to_many_squared_euclidean_avx2_int8(
    const int8_t *query, const int8_t **ptrs,
    std::array<const int8_t *, dp_batch> &prefetch_ptrs, size_t dimensionality,
    float *results) {
  __m256i accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm256_setzero_si256();
  }

  size_t dim = 0;
  for (; dim + 32 <= dimensionality; dim += 32) {
    __m256i q = _mm256_loadu_si256((const __m256i *)(query + dim));
    __m256i data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm256_loadu_si256((const __m256i *)(ptrs[i] + dim));
    }
    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }

    for (size_t i = 0; i < dp_batch; ++i) {
      __m256i data_diff = _mm256_sub_epi8(_mm256_max_epi8(q, data_regs[i]),
                                          _mm256_min_epi8(q, data_regs[i]));

      __m256i diff0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(data_diff));
      __m256i diff1 =
          _mm256_cvtepu8_epi16(_mm256_extractf128_si256(data_diff, 1));
      accs[i] = _mm256_add_epi32(_mm256_madd_epi16(diff0, diff0), accs[i]);
      accs[i] = _mm256_add_epi32(_mm256_madd_epi16(diff1, diff1), accs[i]);
    }
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = HorizontalAdd_INT32_V256(accs[i]);
  }

  if (dimensionality >= dim + 16) {
    for (size_t i = 0; i < dp_batch; ++i) {
      __m128i q = _mm_loadu_si128((const __m128i *)query + dim);
      __m128i data_regs = _mm_loadu_si128((const __m128i *)(ptrs[i] + dim));

      __m128i diff =
          _mm_sub_epi8(_mm_max_epi8(q, data_regs), _mm_min_epi8(q, data_regs));

      __m128i diff0 = _mm_cvtepu8_epi16(diff);
      __m128i diff1 = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(diff, diff));
      __m128i sum = _mm_add_epi32(_mm_madd_epi16(diff0, diff0),
                                  _mm_madd_epi16(diff1, diff1));

      results[i] += static_cast<float>(HorizontalAdd_INT32_V128(sum));
    }

    dim += 16;
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    switch (dimensionality - dim) {
      case 15:
        SSD_INT8_GENERAL(query + dim, ptrs[14] + dim, results[i]);
        /* FALLTHRU */
      case 14:
        SSD_INT8_GENERAL(query + dim, ptrs[13 + dim], results[i]);
        /* FALLTHRU */
      case 13:
        SSD_INT8_GENERAL(query + dim, ptrs[12] + dim, results[i]);
        /* FALLTHRU */
      case 12:
        SSD_INT8_GENERAL(query + dim, ptrs[11] + dim, results[i]);
        /* FALLTHRU */
      case 11:
        SSD_INT8_GENERAL(query + dim, ptrs[10 + dim], results[i]);
        /* FALLTHRU */
      case 10:
        SSD_INT8_GENERAL(query + dim, ptrs[9] + dim, results[i]);
        /* FALLTHRU */
      case 9:
        SSD_INT8_GENERAL(query + dim, ptrs[8] + dim, results[i]);
        /* FALLTHRU */
      case 8:
        SSD_INT8_GENERAL(query + dim, ptrs[7] + dim, results[i]);
        /* FALLTHRU */
      case 7:
        SSD_INT8_GENERAL(query + dim, ptrs[6] + dim, results[i]);
        /* FALLTHRU */
      case 6:
        SSD_INT8_GENERAL(query + dim, ptrs[5] + dim, results[i]);
        /* FALLTHRU */
      case 5:
        SSD_INT8_GENERAL(query + dim, ptrs[4] + dim, results[i]);
        /* FALLTHRU */
      case 4:
        SSD_INT8_GENERAL(query + dim, ptrs[3] + dim, results[i]);
        /* FALLTHRU */
      case 3:
        SSD_INT8_GENERAL(query + dim, ptrs[2] + dim, results[i]);
        /* FALLTHRU */
      case 2:
        SSD_INT8_GENERAL(query + dim, ptrs[1] + dim, results[i]);
        /* FALLTHRU */
      case 1:
        SSD_INT8_GENERAL(query + dim, ptrs[0] + dim, results[i]);
    }
  }
}

#endif


}  // namespace zvec::ailego::DistanceBatch