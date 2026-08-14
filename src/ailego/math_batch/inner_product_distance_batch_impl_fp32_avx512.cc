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

#include <array>
#include <ailego/math/matrix_utility.i>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include "distance_batch_math.h"

namespace zvec::ailego::DistanceBatch {

#if defined(__AVX512F__)

template <size_t BatchSize>
static void compute_one_to_many_inner_product_avx512f_fp32(
    const float *query, const float **ptrs,
    std::array<const float *, BatchSize> &prefetch_ptrs, size_t dimensionality,
    float *results) {
  __m512 accumulators[BatchSize];
  for (size_t i = 0; i < BatchSize; ++i) {
    accumulators[i] = _mm512_setzero_ps();
  }

  size_t dim = 0;
  for (; dim + 16 <= dimensionality; dim += 16) {
    const __m512 query_values = _mm512_loadu_ps(query + dim);
    for (size_t i = 0; i < BatchSize; ++i) {
      const __m512 vector_values = _mm512_loadu_ps(ptrs[i] + dim);
      accumulators[i] =
          _mm512_fmadd_ps(query_values, vector_values, accumulators[i]);
    }
    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < BatchSize; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }
  }

  if (dim < dimensionality) {
    const auto remaining = static_cast<unsigned>(dimensionality - dim);
    const __mmask16 mask = static_cast<__mmask16>((1u << remaining) - 1u);
    const __m512 query_values = _mm512_maskz_loadu_ps(mask, query + dim);
    for (size_t i = 0; i < BatchSize; ++i) {
      const __m512 vector_values = _mm512_maskz_loadu_ps(mask, ptrs[i] + dim);
      accumulators[i] =
          _mm512_fmadd_ps(query_values, vector_values, accumulators[i]);
    }
  }

  for (size_t i = 0; i < BatchSize; ++i) {
    results[i] = HorizontalAdd_FP32_V512(accumulators[i]);
  }
}

void compute_one_to_many_inner_product_avx512f_fp32_1(
    const float *query, const float **ptrs,
    std::array<const float *, 1> &prefetch_ptrs, size_t dim, float *results) {
  compute_one_to_many_inner_product_avx512f_fp32<1>(query, ptrs, prefetch_ptrs,
                                                    dim, results);
}

void compute_one_to_many_inner_product_avx512f_fp32_12(
    const float *query, const float **ptrs,
    std::array<const float *, 12> &prefetch_ptrs, size_t dim, float *results) {
  compute_one_to_many_inner_product_avx512f_fp32<12>(query, ptrs, prefetch_ptrs,
                                                     dim, results);
}

void compute_one_to_many_inner_product_avx512f_fp32_8(
    const float *query, const float **ptrs,
    std::array<const float *, 8> &prefetch_ptrs, size_t dim, float *results) {
  compute_one_to_many_inner_product_avx512f_fp32<8>(query, ptrs, prefetch_ptrs,
                                                    dim, results);
}

// Sequential sweep over a packed block of vectors (stride between vectors ==
// dim), with a fixed lookahead prefetch to keep memory-level parallelism
// across short chained blocks.
void compute_contiguous_inner_product_avx512f_fp32(const float *block,
                                                   const float *query,
                                                   size_t num, size_t dim,
                                                   float *results) {
  // Lookahead distance chosen so prefetches issued while computing vector i
  // have retired from DRAM by the time vector i+PF is consumed.
  constexpr size_t PF = 6;
  const float *vec = block;
  for (size_t i = 0; i < num; ++i, vec += dim) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    const float *ahead = (i + PF < num) ? vec + PF * dim : nullptr;
    size_t d = 0;
    for (; d + 32 <= dim; d += 32) {
      acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(query + d),
                             _mm512_loadu_ps(vec + d), acc0);
      acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(query + d + 16),
                             _mm512_loadu_ps(vec + d + 16), acc1);
      if (ahead) {
        _mm_prefetch(reinterpret_cast<const char *>(ahead + d), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(ahead + d + 16),
                     _MM_HINT_T0);
      }
    }
    if (d + 16 <= dim) {
      acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(query + d),
                             _mm512_loadu_ps(vec + d), acc0);
      d += 16;
    }
    if (d < dim) {
      const auto remaining = static_cast<unsigned>(dim - d);
      const __mmask16 mask = static_cast<__mmask16>((1u << remaining) - 1u);
      acc1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, query + d),
                             _mm512_maskz_loadu_ps(mask, vec + d), acc1);
    }
    results[i] = HorizontalAdd_FP32_V512(_mm512_add_ps(acc0, acc1));
  }
}

#endif

}  // namespace zvec::ailego::DistanceBatch
