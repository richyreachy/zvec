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

#if defined(__AVX2__)

inline float sum4(__m128 v) {
  v = _mm_add_ps(v, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), 8)));
  return _mm_cvtss_f32(v) + _mm_cvtss_f32(_mm_shuffle_ps(v, v, 1));
}

inline __m128 sum_top_bottom_avx(__m256 v) {
  const __m128 high = _mm256_extractf128_ps(v, 1);
  const __m128 low = _mm256_castps256_ps128(v);
  return _mm_add_ps(high, low);
}

#endif

#if defined(__AVX512F__)

#include <ailego/math/matrix_utility.i>

namespace zvec::ailego::DistanceBatch {

//! Accumulation steps for the contiguous fp32 sweep below. Each step folds
//! one 16-lane strip of the query/vector pair into the accumulator.
struct InnerProductStepFp32Avx512 {
  static inline __m512 Accumulate(__m512 acc, __m512 q, __m512 v) {
    return _mm512_fmadd_ps(q, v, acc);
  }
};

struct SquaredEuclideanStepFp32Avx512 {
  static inline __m512 Accumulate(__m512 acc, __m512 q, __m512 v) {
    const __m512 diff = _mm512_sub_ps(q, v);
    return _mm512_fmadd_ps(diff, diff, acc);
  }
};

// Sequential sweep over a packed block of vectors (stride between vectors ==
// dim), with a fixed lookahead prefetch to keep memory-level parallelism
// across short chained blocks. The metric is factored out as StepOp so every
// fp32 metric shares the same loop structure.
template <typename StepOp>
static inline void compute_contiguous_fp32_avx512f(const float *block,
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
      acc0 = StepOp::Accumulate(acc0, _mm512_loadu_ps(query + d),
                                _mm512_loadu_ps(vec + d));
      acc1 = StepOp::Accumulate(acc1, _mm512_loadu_ps(query + d + 16),
                                _mm512_loadu_ps(vec + d + 16));
      if (ahead) {
        _mm_prefetch(reinterpret_cast<const char *>(ahead + d), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(ahead + d + 16),
                     _MM_HINT_T0);
      }
    }
    if (d + 16 <= dim) {
      acc0 = StepOp::Accumulate(acc0, _mm512_loadu_ps(query + d),
                                _mm512_loadu_ps(vec + d));
      d += 16;
    }
    if (d < dim) {
      const auto remaining = static_cast<unsigned>(dim - d);
      const __mmask16 mask = static_cast<__mmask16>((1u << remaining) - 1u);
      acc1 = StepOp::Accumulate(acc1, _mm512_maskz_loadu_ps(mask, query + d),
                                _mm512_maskz_loadu_ps(mask, vec + d));
    }
    results[i] = HorizontalAdd_FP32_V512(_mm512_add_ps(acc0, acc1));
  }
}

}  // namespace zvec::ailego::DistanceBatch

#endif
