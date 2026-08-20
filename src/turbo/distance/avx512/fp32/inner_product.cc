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

#include "avx512/fp32/inner_product.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#include <array>
#include <zvec/ailego/internal/platform.h>
#endif

namespace zvec::turbo::avx512 {

#if defined(__AVX512F__)
namespace {

float dot_product(const float *a, const float *b, size_t dim) {
  __m512 accumulator = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m512 lhs = _mm512_loadu_ps(a + i);
    const __m512 rhs = _mm512_loadu_ps(b + i);
    accumulator = _mm512_add_ps(accumulator, _mm512_mul_ps(lhs, rhs));
  }

  float sum = _mm512_reduce_add_ps(accumulator);
  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

// One-to-many inner product kernel: the query block is loaded once per
// iteration and reused across all `dp_batch` vectors; the tail is handled
// with a masked load so no scalar remainder loop is needed.
template <size_t dp_batch>
void inner_product_batch_impl(
    const float *query, const float *const *ptrs,
    std::array<const float *, dp_batch> &prefetch_ptrs, size_t dimensionality,
    float *results) {
  __m512 accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm512_setzero_ps();
  }

  size_t dim = 0;
  for (; dim + 16 <= dimensionality; dim += 16) {
    const __m512 q = _mm512_loadu_ps(query + dim);
    __m512 data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm512_loadu_ps(ptrs[i] + dim);
    }

    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }

    for (size_t i = 0; i < dp_batch; ++i) {
      accs[i] = _mm512_fmadd_ps(q, data_regs[i], accs[i]);
    }
  }

  if (dim < dimensionality) {
    const __mmask16 mask =
        static_cast<__mmask16>((1u << (dimensionality - dim)) - 1);
    const __m512 q = _mm512_maskz_loadu_ps(mask, query + dim);
    for (size_t i = 0; i < dp_batch; ++i) {
      const __m512 data = _mm512_maskz_loadu_ps(mask, ptrs[i] + dim);
      accs[i] = _mm512_mask3_fmadd_ps(q, data, accs[i], mask);
    }
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = -_mm512_reduce_add_ps(accs[i]);
  }
}

// Dispatch batched inner product over all `n` vectors with prefetching.
// Chunk twelve records per block (independent accumulators saturate FMA
// ports and memory-level parallelism on flat scans), then an eight-wide
// tail block, then singles.
template <size_t BatchSize, size_t PrefetchStep>
size_t inner_product_batch_chunked(const void *const *vectors,
                                   const float *query, size_t n, size_t dim,
                                   float *distances) {
  size_t i = 0;
  for (; i + BatchSize <= n; i += BatchSize) {
    std::array<const float *, BatchSize> prefetch_ptrs;
    for (size_t j = 0; j < BatchSize; ++j) {
      if (i + j + BatchSize * PrefetchStep < n) {
        prefetch_ptrs[j] = static_cast<const float *>(
            vectors[i + j + BatchSize * PrefetchStep]);
      } else {
        prefetch_ptrs[j] = nullptr;
      }
    }
    inner_product_batch_impl<BatchSize>(
        query, reinterpret_cast<const float *const *>(&vectors[i]),
        prefetch_ptrs, dim, distances + i);
  }
  return i;
}

void inner_product_batch(const void *const *vectors, const void *query,
                         size_t n, size_t dim, float *distances) {
  const float *typed_query = static_cast<const float *>(query);
  size_t i = inner_product_batch_chunked<12, 2>(vectors, typed_query, n, dim,
                                                distances);
  i += inner_product_batch_chunked<8, 2>(&vectors[i], typed_query, n - i, dim,
                                         distances + i);
  for (; i < n; i++) {
    std::array<const float *, 1> prefetch_ptrs{nullptr};
    inner_product_batch_impl<1>(
        typed_query, reinterpret_cast<const float *const *>(&vectors[i]),
        prefetch_ptrs, dim, distances + i);
  }
}

// Sequential sweep over a packed block of vectors (stride between vectors ==
// dim), with a fixed lookahead prefetch to keep memory-level parallelism
// across the linear scan.
void inner_product_contiguous(const float *block, const float *query, size_t n,
                              size_t dim, float *distances) {
  // Lookahead distance chosen so prefetches issued while computing vector i
  // have retired from DRAM by the time vector i+PF is consumed. The lookahead
  // is unconditional: flat scans call this kernel on consecutive chunks of
  // one packed segment, so prefetching past this chunk warms the next one,
  // and PREFETCH never faults even on unmapped addresses at the segment end.
  constexpr size_t PF = 6;
  const float *vec = block;
  for (size_t i = 0; i < n; ++i, vec += dim) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    const float *ahead = vec + PF * dim;
    size_t d = 0;
    for (; d + 32 <= dim; d += 32) {
      acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(query + d),
                             _mm512_loadu_ps(vec + d), acc0);
      acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(query + d + 16),
                             _mm512_loadu_ps(vec + d + 16), acc1);
      ailego_prefetch(ahead + d);
      ailego_prefetch(ahead + d + 16);
    }
    if (d + 16 <= dim) {
      acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(query + d),
                             _mm512_loadu_ps(vec + d), acc0);
      d += 16;
    }
    if (d < dim) {
      const __mmask16 mask = static_cast<__mmask16>((1u << (dim - d)) - 1);
      acc1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, query + d),
                             _mm512_maskz_loadu_ps(mask, vec + d), acc1);
    }
    distances[i] = -_mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  }
}

}  // namespace
#endif

void inner_product_fp32_distance_avx512(const void *a, const void *b,
                                        size_t dim, float *distance) {
#if defined(__AVX512F__)
  *distance = -dot_product(static_cast<const float *>(a),
                           static_cast<const float *>(b), dim);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void inner_product_fp32_batch_distance_avx512(const void *const *vectors,
                                              const void *query, size_t n,
                                              size_t dim, float *distances) {
#if defined(__AVX512F__)
  inner_product_batch(vectors, query, n, dim, distances);
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

void inner_product_fp32_contiguous_batch_distance_avx512(const void *block,
                                                         const void *query,
                                                         size_t n, size_t dim,
                                                         float *distances) {
#if defined(__AVX512F__)
  inner_product_contiguous(static_cast<const float *>(block),
                           static_cast<const float *>(query), n, dim,
                           distances);
#else
  (void)block;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

}  // namespace zvec::turbo::avx512
