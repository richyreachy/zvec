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

// Compute raw integer inner products for a batch of int8 vectors against a
// single query. Uses AVX512-VNNI dpbusd instruction.
// `query` is treated as uint8 (preprocessed), `vectors[i]` as int8.
template <size_t batch_size>
__attribute__((always_inline)) void inner_product_int8_batch_avx2_impl(
    const void *query, const void *const *vectors,
    const std::array<const void *, batch_size> &prefetch_ptrs,
    size_t dimensionality, float *distances) {}

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
