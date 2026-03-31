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

#if defined(__AVX512F__)
#include <immintrin.h>
#include <array>
#include <cstdint>

//! Calculate Fused-Multiply-Add (AVX512)
#define FMA_FP32_AVX512(zmm_m, zmm_q, zmm_sum) \
  zmm_sum = _mm512_fmadd_ps(zmm_m, zmm_q, zmm_sum);


static inline float HorizontalAdd_FP32_V256(__m256 v) {
  __m256 x1 = _mm256_hadd_ps(v, v);
  __m256 x2 = _mm256_hadd_ps(x1, x1);
  __m128 x3 = _mm256_extractf128_ps(x2, 1);
  __m128 x4 = _mm_add_ss(_mm256_castps256_ps128(x2), x3);
  return _mm_cvtss_f32(x4);
}

static inline float HorizontalAdd_FP32_V512(__m512 v) {
  __m256 low = _mm512_castps512_ps256(v);
  __m256 high =
      _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
  return HorizontalAdd_FP32_V256(_mm256_add_ps(low, high));
}

#endif  // __AVX512F__