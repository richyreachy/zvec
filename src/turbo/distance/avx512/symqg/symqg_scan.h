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

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace zvec::turbo::detail {

// Keep one arithmetic implementation for specialized and generic scans.
// Fixed == 0 preserves support for all other valid dimensions.
// Fused selects approximate navigation scoring. The public kernel uses
// Fused == false with its existing floating-point contraction setting.
template <size_t Fixed, bool Fused>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512dq,fma")))
#endif
inline bool SymQGScan(
    const uint8_t *codes, const uint8_t *lut, size_t dimension, const void *add,
    const void *scale, float center_distance, float delta, float sum_vl,
    float k1xsumq, float *distances) {
  const size_t padded_dim = Fixed ? Fixed : dimension;

  __m512i sums0 = _mm512_setzero_si512();
  __m512i sums1 = _mm512_setzero_si512();
  const __m512i nibble = _mm512_set1_epi8(15);
  // Same packed layout and shuffle accumulation as RaBitQ's FastScan. Keep
  // sums in registers through widening and score conversion, avoiding the
  // generic path's temporary uint16/uint32 arrays and per-lane conversion.
  for (size_t dim = 0; dim < padded_dim; dim += 1024) {
    const size_t slice = (padded_dim - dim < 1024) ? padded_dim - dim : 1024;
    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
#pragma GCC unroll 4
    for (size_t offset = dim * 4; offset < (dim + slice) * 4; offset += 64) {
      const __m512i c = _mm512_loadu_si512(codes + offset);
      const __m512i table = _mm512_loadu_si512(lut + offset);
      const __m512i lo =
          _mm512_shuffle_epi8(table, _mm512_and_si512(c, nibble));
      const __m512i hi = _mm512_shuffle_epi8(
          table, _mm512_and_si512(_mm512_srli_epi16(c, 4), nibble));
      a0 = _mm512_add_epi16(a0, lo);
      a1 = _mm512_add_epi16(a1, _mm512_srli_epi16(lo, 8));
      a2 = _mm512_add_epi16(a2, hi);
      a3 = _mm512_add_epi16(a3, _mm512_srli_epi16(hi, 8));
    }
    a0 = _mm512_sub_epi16(a0, _mm512_slli_epi16(a1, 8));
    a2 = _mm512_sub_epi16(a2, _mm512_slli_epi16(a3, 8));
    const __m512i r0 = _mm512_add_epi16(_mm512_mask_blend_epi64(0xF0, a0, a1),
                                        _mm512_shuffle_i64x2(a0, a1, 0x4E));
    const __m512i r1 = _mm512_add_epi16(_mm512_mask_blend_epi64(0xF0, a2, a3),
                                        _mm512_shuffle_i64x2(a2, a3, 0x4E));
    const __m512i sum = _mm512_add_epi16(_mm512_shuffle_i64x2(r0, r1, 0x88),
                                         _mm512_shuffle_i64x2(r0, r1, 0xDD));
    sums0 = _mm512_add_epi32(
        sums0, _mm512_cvtepu16_epi32(_mm512_castsi512_si256(sum)));
    sums1 = _mm512_add_epi32(
        sums1, _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(sum, 1)));
  }

  const __m512 center = _mm512_set1_ps(center_distance);
  const __m512 width = _mm512_set1_ps(delta);
  const __m512 vl = _mm512_set1_ps(sum_vl);
  const __m512 sq = _mm512_set1_ps(k1xsumq);
  for (size_t i = 0; i < 32; i += 16) {
    const __m512 sum = _mm512_cvtepi32_ps(i == 0 ? sums0 : sums1);
    __m512 estimate;
    if constexpr (Fused)
      estimate = _mm512_fmadd_ps(width, sum, vl);
    else
      estimate = _mm512_add_ps(_mm512_mul_ps(width, sum), vl);
    estimate = _mm512_add_ps(estimate, sq);

    const __m512 base = _mm512_add_ps(
        _mm512_loadu_ps(static_cast<const char *>(add) + i * sizeof(float)),
        center);
    const __m512 factor = _mm512_loadu_ps(
        static_cast<const char *>(scale) + i * sizeof(float));
    if constexpr (Fused)
      _mm512_storeu_ps(distances + i, _mm512_fmadd_ps(factor, estimate, base));
    else
      _mm512_storeu_ps(distances + i,
                      _mm512_add_ps(base, _mm512_mul_ps(factor, estimate)));
  }
  return true;
}

}  // namespace zvec::turbo::detail
