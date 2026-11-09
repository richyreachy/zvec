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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <vector>
#include <turbo/distance/common/symqg_kernels.h>
#if RABITQ_SUPPORTED
#include <immintrin.h>
#include <rabitqlib/fastscan/encoder.hpp>
#include <rabitqlib/fastscan/fastscan.hpp>
#include <rabitqlib/fastscan/query.hpp>
#endif

namespace zvec::turbo {
// Convert node-centered squared-L2 coefficients to 1 - dot(query, neighbor).
// With g_cos = 1 - dot(query, center), the additive correction is
// (f_add_l2 + ||center||^2 - ||neighbor||^2) / 2. Keeping both norms also
// handles zero vectors, for which simply halving L2 would be incorrect.
// The caller supplies normalized coordinates, without any norm metadata.
template <class AddFactors, class ScaleFactors>
inline void ConvertSymQGCosineFactors(const float *center, const float *vectors,
                                      size_t count, size_t dimension,
                                      AddFactors f_add,
                                      ScaleFactors f_rescale) {
  const float center_norm =
      std::inner_product(center, center + dimension, center, 0.0f);
  for (size_t i = 0; i < count; ++i) {
    const float *vector = vectors + i * dimension;
    const float norm =
        std::inner_product(vector, vector + dimension, vector, 0.0f);
    f_add[i] = 0.5f * (f_add[i] + center_norm - norm);
    f_rescale[i] = 0.5f * f_rescale[i];
  }
}

#if RABITQ_SUPPORTED


// Query-side state for the SymphonyQG fastscan hot loop. Produces the same
// 8-bit LUT, delta and correction terms as rabitqlib's Lut/BatchQuery: the
// AVX512 kernel builds them in two fused passes over the rotated query (the
// 16 entries of each 4-dim codebook are the full subset-sum lattice of four
// query values), and the rabitqlib Lut path is the reference fallback.
class SymQGQuery {
 public:
  void reset(const float *rotated_query, size_t padded_dim) {
    g_add_ = 0;
    priority_bonus_ = 0;
    lut_storage_.resize((padded_dim << 2) + 63);
    auto *aligned_lut = reinterpret_cast<uint8_t *>(
        (reinterpret_cast<uintptr_t>(lut_storage_.data()) + 63) &
        ~uintptr_t{63});
    if (symqg_lut_build(rotated_query, padded_dim, aligned_lut, &delta_,
                        &sum_vl_, &k1xsumq_)) {
      lut_ = aligned_lut;
      return;
    }
    ref_lut_scratch_.resize(padded_dim * 4);
    ref_lut_.reset(rotated_query, padded_dim, ref_lut_scratch_.data());
    lut_ = ref_lut_.lut();
    delta_ = ref_lut_.delta();
    sum_vl_ = ref_lut_.sum_vl();
    const float sumq =
        std::accumulate(rotated_query, rotated_query + padded_dim, 0.0f);
    k1xsumq_ = sumq * -0.5f;
  }

  void reset_block(const float *query, size_t dimension, const float *book,
                   size_t block, size_t code_dim, bool structured = false) {
    g_add_ = priority_bonus_ = k1xsumq_ = 0;
    lut_storage_.resize(code_dim * 4 + 63);
    auto *out = reinterpret_cast<uint8_t *>(
        (reinterpret_cast<uintptr_t>(lut_storage_.data()) + 63) &
        ~uintptr_t{63});
    ref_lut_scratch_.resize(code_dim * 4);
    const bool built =
        structured
            ? symqg_h8_lut(query, dimension, block, code_dim,
                           ref_lut_scratch_.data(), out, &delta_, &sum_vl_)
            : symqg_block_lut(query, dimension, book, block, code_dim,
                              ref_lut_scratch_.data(), out, &delta_, &sum_vl_);
    if (!built) {
      sum_vl_ = 0;
      float range = 0;
      for (size_t group = 0; group < code_dim / 4; ++group) {
        const size_t offset = group * block;
        float lo = 0, hi = 0;
        for (size_t code = 0; code < 16; ++code) {
          float value = 0;
          for (size_t d = 0; d < block && offset + d < dimension; ++d)
            value += query[offset + d] * book[d * 16 + code];
          ref_lut_scratch_[group * 16 + code] = value;
          lo = std::min(lo, value);
          hi = std::max(hi, value);
        }
        range = std::max(range, hi - lo);
        sum_vl_ += lo;
        for (size_t code = 0; code < 16; ++code)
          ref_lut_scratch_[group * 16 + code] -= lo;
      }
      delta_ = range > 0 ? range / 255 : 1;
      for (size_t i = 0; i < code_dim * 4; ++i)
        out[i] = uint8_t(std::clamp(
            std::nearbyint(ref_lut_scratch_[i] / delta_), 0.f, 255.f));
    }
    lut_ = out;
  }

  [[nodiscard]] const uint8_t *lut() const {
    return lut_;
  }
  [[nodiscard]] float delta() const {
    return delta_;
  }
  [[nodiscard]] float sum_vl_lut() const {
    return sum_vl_;
  }
  [[nodiscard]] float k1xsumq() const {
    return k1xsumq_ + priority_bonus_;
  }
  // f_rescale is negative: a positive term gives optimistic navigation
  // scores. This heuristic is separate from certified terminal bounds.
  void set_priority_bonus(float value) {
    priority_bonus_ = value;
  }
  [[nodiscard]] float g_add() const {
    return g_add_;
  }
  void set_g_add(float dist) {
    g_add_ = dist;
  }

 private:
  std::vector<uint8_t> lut_storage_;
  rabitqlib::Lut<float> ref_lut_;
  std::vector<float> ref_lut_scratch_;
  const uint8_t *lut_ = nullptr;
  float delta_ = 0.0f;
  float sum_vl_ = 0.0f;
  float k1xsumq_ = 0.0f;
  float priority_bonus_ = 0.0f;
  float g_add_ = 0.0f;
};

// The pinned RaBitQ 0.7 dependency accumulates into int32_t, including the
// maximum supported dimension with four rotation views.
template <size_t Fixed>
__attribute__((target("avx512f,avx512bw,avx512dq"))) inline bool SymQGFixedScan(
    const uint8_t *codes, const uint8_t *lut, size_t, const void *add,
    const void *scale, float center_distance, float delta, float sum_vl,
    float k1xsumq, float *distances) {
  constexpr size_t padded_dim = Fixed;

  __m512i sums0 = _mm512_setzero_si512();
  __m512i sums1 = _mm512_setzero_si512();
  const __m512i nibble = _mm512_set1_epi8(15);
  // Specialize common padded dimensions so the compiler can unroll each
  // packed LUT scan. Keep sums in registers through widening and conversion.
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

  // These navigation estimates allow the caller's floating-point contraction
  // policy; they need not round identically to the generic scan kernel.
  const __m512 center = _mm512_set1_ps(center_distance);
  const __m512 width = _mm512_set1_ps(delta);
  const __m512 vl = _mm512_set1_ps(sum_vl);
  const __m512 sq = _mm512_set1_ps(k1xsumq);
  for (size_t i = 0; i < 32; i += 16) {
    __m512 estimate =
        _mm512_mul_ps(width, _mm512_cvtepi32_ps(i == 0 ? sums0 : sums1));
    estimate = _mm512_add_ps(estimate, vl);
    estimate = _mm512_add_ps(estimate, sq);
    estimate = _mm512_mul_ps(
        _mm512_loadu_ps(static_cast<const char *>(scale) + i * sizeof(float)),
        estimate);
    const __m512 base = _mm512_add_ps(
        _mm512_loadu_ps(static_cast<const char *>(add) + i * sizeof(float)),
        center);
    _mm512_storeu_ps(distances + i, _mm512_add_ps(base, estimate));
  }
  return true;
}

inline bool SymQGFixedScanDispatch(const uint8_t *codes, const uint8_t *lut,
                                   size_t dim, const void *add,
                                   const void *scale, float center, float delta,
                                   float sum, float bias, float *distances) {
  // Keep the portable fallback for unsupported CPUs and other dimensions.
  if (!__builtin_cpu_supports("avx512f") ||
      !__builtin_cpu_supports("avx512bw") ||
      !__builtin_cpu_supports("avx512dq"))
    return symqg_scan(codes, lut, dim, add, scale, center, delta, sum, bias,
                      distances);
  switch (dim) {
    case 64:
      return SymQGFixedScan<64>(codes, lut, dim, add, scale, center, delta, sum,
                                bias, distances);
    case 128:
      return SymQGFixedScan<128>(codes, lut, dim, add, scale, center, delta,
                                 sum, bias, distances);
    case 256:
      return SymQGFixedScan<256>(codes, lut, dim, add, scale, center, delta,
                                 sum, bias, distances);
    case 512:
      return SymQGFixedScan<512>(codes, lut, dim, add, scale, center, delta,
                                 sum, bias, distances);
    case 1024:
      return SymQGFixedScan<1024>(codes, lut, dim, add, scale, center, delta,
                                  sum, bias, distances);
    default:
      return symqg_scan(codes, lut, dim, add, scale, center, delta, sum, bias,
                        distances);
  }
}

inline void ScanSymQGBatchUnrolled(const char *codes, const SymQGQuery &query,
                                   size_t padded_dim, float *distances) {
  constexpr size_t batch_size = rabitqlib::fastscan::kBatchSize;
  rabitqlib::ConstQGBatchDataMap<float> batch(codes, padded_dim);
  // RaBitQ 0.7 stores factors as packed bytes. Load them directly through
  // the SIMD byte-addressed interface without creating float objects here.
  using Offsets = rabitqlib::detail::BatchCodeOffsets<float>;
  if (SymQGFixedScanDispatch(batch.bin_code(), query.lut(), padded_dim,
                             codes + Offsets::add(padded_dim),
                             codes + Offsets::rescale(padded_dim),
                             query.g_add(), query.delta(), query.sum_vl_lut(),
                             query.k1xsumq(), distances)) {
    return;
  }
  std::array<uint32_t, batch_size> sums{};
  // Match the installed FastScan API while retaining bounded slices for 0.1.
  using Accumulator = std::conditional_t<
      std::is_invocable_v<decltype(&rabitqlib::fastscan::accumulate),
                          const uint8_t *, const uint8_t *, int32_t *, size_t>,
      int32_t, uint16_t>;
  std::array<Accumulator, batch_size> partial;
  for (size_t dim = 0; dim < padded_dim; dim += 1024) {
    rabitqlib::fastscan::accumulate(batch.bin_code() + dim * 4,
                                    query.lut() + dim * 4, partial.data(),
                                    std::min(size_t{1024}, padded_dim - dim));
    for (size_t i = 0; i < batch_size; ++i) sums[i] += partial[i];
  }
  for (size_t i = 0; i < batch_size; ++i) {
    distances[i] =
        batch.f_add()[i] + query.g_add() +
        batch.f_rescale()[i] *
            (query.delta() * sums[i] + query.sum_vl_lut() + query.k1xsumq());
  }
}

inline void ScanSymQGBatch(const char *codes, const SymQGQuery &query,
                           size_t dimension, float *distances) {
  ScanSymQGBatchUnrolled(codes, query, dimension, distances);
}

#endif
}  // namespace zvec::turbo
