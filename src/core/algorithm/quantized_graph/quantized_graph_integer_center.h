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
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace zvec::core {
// Integer center estimates for normalized cosine and squared L2. L2 uses
// wider query codes to keep its terminal refinement intervals tight. The
// adapter checks dimensions <= 4096 and VNNI before any query executes.
struct IntegerCenter {
  struct Query {
    std::vector<int8_t> codes;
    std::vector<int16_t> codes16;
    float scale = 0, error = 0, residual_norm = 0, coded_norm = 0;
    int32_t code_sum = 0;
    double mean_dot = 0, norm = 0;
  };
  static float upper_norm(double squared) {
    return std::nextafter(float(std::sqrt(squared) + 1e-12),
                          std::numeric_limits<float>::infinity());
  }
  // Keep cosine and router queries at 8 bits; L2 requests 16-bit codes.
  template <bool Wide = false>
  __attribute__((target("avx512f,avx512bw,fma"))) static bool prepare(
      const float *q, const float *mean, size_t n, Query &out) {
    if constexpr (Wide)
      out.codes16.resize(n);
    else
      out.codes.resize(n);
    constexpr int maximum_code = Wide ? 32767 : 127;
    const size_t blocks = n / 16;
    __m512 maximum_vector = _mm512_setzero_ps();
    for (size_t block = 0; block < blocks; ++block) {
      const size_t j = block * 16;
      const __m512 r =
          _mm512_sub_ps(_mm512_loadu_ps(q + j), _mm512_loadu_ps(mean + j));
      const __m512 magnitude = _mm512_castsi512_ps(_mm512_andnot_si512(
          _mm512_set1_epi32(int32_t(0x80000000u)), _mm512_castps_si512(r)));
      maximum_vector = _mm512_max_ps(maximum_vector, magnitude);
    }
    float maximum = _mm512_reduce_max_ps(maximum_vector);
    for (size_t j = blocks * 16; j < n; ++j)
      maximum = std::max(maximum, std::abs(q[j] - mean[j]));
    out.scale = maximum / float(maximum_code);
    if (!std::isfinite(out.scale)) return false;
    const __m512 inverse =
        _mm512_set1_ps(out.scale > 0 ? 1.f / out.scale : 0.f);
    const __m512d scale = _mm512_set1_pd(out.scale);
    __m512d norm_acc = _mm512_setzero_pd(), mean_acc = _mm512_setzero_pd(),
            residual_acc = _mm512_setzero_pd(), coded_acc = _mm512_setzero_pd(),
            error_acc = _mm512_setzero_pd();
    __m512i sum_acc = _mm512_setzero_si512();
    for (size_t block = 0; block < blocks; ++block) {
      const size_t j = block * 16;
      const __m512 r32 =
          _mm512_sub_ps(_mm512_loadu_ps(q + j), _mm512_loadu_ps(mean + j));
      __m512i k = _mm512_cvt_roundps_epi32(
          _mm512_mul_ps(r32, inverse),
          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      k = _mm512_max_epi32(
          _mm512_set1_epi32(-maximum_code),
          _mm512_min_epi32(_mm512_set1_epi32(maximum_code), k));
      if constexpr (Wide)
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(out.codes16.data() + j),
                            _mm512_cvtsepi32_epi16(k));
      else
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out.codes.data() + j),
                         _mm512_cvtsepi32_epi8(k));
      sum_acc = _mm512_add_epi32(sum_acc, k);
      alignas(64) int32_t unpacked[16];
      _mm512_store_si512(unpacked, k);
      for (size_t half = 0; half < 2; ++half) {
        const size_t h = j + half * 8;
        const __m512d q64 = _mm512_cvtps_pd(_mm256_loadu_ps(q + h));
        const __m512d m64 = _mm512_cvtps_pd(_mm256_loadu_ps(mean + h));
        const __m512d r = _mm512_sub_pd(q64, m64);
        const __m512d u = _mm512_mul_pd(
            scale,
            _mm512_cvtepi32_pd(_mm256_load_si256(
                reinterpret_cast<const __m256i *>(unpacked + half * 8))));
        const __m512d error = _mm512_sub_pd(r, u);
        norm_acc = _mm512_fmadd_pd(q64, q64, norm_acc);
        mean_acc = _mm512_fmadd_pd(q64, m64, mean_acc);
        residual_acc = _mm512_fmadd_pd(r, r, residual_acc);
        coded_acc = _mm512_fmadd_pd(u, u, coded_acc);
        error_acc = _mm512_fmadd_pd(error, error, error_acc);
      }
    }
    double norm = _mm512_reduce_add_pd(norm_acc),
           residual = _mm512_reduce_add_pd(residual_acc),
           coded = _mm512_reduce_add_pd(coded_acc),
           error = _mm512_reduce_add_pd(error_acc);
    out.mean_dot = _mm512_reduce_add_pd(mean_acc);
    out.code_sum = _mm512_reduce_add_epi32(sum_acc);
    for (size_t j = blocks * 16; j < n; ++j) {
      const double r = double(q[j]) - mean[j];
      const double k = out.scale > 0 ? std::round(r / out.scale) : 0;
      const int code =
          int(std::clamp(k, -double(maximum_code), double(maximum_code)));
      if constexpr (Wide)
        out.codes16[j] = int16_t(code);
      else
        out.codes[j] = int8_t(code);
      out.code_sum += code;
      const double u = double(out.scale) * code;
      error += (r - u) * (r - u);
      residual += r * r;
      coded += u * u;
      norm += double(q[j]) * q[j];
      out.mean_dot += double(q[j]) * mean[j];
    }
    out.error = upper_norm(error);
    out.residual_norm = upper_norm(residual);
    out.coded_norm = upper_norm(coded);
    out.norm = std::sqrt(norm) + 1e-12;
    return std::isfinite(out.norm) && std::isfinite(out.error) &&
           std::isfinite(out.mean_dot);
  }
  static void annotate(const float *c, const float *mean, const uint8_t *codes,
                       size_t n, float offset, float scale, float &error,
                       float &residual_norm, float &coded_norm,
                       double &mean_dot) {
    double residual = 0, coded = 0, err = 0;
    mean_dot = 0;
    for (size_t j = 0; j < n; ++j) {
      const double s = double(c[j]) - mean[j];
      const double v = double(scale) * codes[j] + offset;
      residual += s * s;
      coded += v * v;
      err += (s - v) * (s - v);
      mean_dot += double(mean[j]) * c[j];
    }
    error = upper_norm(err);
    residual_norm = upper_norm(residual);
    coded_norm = upper_norm(coded);
  }
  __attribute__((target("avx512f,avx512bw,avx512vnni"))) static int32_t dot(
      const uint8_t *c, const int8_t *q, size_t n) {
    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
    const size_t blocks = n / 64;
    size_t b = 0;
    for (; b + 4 <= blocks; b += 4) {
      a0 = _mm512_dpbusd_epi32(a0, _mm512_loadu_si512(c + 64 * b),
                               _mm512_loadu_si512(q + 64 * b));
      a1 = _mm512_dpbusd_epi32(a1, _mm512_loadu_si512(c + 64 * (b + 1)),
                               _mm512_loadu_si512(q + 64 * (b + 1)));
      a2 = _mm512_dpbusd_epi32(a2, _mm512_loadu_si512(c + 64 * (b + 2)),
                               _mm512_loadu_si512(q + 64 * (b + 2)));
      a3 = _mm512_dpbusd_epi32(a3, _mm512_loadu_si512(c + 64 * (b + 3)),
                               _mm512_loadu_si512(q + 64 * (b + 3)));
    }
    __m512i acc =
        _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3));
    for (; b < blocks; ++b)
      acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512(c + 64 * b),
                                _mm512_loadu_si512(q + 64 * b));
    int32_t sum = _mm512_reduce_add_epi32(acc);
    for (size_t j = blocks * 64; j < n; ++j) sum += int32_t(c[j]) * q[j];
    return sum;
  }
  static float distance(const Query &q, const uint8_t *c, size_t n,
                        float offset, float scale, double center_mean_dot,
                        double mean_norm2, float coded_norm,
                        double &reconstructed) {
    const int32_t product = dot(c, q.codes.data(), n);
    const double residual_dot = double(q.scale) * (double(scale) * product +
                                                   double(offset) * q.code_sum);
    reconstructed = double(q.coded_norm) * q.coded_norm +
                    double(coded_norm) * coded_norm - 2. * residual_dot;
    return float(1. -
                 (q.mean_dot + center_mean_dot - mean_norm2 + residual_dot));
  }
  // At n <= 4096, each signed 32-bit lane is bounded by
  // 256 * 255 * 32768 < INT32_MAX. Widen before the final horizontal sum.
  __attribute__((target("avx512f,avx512bw,avx512vnni"))) static int64_t dot16(
      const uint8_t *c, const int16_t *q, size_t n) {
    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 128 <= n; i += 128) {
      a0 = _mm512_dpwssd_epi32(
          a0,
          _mm512_cvtepu8_epi16(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + i + 0))),
          _mm512_loadu_si512(q + i + 0));
      a1 = _mm512_dpwssd_epi32(
          a1,
          _mm512_cvtepu8_epi16(_mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(c + i + 32))),
          _mm512_loadu_si512(q + i + 32));
      a2 = _mm512_dpwssd_epi32(
          a2,
          _mm512_cvtepu8_epi16(_mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(c + i + 64))),
          _mm512_loadu_si512(q + i + 64));
      a3 = _mm512_dpwssd_epi32(
          a3,
          _mm512_cvtepu8_epi16(_mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(c + i + 96))),
          _mm512_loadu_si512(q + i + 96));
    }
    __m512i acc =
        _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3));
    for (; i + 32 <= n; i += 32)
      acc = _mm512_dpwssd_epi32(acc,
                                _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                                    reinterpret_cast<const __m256i *>(c + i))),
                                _mm512_loadu_si512(q + i));
    const __m512i wide = _mm512_add_epi64(
        _mm512_cvtepi32_epi64(_mm512_castsi512_si256(acc)),
        _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(acc, 1)));
    int64_t sum = _mm512_reduce_add_epi64(wide);
    for (; i < n; ++i) sum += int64_t(c[i]) * q[i];
    return sum;
  }
  // Original norms give the navigation estimate. The reconstructed-vector
  // distance is kept separately for the terminal triangle-inequality bound.
  static float l2_distance(const Query &q, const uint8_t *c, size_t n,
                           float offset, float scale, float original_norm2,
                           float coded_norm, double &reconstructed) {
    const int64_t product = dot16(c, q.codes16.data(), n);
    const double inner = double(q.scale) * (double(scale) * product +
                                            double(offset) * q.code_sum);
    reconstructed = double(q.coded_norm) * q.coded_norm +
                    double(coded_norm) * coded_norm - 2. * inner;
    return float(q.norm * q.norm + original_norm2 - 2. * inner);
  }
  // Conditional bound under finite, normal-range arithmetic. First remove
  // rounding slack from the reconstructed squared distance, then subtract
  // both reconstruction-error radii. The final guard covers FP32 L2 scoring.
  static float l2_lower_bound(double reconstructed, const Query &q, float error,
                              float coded_norm, size_t n) {
    const double norm_sum = double(q.coded_norm) + coded_norm;
    const double arithmetic =
        (8. * std::numeric_limits<float>::epsilon() +
         64. * n * std::numeric_limits<double>::epsilon()) *
            norm_sum * norm_sum +
        1e-10 * (1. + norm_sum);
    const double radius =
        std::max(0., std::sqrt(std::max(0., reconstructed - arithmetic)) -
                         q.error - error);
    const double a = (n + 4.) * std::numeric_limits<float>::epsilon();
    const double lower = radius * radius * (1. - a / (1. - a)) -
                         4. * (n + 4.) * std::numeric_limits<float>::min();
    return std::max(
        0.f,
        std::nextafter(float(lower), -std::numeric_limits<float>::infinity()));
  }
  static float lower_bound(float approximate, const Query &q, float error,
                           float residual_norm, float coded_norm,
                           float original_norm2, double center_mean_dot,
                           double mean_norm2, size_t n, double reconstructed) {
    const double quantization = std::min(
        double(q.error) * residual_norm + double(q.coded_norm) * error,
        double(error) * q.residual_norm + double(coded_norm) * q.error);
    const double a =
        (2. * n + 16) * (.5 * std::numeric_limits<float>::epsilon());
    const double norm_product = q.norm * std::sqrt(double(original_norm2));
    const double arithmetic =
        a / (1 - a) * norm_product +
        64. * (n + 16) * std::numeric_limits<double>::epsilon() *
            (1. + norm_product + std::abs(q.mean_dot) +
             std::abs(center_mean_dot) + std::abs(mean_norm2) +
             double(q.coded_norm) * coded_norm) +
        8. * std::numeric_limits<float>::epsilon() *
            (1. + std::abs(double(approximate)));
    // q and c share the same shift, so their residual difference is q-c.
    // 1-q.c = 1-(||q||^2+||c||^2)/2 + ||q-c||^2/2; upper original norms
    // and the L2 lower bound therefore give another cosine lower bound.
    const double triangle =
        1. - .5 * (q.norm * q.norm + original_norm2) +
        .5 * l2_lower_bound(reconstructed, q, error, coded_norm, n);
    return std::nextafter(
        float(std::max(double(approximate) - quantization, triangle) -
              arithmetic),
        -std::numeric_limits<float>::infinity());
  }
};
}  // namespace zvec::core
