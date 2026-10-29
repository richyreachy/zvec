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
#include <cstring>
#include <limits>
#include <vector>

namespace zvec::core {
// Integer center estimates for normalized cosine and squared L2. Wider query
// codes keep terminal refinement intervals tight without enlarging centers. The
// adapter checks dimensions <= 4096 and VNNI before any query executes.
struct IntegerCenter {
  struct Query {
    std::vector<int16_t> codes;
    float scale = 0, error = 0, residual_norm = 0, coded_norm = 0;
    int32_t code_sum = 0;
    double mean_dot = 0, norm = 0;
  };
  // Adjacent normal floats can be stepped using their ordered IEEE bits.
  // Retain libm for zero, subnormals and nonfinite values, including its
  // exceptional underflow/overflow behavior at the normal-range boundary.
  static float next_down(float value) {
    static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4);
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t magnitude = bits & 0x7fffffffU;
    if (magnitude > 0x00800000U && magnitude < 0x7f7fffffU) {
      bits += (bits >> 31) ? 1U : uint32_t(-1);
      std::memcpy(&value, &bits, sizeof(value));
      return value;
    }
    return std::nextafter(value, -std::numeric_limits<float>::infinity());
  }
  static float upper_norm(double squared) {
    return std::nextafter(float(std::sqrt(squared) + 1e-12),
                          std::numeric_limits<float>::infinity());
  }
  // Full-dimensional queries use 16-bit codes against 8-bit centers.
  __attribute__((target("avx512f,avx512bw,fma"))) static bool prepare(
      const float *q, const float *mean, size_t n, Query &out) {
    out.codes.resize(n);
    constexpr int maximum_code = 32767;
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
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(out.codes.data() + j),
                          _mm512_cvtsepi32_epi16(k));
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
      out.codes[j] = int16_t(code);
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
  static float distance(const Query &q, const uint8_t *c, size_t n,
                        float offset, float scale, double center_mean_dot,
                        double mean_norm2, float coded_norm,
                        double &reconstructed) {
    const int64_t product = dot16(c, q.codes.data(), n);
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
    const int64_t product = dot16(c, q.codes.data(), n);
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
    return std::max(0.f, next_down(float(lower)));
  }
  // AM-GM bounds the product of norms without a per-center square root.
  // Roundoff slack covers the products and addition in the upper bound.
  static double norm_product_bound(double query_norm, float center_norm2) {
    return (.5 * query_norm * query_norm + .5 * double(center_norm2)) *
           (1. + 4. * std::numeric_limits<double>::epsilon());
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
    const double norm_product = norm_product_bound(q.norm, original_norm2);
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
    return next_down(float(
        std::max(double(approximate) - quantization, triangle) - arithmetic));
  }
};
}  // namespace zvec::core
