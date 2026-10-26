#pragma once
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
namespace zvec::core {
struct HalfCenter {
  __attribute__((target("f16c"))) static bool encode(const float *x, size_t n,
                                                     uint16_t *out,
                                                     float &error,
                                                     float &norm2) {
    double err = 0, norm = 0;
    for (size_t i = 0; i < n; ++i) {
      out[i] = _cvtss_sh(x[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      const float y = _cvtsh_ss(out[i]);
      if (!std::isfinite(y)) return false;
      double e = double(y) - x[i];
      err += e * e;
      norm += double(x[i]) * x[i];
    }
    error = std::nextafter(float(std::sqrt(err)),
                           std::numeric_limits<float>::infinity());
    norm2 = std::nextafter(float(norm), std::numeric_limits<float>::infinity());
    return true;
  }
  __attribute__((target("f16c"))) static void decode(const uint16_t *h,
                                                     size_t n, float *out) {
    for (size_t i = 0; i < n; ++i) out[i] = _cvtsh_ss(h[i]);
  }
  __attribute__((target("avx512f,f16c,fma"))) static float distance(
      const float *q, const uint16_t *h, size_t n) {
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
      __m256i raw =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(h + i));
      __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(q + i), _mm512_cvtph_ps(raw));
      acc = _mm512_fmadd_ps(diff, diff, acc);
    }
    float result = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) {
      float d = q[i] - _cvtsh_ss(h[i]);
      result += d * d;
    }
    return result;
  }
  __attribute__((target("avx512f,f16c,fma"))) static float cosine_distance(
      const float *q, const uint16_t *h, size_t n) {
    __m512 a[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(),
                   _mm512_setzero_ps(), _mm512_setzero_ps()};
    size_t i = 0;
    for (; i + 64 <= n; i += 64)
      for (size_t lane = 0; lane < 4; ++lane) {
        const size_t j = i + 16 * lane;
        const __m256i raw =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(h + j));
        a[lane] = _mm512_fmadd_ps(_mm512_loadu_ps(q + j), _mm512_cvtph_ps(raw),
                                  a[lane]);
      }
    float dot = _mm512_reduce_add_ps(
        _mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3])));
    for (; i < n; ++i) dot += q[i] * _cvtsh_ss(h[i]);
    return 1.f - dot;
  }
  // Triangle-inequality interval with conservative floating arithmetic padding.
  // Research implementation assumes finite normal-range data and no overflow.
  static float lower_bound(float approximate, float error, float center_norm2,
                           float query_norm2, size_t n) {
#if defined(QG_TIGHT_CENTER_BOUND)
    (void)center_norm2;
    (void)query_norm2;
    // Direct sum-of-squared-differences kernels: gamma_(n+4) covers
    // rounded differences, products and a worst-case serial reduction.
    // Positive IEEE-754 doubles: one-bit steps provide outward rounding
    // without a libm call at each arithmetic operation.
    auto down = [](double x) {
      if (!(x > 0)) return 0.;
      uint64_t b;
      std::memcpy(&b, &x, 8);
      --b;
      std::memcpy(&x, &b, 8);
      return x;
    };
    auto up = [](double x) {
      uint64_t b;
      std::memcpy(&b, &x, 8);
      ++b;
      std::memcpy(&x, &b, 8);
      return x;
    };
    const double a = (n + 4) * (.5 * std::numeric_limits<float>::epsilon());
    const double gamma = up(a / (1. - a));
    const double tiny =
        4. * (n + 4) * double(std::numeric_limits<float>::min());
    const double s_lower =
        down(down(double(approximate) - tiny) / up(1. + gamma));
    const double radius = down(down(std::sqrt(s_lower)) - error);
    const double lower =
        down(down(down(1. - gamma) * down(radius * radius)) - tiny);
    return std::max(
        0.f,
        std::nextafter(float(lower), -std::numeric_limits<float>::infinity()));
#else
    const double a = 8. * (n + 16) * std::numeric_limits<float>::epsilon();
    const double gamma = a / (1. - a);
    const double arithmetic = gamma * (double(query_norm2) + 2. * center_norm2 +
                                       2. * error * error + 1.);
    const double radius = std::max(
        0., std::sqrt(std::max(0., double(approximate) - arithmetic)) - error);
    const double lower = std::max(0., radius * radius - arithmetic);
    return std::max(
        0.f,
        std::nextafter(float(lower), -std::numeric_limits<float>::infinity()));
#endif
  }
};
}  // namespace zvec::core
namespace zvec::core {
struct SQ8Center {
  __attribute__((target("avx512f,avx512bw,fma"))) static float cosine_distance(
      const float *q, const uint8_t *h, size_t n, float offset, float scale) {
    const __m512 lo = _mm512_set1_ps(offset), step = _mm512_set1_ps(scale);
    __m512 a[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(),
                   _mm512_setzero_ps(), _mm512_setzero_ps()};
    size_t i = 0;
    for (; i + 64 <= n; i += 64)
      for (size_t lane = 0; lane < 4; ++lane) {
        const size_t j = i + 16 * lane;
        const __m128i raw =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(h + j));
        const __m512 v = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(raw)), step, lo);
        a[lane] = _mm512_fmadd_ps(_mm512_loadu_ps(q + j), v, a[lane]);
      }
    float dot = _mm512_reduce_add_ps(
        _mm512_add_ps(_mm512_add_ps(a[0], a[1]), _mm512_add_ps(a[2], a[3])));
    for (; i < n; ++i) dot += q[i] * std::fma(float(h[i]), scale, offset);
    return 1.f - dot;
  }
  // Cauchy-Schwarz plus deliberately loose bounds for both FP32 dot kernels.
  // Finite, non-overflowing inputs; normalized cosine coordinates in this task.
  static float cosine_lower_bound(float approximate, float error, float norm2,
                                  float qnorm2, size_t n) {
    const double qnorm = std::sqrt(double(qnorm2)),
                 norm = std::sqrt(double(norm2));
    const double eta = 8. * (n + 16) * std::numeric_limits<float>::epsilon(),
                 gamma = eta / (1. - eta);
    const double arithmetic = gamma * qnorm * (2. * norm + error) +
                              8. * std::numeric_limits<float>::epsilon() *
                                  (1. + std::abs(double(approximate)));
    return std::nextafter(
        float(double(approximate) - qnorm * error - arithmetic),
        -std::numeric_limits<float>::infinity());
  }
  static bool encode(const float *x, size_t n, uint8_t *out, float &error,
                     float &norm2, float &offset, float &scale) {
    offset = *std::min_element(x, x + n);
    float maximum = *std::max_element(x, x + n);
    scale = (maximum - offset) / 255.f;
    double err = 0, norm = 0;
    if (!std::isfinite(offset) || !std::isfinite(scale)) return false;
    for (size_t i = 0; i < n; ++i) {
      const float code =
          scale > 0 ? std::floor((x[i] - offset) / scale + .5f) : 0;
      out[i] = uint8_t(std::max(0.f, std::min(255.f, code)));
      const float y = std::fma(float(out[i]), scale, offset);
      double e = double(y) - x[i];
      err += e * e;
      norm += double(x[i]) * x[i];
    }
    error = std::nextafter(float(std::sqrt(err)),
                           std::numeric_limits<float>::infinity());
    norm2 = std::nextafter(float(norm), std::numeric_limits<float>::infinity());
    return true;
  }
  static void decode(const uint8_t *h, size_t n, float *out, float offset,
                     float scale) {
    for (size_t i = 0; i < n; ++i)
      out[i] = std::fma(float(h[i]), scale, offset);
  }
  __attribute__((target("avx512f,avx512bw,fma"))) static float distance(
      const float *q, const uint8_t *h, size_t n, float offset, float scale) {
    __m512 acc = _mm512_setzero_ps(), lo = _mm512_set1_ps(offset),
           step = _mm512_set1_ps(scale);
    size_t i = 0;
#if defined(QG_CENTER_UNROLL)
    __m512 accum[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(),
                       _mm512_setzero_ps(), _mm512_setzero_ps()};
    for (; i + 64 <= n; i += 64) {
      for (size_t lane = 0; lane < 4; ++lane) {
        const size_t j = i + lane * 16;
        __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(h + j));
        __m512 v = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(raw)), step, lo);
        __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(q + j), v);
        accum[lane] = _mm512_fmadd_ps(diff, diff, accum[lane]);
      }
    }
    acc = _mm512_add_ps(_mm512_add_ps(accum[0], accum[1]),
                        _mm512_add_ps(accum[2], accum[3]));
#endif
    for (; i + 16 <= n; i += 16) {
      __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(h + i));
      __m512 v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(raw)),
                                 step, lo);
      __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(q + i), v);
      acc = _mm512_fmadd_ps(diff, diff, acc);
    }
    float result = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) {
      float d = q[i] - std::fma(float(h[i]), scale, offset);
      result += d * d;
    }
    return result;
  }
};
}  // namespace zvec::core
