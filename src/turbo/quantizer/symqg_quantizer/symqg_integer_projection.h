// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif
namespace zvec::turbo::research {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f,avx512bw"))) inline void projection_quantize_u8(
    const float *query, size_t columns, uint8_t *output, float &delta,
    float &minimum) {
  __m512 lo = _mm512_loadu_ps(query), hi = lo;
  for (size_t j = 16; j < columns; j += 16) {
    const __m512 x = _mm512_loadu_ps(query + j);
    lo = _mm512_min_ps(lo, x);
    hi = _mm512_max_ps(hi, x);
  }
  minimum = _mm512_reduce_min_ps(lo);
  const float maximum = _mm512_reduce_max_ps(hi);
  delta = maximum > minimum ? (maximum - minimum) / 255.f : 1.f;
  const __m512 origin = _mm512_set1_ps(minimum), step = _mm512_set1_ps(delta);
  for (size_t j = 0; j < columns; j += 16) {
    const __m512 x =
        _mm512_div_ps(_mm512_sub_ps(_mm512_loadu_ps(query + j), origin), step);
    const __m128i bytes = _mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(x));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + j), bytes);
  }
}
__attribute__((target("avx512f,avx512bw,avx512vnni"))) inline void
projection_dot_i8(const int8_t *matrix, const uint8_t *q, size_t rows,
                  size_t columns, const float *scales, const int32_t *sums,
                  float delta, float minimum, float *out) {
  for (size_t row = 0; row < rows; row += 4) {
    __m512i a = _mm512_setzero_si512(), b = a, c = a, d = a;
    for (size_t j = 0; j < columns; j += 64) {
      __m512i x = _mm512_loadu_si512(q + j);
      a = _mm512_dpbusd_epi32(a, x,
                              _mm512_loadu_si512(matrix + row * columns + j));
      b = _mm512_dpbusd_epi32(
          b, x, _mm512_loadu_si512(matrix + (row + 1) * columns + j));
      c = _mm512_dpbusd_epi32(
          c, x, _mm512_loadu_si512(matrix + (row + 2) * columns + j));
      d = _mm512_dpbusd_epi32(
          d, x, _mm512_loadu_si512(matrix + (row + 3) * columns + j));
    }
    out[row] = scales[row] *
               (delta * _mm512_reduce_add_epi32(a) + minimum * sums[row]);
    out[row + 1] = scales[row + 1] * (delta * _mm512_reduce_add_epi32(b) +
                                      minimum * sums[row + 1]);
    out[row + 2] = scales[row + 2] * (delta * _mm512_reduce_add_epi32(c) +
                                      minimum * sums[row + 2]);
    out[row + 3] = scales[row + 3] * (delta * _mm512_reduce_add_epi32(d) +
                                      minimum * sums[row + 3]);
  }
}
#endif
class SymQGIntegerProjection {
 public:
  void prepare(const std::vector<float> &matrix, size_t rows, size_t columns) {
    if (!rows || !columns || matrix.size() / columns != rows ||
        matrix.size() % columns)
      throw std::invalid_argument("invalid projection matrix shape");
    rows_ = rows;
    columns_ = columns;
    matrix_.resize(matrix.size());
    scales_.resize(rows);
    sums_.assign(rows, 0);
    for (size_t i = 0; i < rows; ++i) {
      float m = 0;
      for (size_t j = 0; j < columns; ++j)
        m = std::max(m, std::abs(matrix[i * columns + j]));
      scales_[i] = m > 0 ? m / 127.f : 1.f;
      for (size_t j = 0; j < columns; ++j) {
        int v = int(std::nearbyint(matrix[i * columns + j] / scales_[i]));
        v = std::clamp(v, -127, 127);
        matrix_[i * columns + j] = int8_t(v);
        sums_[i] += v;
      }
    }
  }
  bool project(const float *query, std::vector<float> &out,
               std::vector<uint8_t> &scratch) const {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool available = __builtin_cpu_supports("avx512vnni") &&
                                  __builtin_cpu_supports("avx512bw");
    if (!available || !rows_ || !columns_ || rows_ % 4 || columns_ % 64)
      return false;
    float lo, delta;
    scratch.resize(columns_);
    out.resize(rows_);
    projection_quantize_u8(query, columns_, scratch.data(), delta, lo);
    projection_dot_i8(matrix_.data(), scratch.data(), rows_, columns_,
                      scales_.data(), sums_.data(), delta, lo, out.data());
    return true;
#else
    (void)query;
    (void)out;
    (void)scratch;
    return false;
#endif
  }

 private:
  size_t rows_ = 0, columns_ = 0;
  std::vector<int8_t> matrix_;
  std::vector<float> scales_;
  std::vector<int32_t> sums_;
};
}  // namespace zvec::turbo::research
