// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include <gtest/gtest.h>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <random>
#include "algorithm/quantized_graph/quantized_graph_integer_center.h"
#include "algorithm/quantized_graph/quantized_graph_integer_router.h"
#include "algorithm/quantized_graph/quantized_graph_sq8_center.h"
namespace zvec::core {
namespace {
__attribute__((target("avx512f"))) void check_row_reduction() {
  std::mt19937 random(80);
  alignas(64) uint32_t values[16][16], actual[16];
  for (size_t trial = 0; trial < 1000; ++trial) {
    uint32_t expected[16]{};
    for (size_t row = 0; row < 16; ++row)
      for (size_t col = 0; col < 16; ++col) {
        values[row][col] = random();
        expected[row] += values[row][col];
      }
    __m512i rows[16];
    for (size_t row = 0; row < 16; ++row)
      rows[row] = _mm512_load_si512(values[row]);
    _mm512_store_si512(actual,
                       QuantizedGraphIntegerRouter::reduce_products(rows));
    for (size_t row = 0; row < 16; ++row) EXPECT_EQ(expected[row], actual[row]);
  }
}
}  // namespace
TEST(QuantizedGraphIntegerCenter, RouterRowReductionMatchesScalar) {
  if (!__builtin_cpu_supports("avx512f")) GTEST_SKIP();
  check_row_reduction();
}

TEST(QuantizedGraphIntegerCenter, NormBoundContainsGeometricProduct) {
  const auto check = [](double q, float c2) {
    const long double reference =
        static_cast<long double>(q) * std::sqrt(static_cast<long double>(c2));
    EXPECT_GE(
        static_cast<long double>(IntegerCenter::norm_product_bound(q, c2)),
        reference)
        << q << ":" << c2;
  };
  for (double q : {0., 1e-12, 1., 1e12, 1e40})
    for (float c2 : {0.f, std::numeric_limits<float>::denorm_min(), 1e-12f, 1.f,
                     1e12f, std::numeric_limits<float>::max()})
      check(q, c2);
  std::mt19937 random(79);
  std::uniform_real_distribution<float> exponent(-30.f, 30.f);
  for (size_t i = 0; i < 100000; ++i) {
    const float c2 = std::pow(10.f, exponent(random));
    const double q =
        i % 2 ? std::pow(10., double(exponent(random))) : std::sqrt(double(c2));
    check(q, c2);
  }
}

TEST(QuantizedGraphIntegerCenter, NextDownMatchesLibm) {
  const auto check = [](uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    const float expected =
        std::nextafter(value, -std::numeric_limits<float>::infinity());
    const float actual = IntegerCenter::next_down(value);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual));
    } else {
      uint32_t expected_bits, actual_bits;
      std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
      std::memcpy(&actual_bits, &actual, sizeof(actual_bits));
      EXPECT_EQ(expected_bits, actual_bits) << bits;
    }
  };
  for (uint32_t bits :
       {0U, 1U, 0x007fffffU, 0x00800000U, 0x00800001U, 0x3f800000U, 0x7f7ffffeU,
        0x7f7fffffU, 0x7f800000U, 0x7f800001U, 0x7fc00000U}) {
    check(bits);
    check(bits | 0x80000000U);
  }
  std::mt19937 random(75);
  for (size_t i = 0; i < 100000; ++i) check(random());
}

TEST(QuantizedGraphIntegerCenter, WideDotMatchesReferenceIncludingTails) {
  if (!__builtin_cpu_supports("avx512f") ||
      !__builtin_cpu_supports("avx512bw") ||
      !__builtin_cpu_supports("avx512dq") ||
      !__builtin_cpu_supports("avx512vnni"))
    GTEST_SKIP();
  std::mt19937 random(72);
  std::vector<uint8_t> center(4096);
  std::vector<int16_t> query(4096);
  for (int trial = 0; trial < 3; ++trial) {
    for (size_t i = 0; i < center.size(); ++i) {
      center[i] = trial == 0 ? random() % 256 : 255;
      query[i] = trial == 0 ? int(random() % 65536) - 32768
                            : (trial == 1 ? 32767 : -32768);
    }
    int64_t reference = 0;
    for (size_t n = 0; n <= center.size(); ++n) {
      EXPECT_EQ(reference, IntegerCenter::dot16(center.data(), query.data(), n))
          << n;
      if (n < center.size()) reference += int64_t(center[n]) * query[n];
    }
  }
}

TEST(QuantizedGraphIntegerCenter, L2BoundsIncludeExactDistances) {
  if (!__builtin_cpu_supports("avx512f") ||
      !__builtin_cpu_supports("avx512bw") ||
      !__builtin_cpu_supports("avx512dq") ||
      !__builtin_cpu_supports("avx512vnni") || !__builtin_cpu_supports("fma"))
    GTEST_SKIP();
  std::mt19937 random(73);
  std::normal_distribution<float> value;
  for (size_t n : {1, 15, 16, 63, 64, 65, 768, 960, 4096}) {
    for (int trial = 0; trial < 68; ++trial) {
      SCOPED_TRACE(testing::Message() << n << ":" << trial);
      const float magnitude = std::pow(10.f, float(trial % 17 - 8));
      std::vector<float> center(n), query(n), zero(n);
      std::vector<uint8_t> codes(n);
      for (size_t j = 0; j < n; ++j) {
        center[j] = magnitude * value(random);
        query[j] = trial % 3 ? center[j] + magnitude * .001f * value(random)
                             : magnitude * value(random);
      }
      if (trial % 19 == 0) std::fill(center.begin(), center.end(), magnitude);
      float error, norm, offset, scale, residual, coded;
      double mean_dot;
      ASSERT_TRUE(SQ8Center::encode(center.data(), n, codes.data(), error, norm,
                                    offset, scale));
      IntegerCenter::annotate(center.data(), zero.data(), codes.data(), n,
                              offset, scale, error, residual, coded, mean_dot);
      IntegerCenter::Query prepared;
      ASSERT_TRUE(
          IntegerCenter::prepare(query.data(), zero.data(), n, prepared));
      double reconstructed;
      IntegerCenter::l2_distance(prepared, codes.data(), n, offset, scale, norm,
                                 coded, reconstructed);
      const float lower = IntegerCenter::l2_lower_bound(reconstructed, prepared,
                                                        error, coded, n);
      double exact = 0;
      float reference = 0;
      for (size_t j = 0; j < n; ++j) {
        const double diff = double(query[j]) - center[j];
        exact += diff * diff;
        const float rounded = query[j] - center[j];
        reference += rounded * rounded;
      }
      ASSERT_TRUE(std::isfinite(lower));
      EXPECT_LE(lower, exact);
      EXPECT_LE(lower, reference);
    }
  }
}

TEST(QuantizedGraphIntegerCenter, ShiftedCosineBoundsIncludeExactDistances) {
  if (!__builtin_cpu_supports("avx512f") ||
      !__builtin_cpu_supports("avx512bw") ||
      !__builtin_cpu_supports("avx512dq") ||
      !__builtin_cpu_supports("avx512vnni") || !__builtin_cpu_supports("fma"))
    GTEST_SKIP();
  std::mt19937 random(74);
  std::normal_distribution<float> value;
  for (size_t n : {1, 15, 16, 63, 64, 65, 768, 960, 4096}) {
    for (int trial = 0; trial < 80; ++trial) {
      SCOPED_TRACE(testing::Message() << n << ":" << trial);
      std::vector<float> center(n), query(n), mean(n);
      std::vector<uint8_t> codes(n);
      double center_norm = 0, query_norm = 0, mean_norm = 0;
      for (size_t j = 0; j < n; ++j) {
        center[j] = value(random);
        query[j] = trial % 3 ? center[j] + .01f * value(random) : value(random);
        center_norm += double(center[j]) * center[j];
        query_norm += double(query[j]) * query[j];
      }
      constexpr float shifts[] = {0, .1f, 1, 1e4f};
      for (size_t j = 0; j < n; ++j) {
        center[j] /= std::sqrt(center_norm);
        query[j] = trial % 19 ? query[j] / std::sqrt(query_norm) : 0;
        mean[j] = shifts[trial % 4] * value(random) / std::sqrt(float(n));
        mean_norm += double(mean[j]) * mean[j];
      }
      float error, norm, offset, scale, residual, coded;
      double mean_dot;
      ASSERT_TRUE(SQ8Center::encode_shifted(center.data(), mean.data(), n,
                                            codes.data(), error, norm, offset,
                                            scale));
      IntegerCenter::annotate(center.data(), mean.data(), codes.data(), n,
                              offset, scale, error, residual, coded, mean_dot);
      IntegerCenter::Query prepared;
      ASSERT_TRUE(
          IntegerCenter::prepare(query.data(), mean.data(), n, prepared));
      double reconstructed;
      const float distance =
          IntegerCenter::distance(prepared, codes.data(), n, offset, scale,
                                  mean_dot, mean_norm, coded, reconstructed);
      const float lower = IntegerCenter::lower_bound(
          distance, prepared, error, residual, coded, norm, mean_dot, mean_norm,
          n, reconstructed);
      double exact = 1;
      float dot = 0;
      for (size_t j = 0; j < n; ++j) {
        exact -= double(query[j]) * center[j];
        dot += query[j] * center[j];
      }
      ASSERT_TRUE(std::isfinite(lower));
      EXPECT_LE(lower, exact);
      EXPECT_LE(lower, 1.f - dot);
    }
  }
}
}  // namespace zvec::core
#endif
