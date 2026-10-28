// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include <gtest/gtest.h>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <random>
#include "algorithm/quantized_graph/quantized_graph_integer_center.h"
#include "algorithm/quantized_graph/quantized_graph_sq8_center.h"
namespace zvec::core {
TEST(QuantizedGraphIntegerCenter, DotMatchesWideReferenceIncludingTails) {
  if (!__builtin_cpu_supports("avx512f") ||
      !__builtin_cpu_supports("avx512bw") ||
      !__builtin_cpu_supports("avx512vnni"))
    GTEST_SKIP();
  std::mt19937 random(71);
  std::vector<uint8_t> center(4096);
  std::vector<int8_t> query(4096);
  for (int trial = 0; trial < 3; ++trial) {
    for (size_t i = 0; i < center.size(); ++i) {
      center[i] = trial == 0 ? random() % 256 : 255;
      query[i] =
          trial == 0 ? int(random() % 255) - 127 : (trial == 1 ? 127 : -127);
    }
    int64_t reference = 0;
    for (size_t n = 0; n <= center.size(); ++n) {
      EXPECT_EQ(reference, IntegerCenter::dot(center.data(), query.data(), n))
          << n;
      if (n < center.size()) reference += int64_t(center[n]) * query[n];
    }
  }
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
          IntegerCenter::prepare<true>(query.data(), zero.data(), n, prepared));
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
      float error, norm, offset, scale, residual, coded, projection,
          projection_error;
      double mean_dot;
      ASSERT_TRUE(SQ8Center::encode_shifted(
          center.data(), mean.data(), n, codes.data(), error, norm, offset,
          scale, projection, projection_error));
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
