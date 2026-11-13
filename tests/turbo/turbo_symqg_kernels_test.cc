// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include <array>
#include <bitset>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/distance/common/symqg_distance.h>
#include <turbo/distance/common/symqg_kernels.h>

namespace zvec::turbo {
#if RABITQ_SUPPORTED
TEST(SymQGKernels, FixedScanIntegerSumsAndUnalignedStorage) {
  if (!__builtin_cpu_supports("avx512f") ||
      !__builtin_cpu_supports("avx512bw") ||
      !__builtin_cpu_supports("avx512dq"))
    GTEST_SKIP();
  std::mt19937 random(261109);
  std::array<float, 32> zero{}, one, expected, actual;
  one.fill(1);
  std::array<char, sizeof(one) + 3> packed;
  std::memcpy(packed.data() + 3, one.data(), sizeof(one));
  for (size_t n : {16, 64, 128, 256, 512, 768, 1024, 2048, 4096, 16384}) {
    std::vector<uint8_t> codes(n * 4 + 1), lut(n * 4 + 3);
    for (size_t trial = 0; trial < 64; ++trial) {
      SCOPED_TRACE(testing::Message() << n << ':' << trial);
      for (auto &x : codes) x = random();
      for (auto &x : lut) x = trial == 0 ? 255 : random();
      ASSERT_TRUE(symqg_scan(codes.data() + 1, lut.data() + 3, n, zero.data(),
                             one.data(), 0, 1, 0, 0, expected.data()));
      ASSERT_TRUE(SymQGFixedScanDispatch(codes.data() + 1, lut.data() + 3, n,
                                         zero.data(), packed.data() + 3, 0, 1,
                                         0, 0, actual.data()));
      EXPECT_EQ(0, std::memcmp(expected.data(), actual.data(), sizeof(actual)));
    }
  }
}

TEST(SymQGKernels, FixedScanScoresMatchScalarReference) {
  if (!__builtin_cpu_supports("avx512f") ||
      !__builtin_cpu_supports("avx512bw") ||
      !__builtin_cpu_supports("avx512dq"))
    GTEST_SKIP();
  std::mt19937 random(261110);
  std::uniform_real_distribution<float> value(-10, 10);
  std::array<float, 32> zero{}, one, add, scale, sums, actual;
  one.fill(1);
  for (size_t n : {64, 128, 256, 512, 1024}) {
    std::vector<uint8_t> codes(n * 4), lut(n * 4);
    for (size_t trial = 0; trial < 128; ++trial) {
      SCOPED_TRACE(testing::Message() << n << ':' << trial);
      for (auto &x : codes) x = random();
      for (auto &x : lut) x = random();
      for (auto &x : add) x = value(random);
      for (auto &x : scale) x = value(random);
      ASSERT_TRUE(symqg_scan(codes.data(), lut.data(), n, zero.data(),
                             one.data(), 0, 1, 0, 0, sums.data()));
      ASSERT_TRUE(SymQGFixedScanDispatch(codes.data(), lut.data(), n,
                                         add.data(), scale.data(), .25f, .03f,
                                         -.75f, .125f, actual.data()));
      for (size_t j = 0; j < 32; ++j) {
        const double term = double(.03f) * sums[j];
        const double expected =
            double(scale[j]) * (term - .75 + .125) + double(add[j]) + .25;
        // Allow either fused or separate rounding, including cancellation.
        const double magnitude =
            std::abs(scale[j]) * (term + .75 + .125) + std::abs(add[j]) + .25;
        EXPECT_NEAR(expected, actual[j],
                    8 * std::numeric_limits<float>::epsilon() * magnitude);
      }
    }
  }
}
#endif

TEST(SymQGKernels, ScanAcceptsUnalignedPackedFactors) {
  std::mt19937 random(77);
  std::uniform_real_distribution<float> value(-10.f, 10.f);
  std::array<float, 32> add, scale, expected, actual;
  std::array<char, 32 * sizeof(float) + 3> packed_add, packed_scale;
  for (size_t n : {16, 64, 768, 1024, 2048, 4096}) {
    std::vector<uint8_t> codes(n * 4), lut(n * 4);
    for (int trial = 0; trial < 16; ++trial) {
      for (auto &x : codes) x = random();
      for (auto &x : lut) x = random();
      for (size_t j = 0; j < 32; ++j) {
        add[j] = value(random);
        scale[j] = value(random);
      }
      std::memcpy(packed_add.data() + 1, add.data(), sizeof(add));
      std::memcpy(packed_scale.data() + 3, scale.data(), sizeof(scale));
      if (!symqg_scan(codes.data(), lut.data(), n, add.data(), scale.data(),
                      .25f, .03f, -.75f, .125f, expected.data()))
        GTEST_SKIP();
      ASSERT_TRUE(symqg_scan(codes.data(), lut.data(), n, packed_add.data() + 1,
                             packed_scale.data() + 3, .25f, .03f, -.75f, .125f,
                             actual.data()));
      EXPECT_EQ(0, std::memcmp(expected.data(), actual.data(), sizeof(actual)));
    }
  }
  actual.fill(17.f);
  for (size_t n : {0, 1, 15, 17}) {
    EXPECT_FALSE(symqg_scan(nullptr, nullptr, n, nullptr, nullptr, 0, 0, 0, 0,
                            actual.data()));
    for (float x : actual) EXPECT_EQ(17.f, x);
  }
}

TEST(SymQGKernels, HadamardMatchesScalarRounding) {
  std::array<float, 64> probe{};
  if (!symqg_hadamard(probe.data(), probe.size())) GTEST_SKIP();
  std::mt19937 random(75);
  std::uniform_real_distribution<float> value(-100, 100);
  for (size_t n = 16; n <= 4096; n *= 2) {
    for (int trial = 0; trial < 32; ++trial) {
      SCOPED_TRACE(testing::Message() << n << ":" << trial);
      std::vector<float> actual(n);
      for (float &x : actual) x = trial == 0 ? 0 : value(random);
      auto expected = actual;
      for (size_t stride = 1; stride < n; stride *= 2)
        for (size_t offset = 0; offset < n; offset += 2 * stride)
          for (size_t j = 0; j < stride; ++j) {
            const float a = expected[offset + j],
                        b = expected[offset + j + stride];
            expected[offset + j] = a + b;
            expected[offset + j + stride] = a - b;
          }
      ASSERT_TRUE(symqg_hadamard(actual.data(), n));
      EXPECT_EQ(0,
                std::memcmp(actual.data(), expected.data(), n * sizeof(float)));
    }
  }
  std::array<float, 128> untouched;
  untouched.fill(17);
  for (size_t n : {0, 15, 48, 80}) {
    EXPECT_FALSE(symqg_hadamard(untouched.data(), n));
    for (float x : untouched) EXPECT_EQ(17, x);
  }
}

TEST(SymQGKernels, PaleyTensorMatchesIndependentMatrix) {
  std::array<float, 768> probe{};
  if (!symqg_hadamard768(probe.data())) GTEST_SKIP();
  // Independent tensor matrix: H12[i/64,j/64] * (-1)^popcount((i%64)&(j%64)).
  static constexpr int h[12][12] = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                    {1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1},
                                    {1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1},
                                    {1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1},
                                    {1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1},
                                    {1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1},
                                    {1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1},
                                    {1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1},
                                    {1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1},
                                    {1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1},
                                    {1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1},
                                    {1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1}};
  for (size_t column = 0; column < 768; ++column) {
    probe.fill(0);
    probe[column] = 1;
    ASSERT_TRUE(symqg_hadamard768(probe.data()));
    for (size_t row = 0; row < 768; ++row) {
      const int sign =
          std::bitset<6>((row % 64) & (column % 64)).count() % 2 ? -1 : 1;
      ASSERT_EQ(h[row / 64][column / 64] * sign, probe[row])
          << row << ":" << column;
    }
  }
  std::mt19937 random(76);
  std::uniform_real_distribution<float> value(-1, 1);
  for (int trial = 0; trial < 16; ++trial) {
    double magnitude = 0;
    for (float &x : probe) {
      x = value(random);
      magnitude += std::abs(double(x));
    }
    const auto input = probe;
    ASSERT_TRUE(symqg_hadamard768(probe.data()));
    for (size_t row = 0; row < 768; ++row) {
      double expected = 0;
      for (size_t column = 0; column < 768; ++column) {
        const int sign =
            std::bitset<6>((row % 64) & (column % 64)).count() % 2 ? -1 : 1;
        expected += double(input[column]) * h[row / 64][column / 64] * sign;
      }
      EXPECT_NEAR(expected, probe[row],
                  64. * std::numeric_limits<float>::epsilon() * magnitude);
    }
  }
}
}  // namespace zvec::turbo
