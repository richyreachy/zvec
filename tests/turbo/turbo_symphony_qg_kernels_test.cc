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
#include <turbo/distance/common/symphony_qg_kernels.h>

namespace zvec::turbo {
TEST(SymphonyQGKernels, HadamardMatchesScalarRounding) {
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

TEST(SymphonyQGKernels, PaleyTensorMatchesIndependentMatrix) {
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
