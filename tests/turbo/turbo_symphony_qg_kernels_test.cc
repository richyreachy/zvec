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
TEST(SymphonyQGKernels, ScanAcceptsUnalignedPackedFactors) {
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
