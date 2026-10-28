// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include <gtest/gtest.h>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <random>
#include "algorithm/quantized_graph/quantized_graph_integer_center.h"
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
}  // namespace zvec::core
#endif
