// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include <gtest/gtest.h>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <array>
#include <limits>
#include <random>
#include <unordered_set>
#include "algorithm/quantized_graph/quantized_graph_visit_set.h"

namespace zvec::core {
TEST(QuantizedGraphVisitSet,
     BatchesMatchMembershipThroughCollisionsAndOverflow) {
  if (!__builtin_cpu_supports("avx512f")) GTEST_SKIP();
  std::mt19937 random(917);
  QuantizedGraphVisitSet visited;
  for (size_t expected : {1, 16, 128}) {
    visited.reset(expected);
    std::unordered_set<uint32_t> reference;
    for (uint32_t iteration = 0; iteration < 400; ++iteration) {
      const uint32_t key = iteration % 2 ? random() : iteration * 65536;
      if (key != UINT32_MAX) reference.insert(key);
      visited.set_visited(key);
      std::array<uint32_t, 32> ids;
      std::array<float, 32> distances;
      for (size_t lane = 0; lane < ids.size(); ++lane) {
        ids[lane] = lane % 3 ? random() : key;
        distances[lane] = float(int(lane) - 16);
      }
      ids[0] = UINT32_MAX;
      distances[1] = std::numeric_limits<float>::infinity();
      distances[2] = -std::numeric_limits<float>::infinity();
      distances[3] = std::numeric_limits<float>::quiet_NaN();
      uint32_t mask = 0;
      for (size_t count = 0; count <= ids.size(); ++count) {
        ASSERT_EQ(mask,
                  visited.unvisited_batch(ids.data(), distances.data(), count))
            << expected << ":" << iteration << ":" << count;
        if (count < ids.size() && ids[count] != UINT32_MAX &&
            std::isfinite(distances[count]) && !reference.count(ids[count]))
          mask |= uint32_t{1} << count;
      }
      ASSERT_TRUE(visited.visited(key));
    }
    visited.reset(expected);
    for (uint32_t id : reference) ASSERT_FALSE(visited.visited(id));
  }
}
}  // namespace zvec::core
#endif
