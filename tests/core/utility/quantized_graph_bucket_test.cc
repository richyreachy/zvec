// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include "algorithm/quantized_graph/quantized_graph_bucket.h"
#include <cmath>
#include <limits>
#include <gtest/gtest.h>

namespace zvec::core {
TEST(QuantizedGraphBucket, OrdersBinsAndRetainsAllCandidatesAcrossReset) {
  QuantizedGraphBucket bucket;
  for (float width : {0.5f, 0.125f}) {
    bucket.reset(10);
    bucket.set_scale(1.0f, width);
    bucket.insert(1, 2.5f * width);
    bucket.insert(2, -width);
    bucket.insert(3, 0.0f);
    bucket.insert(4, 2.5f * width);
    bucket.insert(5, std::numeric_limits<float>::max());
    for (uint32_t id : {2u, 3u, 4u, 1u, 5u}) {
      ASSERT_TRUE(bucket.has_next());
      EXPECT_EQ(id, bucket.pop());
    }
    EXPECT_FALSE(bucket.has_next());
    bucket.insert(6, 0.0f);
  }
  bucket.reset(10);
  EXPECT_FALSE(bucket.has_next());
}

TEST(QuantizedGraphBucket, TinyWidthFallsBackToFiniteQuotients) {
  QuantizedGraphBucket bucket;
  bucket.reset(10);
  bucket.set_scale(1e-6f, 1e-34f);
  const float width = bucket.width();
  ASSERT_GT(width, 0.0f);
  ASSERT_FALSE(std::isfinite(1.0f / width));
  EXPECT_EQ(1u, bucket.locate(0.0f));
  EXPECT_EQ(2u, bucket.locate(width));
  EXPECT_EQ(3u, bucket.locate(2.0f * width));
  EXPECT_EQ(0u, bucket.locate(-width));
  EXPECT_EQ(QuantizedGraphBucket::N - 1,
            bucket.locate(std::numeric_limits<float>::max()));
  bucket.reset(10);
  EXPECT_EQ(2u, bucket.locate(1.0f));
}
TEST(QuantizedGraphBucket, BatchBinsMatchScalarAtBoundariesAndExtremes) {
  QuantizedGraphBucket bucket;
  for (float coefficient : {0.016f, 0.004f, 1e-34f}) {
    bucket.reset(10);
    bucket.set_scale(coefficient == 1e-34f ? 1e-6f : 1.0f, coefficient);
    std::vector<float> scores = {0.f,
                                 -0.f,
                                 -1.f,
                                 std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::quiet_NaN()};
    for (size_t bin = 0; bin < QuantizedGraphBucket::N + 2; ++bin) {
      const float boundary = float(bin) * bucket.width();
      scores.push_back(
          std::nextafter(boundary, -std::numeric_limits<float>::infinity()));
      scores.push_back(boundary);
      scores.push_back(
          std::nextafter(boundary, std::numeric_limits<float>::infinity()));
    }
    std::vector<uint32_t> bins(scores.size());
    bucket.locate_batch(scores.data(), bins.data(), scores.size());
    for (size_t i = 0; i < scores.size(); ++i)
      ASSERT_EQ(bucket.locate(scores[i]), bins[i]) << i;
  }
}
}  // namespace zvec::core
