// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include "algorithm/quantized_graph/quantized_graph_bucket.h"
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <gtest/gtest.h>

namespace zvec::core {
TEST(QuantizedGraphBucket, AdmissionMaskNeverPrunesAnExpandableBin) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (!__builtin_cpu_supports("avx512f")) GTEST_SKIP();
#endif
  QuantizedGraphBucket bucket;
  std::array<uint32_t, 32> bins{};
  bins[0] = 0;
  bins[1] = 1;
  bins[2] = 11;
  bins[3] = 100;
  bucket.set_scale(1.f, .01f);
  EXPECT_EQ(3u, bucket.admissible_mask(bins.data(), 4, .05f, 0.f));
  std::mt19937 random(62);
  for (float width :
       {0.f, std::numeric_limits<float>::denorm_min(), 1e-30f, .004f, .008f,
        1.f, 1e30f, std::numeric_limits<float>::max()}) {
    bucket.set_scale(1.f, width);
    for (size_t count = 0; count <= 32; ++count) {
      for (size_t trial = 0; trial < 32; ++trial) {
        for (auto &bin : bins) bin = random() % QuantizedGraphBucket::N;
        const float worst = float(random() % 2048) * width;
        const float margin = trial % 2 ? .085f : 0.f;
        const float allowance = margin * std::max(worst, 1e-6f);
        const uint32_t mask =
            bucket.admissible_mask(bins.data(), count, worst, allowance);
        const uint32_t valid =
            count == 32 ? ~uint32_t{0} : (uint32_t{1} << count) - 1;
        EXPECT_EQ(0u, mask & ~valid);
        for (size_t j = 0; j < count; ++j) {
          // Use the actual queue's externally visible stopping score as the
          // reference, including zero, infinite and rounded bin boundaries.
          bucket.insert_at(uint32_t(j), bins[j]);
          const float next = bucket.next_distance();
          EXPECT_EQ(j, bucket.pop());
          if (!(mask & (uint32_t{1} << j))) {
            EXPECT_TRUE(next - worst >= allowance)
                << width << ":" << bins[j] << ":" << worst;
            EXPECT_TRUE(std::fma(float(bins[j]) - 1.f, width, -worst) >=
                        allowance);
          }
        }
      }
    }
  }
}

TEST(QuantizedGraphBucket, BatchStoragePreservesOrderThroughGrowthAndReuse) {
  QuantizedGraphBucket bucket;
  std::mt19937 random(61);
  std::map<size_t, std::vector<uint32_t>> reference;
  for (uint32_t iteration = 0; iteration < 4000; ++iteration) {
    if (iteration % 101 == 0) {
      bucket.reset();
      reference.clear();
    }
    const size_t count = random() % 33;
    bucket.reserve_insertions(count);
    for (size_t i = 0; i < count; ++i) {
      const size_t bin = random() % QuantizedGraphBucket::N;
      const uint32_t id = random();
      bucket.insert_reserved(id, bin);
      reference[bin].push_back(id);
    }
    const size_t pops = iteration % 7 == 0 ? 1000 : random() % 32;
    for (size_t i = 0; i < pops && !reference.empty(); ++i) {
      auto first = reference.begin();
      ASSERT_TRUE(bucket.has_next());
      ASSERT_EQ(first->first, bucket.first());
      ASSERT_EQ(first->second.back(), bucket.next_id());
      ASSERT_EQ(first->second.back(), bucket.pop());
      first->second.pop_back();
      if (first->second.empty()) reference.erase(first);
    }
    ASSERT_EQ(!reference.empty(), bucket.has_next());
  }
}

TEST(QuantizedGraphBucket, OrdersBinsAndRetainsAllCandidatesAcrossReset) {
  QuantizedGraphBucket bucket;
  for (float width : {0.5f, 0.125f}) {
    bucket.reset();
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
  bucket.reset();
  EXPECT_FALSE(bucket.has_next());
}

TEST(QuantizedGraphBucket, TinyWidthFallsBackToFiniteQuotients) {
  QuantizedGraphBucket bucket;
  bucket.reset();
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
  bucket.reset();
  EXPECT_EQ(2u, bucket.locate(1.0f));
}
TEST(QuantizedGraphBucket, BatchBinsMatchScalarAtBoundariesAndExtremes) {
  QuantizedGraphBucket bucket;
  for (float coefficient : {0.016f, 0.004f, 1e-34f}) {
    bucket.reset();
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
