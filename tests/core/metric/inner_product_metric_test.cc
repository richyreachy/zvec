// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
#include "zvec/core/framework/index_factory.h"

using namespace zvec;
using namespace zvec::core;

TEST(InnerProductMetric, General) {
  auto metric = IndexFactory::CreateMetric("InnerProduct");
  ASSERT_TRUE(metric);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_BINARY32, 64);
  ASSERT_NE(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_BINARY64, 64);
  ASSERT_NE(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_FP16, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_FP32, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_INT4, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_INT8, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_BINARY32, 64);
  EXPECT_TRUE(metric->is_matched(meta));
  EXPECT_FALSE(metric->is_matched(meta2));
  EXPECT_TRUE(metric->is_matched(
      meta, IndexQueryMeta(IndexMeta::DataType::DT_INT8, 64)));
  EXPECT_FALSE(metric->is_matched(
      meta, IndexQueryMeta(IndexMeta::DataType::DT_INT8, 63)));

  EXPECT_FALSE(metric->distance_matrix(0, 0));
  EXPECT_FALSE(metric->distance_matrix(3, 5));
  EXPECT_FALSE(metric->distance_matrix(31, 65));
  EXPECT_TRUE(metric->distance_matrix(1, 1));
  EXPECT_TRUE(metric->distance_matrix(2, 1));
  EXPECT_TRUE(metric->distance_matrix(2, 2));
  EXPECT_TRUE(metric->distance_matrix(4, 1));
  EXPECT_TRUE(metric->distance_matrix(4, 2));
  EXPECT_TRUE(metric->distance_matrix(4, 4));
  EXPECT_TRUE(metric->distance_matrix(8, 1));
  EXPECT_TRUE(metric->distance_matrix(8, 2));
  EXPECT_TRUE(metric->distance_matrix(8, 4));
  EXPECT_TRUE(metric->distance_matrix(8, 8));
  EXPECT_TRUE(metric->distance_matrix(16, 1));
  EXPECT_TRUE(metric->distance_matrix(16, 2));
  EXPECT_TRUE(metric->distance_matrix(16, 4));
  EXPECT_TRUE(metric->distance_matrix(16, 8));
  EXPECT_TRUE(metric->distance_matrix(16, 16));
  EXPECT_TRUE(metric->distance_matrix(32, 1));
  EXPECT_TRUE(metric->distance_matrix(32, 2));
  EXPECT_TRUE(metric->distance_matrix(32, 4));
  EXPECT_TRUE(metric->distance_matrix(32, 8));
  EXPECT_TRUE(metric->distance_matrix(32, 16));
  EXPECT_TRUE(metric->distance_matrix(32, 32));

  EXPECT_TRUE(metric->support_normalize());
  float result = 1.0f;
  metric->normalize(&result);
  EXPECT_FLOAT_EQ(-1.0f, result);
}

//! Compare contiguous_batch_distance against the per-vector distance handle
template <typename T>
static void TestContiguousBatchMatchesSingleDistance(
    IndexMeta::DataType data_type, float epsilon) {
  // Odd dimension exercises both the full 16-lane strips and the masked tail
  // of the contiguous sweep; the count exceeds its prefetch lookahead.
  constexpr size_t kDimension = 69;
  constexpr size_t kVectorCount = 25;

  IndexMeta meta(data_type, kDimension);
  auto metric = IndexFactory::CreateMetric("InnerProduct");
  ASSERT_TRUE(metric);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));

  auto single_distance = metric->distance();
  auto contiguous_distance = metric->contiguous_batch_distance();
  ASSERT_TRUE(single_distance);
  ASSERT_TRUE(contiguous_distance);

  std::vector<T> query(kDimension);
  std::vector<T> block(kVectorCount * kDimension);
  for (size_t d = 0; d < kDimension; ++d) {
    query[d] = static_cast<float>(static_cast<int>(d % 17) - 8) / 17.0f;
  }
  for (size_t i = 0; i < kVectorCount; ++i) {
    for (size_t d = 0; d < kDimension; ++d) {
      block[i * kDimension + d] =
          static_cast<float>(((i + 3) * (d + 5)) % 29) / 29.0f;
    }
  }

  std::array<float, kVectorCount> expected{};
  std::array<float, kVectorCount> actual{};
  for (size_t i = 0; i < kVectorCount; ++i) {
    single_distance(block.data() + i * kDimension, query.data(), kDimension,
                    &expected[i]);
  }
  contiguous_distance(block.data(), query.data(), kVectorCount, kDimension,
                      actual.data());

  for (size_t i = 0; i < kVectorCount; ++i) {
    EXPECT_NEAR(expected[i], actual[i],
                epsilon * std::max(1.0f, std::abs(expected[i])));
  }
}

TEST(InnerProductMetric, ContiguousBatchFp32MatchesSingleDistance) {
  TestContiguousBatchMatchesSingleDistance<float>(IndexMeta::DataType::DT_FP32,
                                                  1e-4f);
}

TEST(InnerProductMetric, ContiguousBatchFp16MatchesSingleDistance) {
  // distance() uses legacy ailego fp16 kernels while the contiguous path uses
  // turbo; different accumulation orders diverge up to ~2e-3 at dim 69.
  TestContiguousBatchMatchesSingleDistance<ailego::Float16>(
      IndexMeta::DataType::DT_FP16, 5e-3f);
}

TEST(InnerProductMetric, ContiguousBatchUnsupportedTypes) {
  IndexMeta meta(IndexMeta::DataType::DT_INT8, 64);
  auto metric = IndexFactory::CreateMetric("InnerProduct");
  ASSERT_TRUE(metric);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  EXPECT_FALSE(metric->contiguous_batch_distance());

  meta.set_meta(IndexMeta::DataType::DT_INT4, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  EXPECT_FALSE(metric->contiguous_batch_distance());
}