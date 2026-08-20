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
#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/turbo/turbo.h>

using namespace zvec::turbo;

namespace {

void fill_query(std::vector<float> &query, size_t dim) {
  query.resize(dim);
  for (size_t d = 0; d < dim; ++d) {
    query[d] = static_cast<float>(static_cast<int>(d % 17) - 8) / 17.0f;
  }
}

void fill_block(std::vector<float> &block, size_t n, size_t dim) {
  block.resize(n * dim);
  for (size_t i = 0; i < n; ++i) {
    for (size_t d = 0; d < dim; ++d) {
      block[i * dim + d] = static_cast<float>(((i + 3) * (d + 5)) % 29) / 29.0f;
    }
  }
}

// The dispatched (auto) kernels must agree with the scalar reference for
// single, pointer-batch and contiguous-batch entry points. Count exceeds one
// 12-chunk plus one 8-chunk plus singles, and the contiguous prefetch
// lookahead; odd dims exercise the masked/scalar tails.
void check_fp32_metric(MetricType metric, size_t dim, size_t n) {
  auto kernels =
      get_distance_kernels(metric, DataType::kFp32, QuantizeType::kFp32);
  auto reference = get_distance_kernels(
      metric, DataType::kFp32, QuantizeType::kFp32, CpuArchType::kScalar);
  ASSERT_TRUE(kernels.dist);
  ASSERT_TRUE(kernels.batch);
  ASSERT_TRUE(kernels.contiguous_batch);
  ASSERT_TRUE(reference.dist);

  std::vector<float> query;
  std::vector<float> block;
  fill_query(query, dim);
  fill_block(block, n, dim);

  std::vector<const void *> ptrs(n);
  for (size_t i = 0; i < n; ++i) {
    ptrs[i] = block.data() + i * dim;
  }

  std::vector<float> expected(n);
  for (size_t i = 0; i < n; ++i) {
    reference.dist(ptrs[i], query.data(), dim, &expected[i]);
  }
  auto tolerance = [](float value) {
    return 1e-5f * (std::fabs(value) + 1.0f);
  };

  std::vector<float> actual(n);
  for (size_t i = 0; i < n; ++i) {
    kernels.dist(ptrs[i], query.data(), dim, &actual[i]);
    EXPECT_NEAR(expected[i], actual[i], tolerance(expected[i]))
        << "single i=" << i;
  }

  std::fill(actual.begin(), actual.end(), 0.0f);
  kernels.batch(ptrs.data(), query.data(), n, dim, actual.data());
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(expected[i], actual[i], tolerance(expected[i]))
        << "batch i=" << i;
  }

  std::fill(actual.begin(), actual.end(), 0.0f);
  kernels.contiguous_batch(block.data(), query.data(), n, dim, actual.data());
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(expected[i], actual[i], tolerance(expected[i]))
        << "contiguous i=" << i;
  }
}

}  // namespace

TEST(TurboBaseDistanceTest, SquaredEuclideanFp32MatchesScalar) {
  check_fp32_metric(MetricType::kSquaredEuclidean, 69, 30);
  check_fp32_metric(MetricType::kSquaredEuclidean, 384, 30);
  check_fp32_metric(MetricType::kSquaredEuclidean, 960, 25);
  check_fp32_metric(MetricType::kSquaredEuclidean, 16, 5);
}

TEST(TurboBaseDistanceTest, InnerProductFp32MatchesScalar) {
  check_fp32_metric(MetricType::kInnerProduct, 69, 30);
  check_fp32_metric(MetricType::kInnerProduct, 384, 30);
  check_fp32_metric(MetricType::kInnerProduct, 768, 25);
  check_fp32_metric(MetricType::kInnerProduct, 16, 5);
}

TEST(TurboBaseDistanceTest, CosineFp32MatchesScalar) {
  check_fp32_metric(MetricType::kCosine, 69, 30);
  check_fp32_metric(MetricType::kCosine, 384, 30);
  check_fp32_metric(MetricType::kCosine, 16, 5);
}

TEST(TurboBaseDistanceTest, ContiguousBatchFp16MatchesSingle) {
  constexpr size_t kDim = 69;
  constexpr size_t kCount = 25;

  for (auto metric : {MetricType::kSquaredEuclidean, MetricType::kCosine,
                      MetricType::kInnerProduct}) {
    auto kernels = get_distance_kernels(
        metric, DataType::kFp16, QuantizeType::kFp16, CpuArchType::kScalar);
    ASSERT_TRUE(kernels.dist);
    ASSERT_TRUE(kernels.contiguous_batch);

    std::vector<float> query_fp32;
    std::vector<float> block_fp32;
    fill_query(query_fp32, kDim);
    fill_block(block_fp32, kCount, kDim);

    std::vector<zvec::ailego::Float16> query(kDim);
    std::vector<zvec::ailego::Float16> block(kCount * kDim);
    for (size_t d = 0; d < kDim; ++d) {
      query[d] = query_fp32[d];
    }
    for (size_t i = 0; i < kCount * kDim; ++i) {
      block[i] = block_fp32[i];
    }

    std::vector<float> expected(kCount);
    std::vector<float> actual(kCount);
    for (size_t i = 0; i < kCount; ++i) {
      kernels.dist(block.data() + i * kDim, query.data(), kDim, &expected[i]);
    }
    kernels.contiguous_batch(block.data(), query.data(), kCount, kDim,
                             actual.data());
    for (size_t i = 0; i < kCount; ++i) {
      EXPECT_NEAR(expected[i], actual[i], 1e-3f) << "i=" << i;
    }
  }
}

TEST(TurboBaseDistanceTest, ContiguousBatchUnsupportedTypes) {
  EXPECT_FALSE(get_contiguous_batch_distance_func(
      MetricType::kSquaredEuclidean, DataType::kInt8, QuantizeType::kRecord));
  EXPECT_FALSE(get_contiguous_batch_distance_func(
      MetricType::kSquaredEuclidean, DataType::kInt4, QuantizeType::kRecord));
}
