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

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/turbo/turbo.h>

using namespace zvec;
using namespace zvec::turbo;

namespace {

// Odd dimension exercises both the full 16-lane strips and the masked tail
// of the contiguous sweep; the count exceeds its prefetch lookahead.
constexpr size_t kDimension = 69;
constexpr size_t kVectorCount = 25;

//! Compare the contiguous batch kernel against a per-vector sweep of the
//! single-distance kernel of the same family.
template <typename T>
void TestContiguousMatchesSingleDistance(MetricType metric_type,
                                         DataType data_type,
                                         QuantizeType quantize_type,
                                         CpuArchType cpu_arch_type,
                                         float epsilon) {
  auto single_distance =
      get_distance_func(metric_type, data_type, quantize_type, cpu_arch_type);
  auto contiguous_distance = get_contiguous_batch_distance_func(
      metric_type, data_type, quantize_type, cpu_arch_type);
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

  std::vector<float> expected(kVectorCount);
  std::vector<float> actual(kVectorCount);
  for (size_t i = 0; i < kVectorCount; ++i) {
    single_distance(block.data() + i * kDimension, query.data(), kDimension,
                    &expected[i]);
  }
  contiguous_distance(block.data(), query.data(), kVectorCount, kDimension,
                      actual.data());

  for (size_t i = 0; i < kVectorCount; ++i) {
    EXPECT_NEAR(expected[i], actual[i], epsilon);
  }
}

}  // namespace

TEST(TurboContiguousDistance, InnerProductFp32Auto) {
  TestContiguousMatchesSingleDistance<float>(
      MetricType::kInnerProduct, DataType::kFp32, QuantizeType::kFp32,
      CpuArchType::kAuto, 1e-4f);
}

TEST(TurboContiguousDistance, InnerProductFp32Scalar) {
  TestContiguousMatchesSingleDistance<float>(
      MetricType::kInnerProduct, DataType::kFp32, QuantizeType::kFp32,
      CpuArchType::kScalar, 1e-4f);
}

TEST(TurboContiguousDistance, SquaredEuclideanFp32Auto) {
  TestContiguousMatchesSingleDistance<float>(
      MetricType::kSquaredEuclidean, DataType::kFp32, QuantizeType::kFp32,
      CpuArchType::kAuto, 1e-4f);
}

TEST(TurboContiguousDistance, SquaredEuclideanFp32Scalar) {
  TestContiguousMatchesSingleDistance<float>(
      MetricType::kSquaredEuclidean, DataType::kFp32, QuantizeType::kFp32,
      CpuArchType::kScalar, 1e-4f);
}

TEST(TurboContiguousDistance, InnerProductFp16Auto) {
  TestContiguousMatchesSingleDistance<ailego::Float16>(
      MetricType::kInnerProduct, DataType::kFp16, QuantizeType::kFp16,
      CpuArchType::kAuto, 1e-3f);
}

TEST(TurboContiguousDistance, SquaredEuclideanFp16Auto) {
  TestContiguousMatchesSingleDistance<ailego::Float16>(
      MetricType::kSquaredEuclidean, DataType::kFp16, QuantizeType::kFp16,
      CpuArchType::kAuto, 1e-3f);
}

// The AVX512 fp32 sweep and the synthesized scalar fallback must agree.
TEST(TurboContiguousDistance, Fp32AutoMatchesScalar) {
  auto auto_ip = get_contiguous_batch_distance_func(
      MetricType::kInnerProduct, DataType::kFp32, QuantizeType::kFp32,
      CpuArchType::kAuto);
  auto scalar_ip = get_contiguous_batch_distance_func(
      MetricType::kInnerProduct, DataType::kFp32, QuantizeType::kFp32,
      CpuArchType::kScalar);
  ASSERT_TRUE(auto_ip);
  ASSERT_TRUE(scalar_ip);

  std::vector<float> query(kDimension);
  std::vector<float> block(kVectorCount * kDimension);
  for (size_t d = 0; d < kDimension; ++d) {
    query[d] = static_cast<float>(static_cast<int>(d % 13) - 6) / 13.0f;
  }
  for (size_t i = 0; i < kVectorCount; ++i) {
    for (size_t d = 0; d < kDimension; ++d) {
      block[i * kDimension + d] =
          static_cast<float>(((i + 7) * (d + 11)) % 31) / 31.0f;
    }
  }

  std::vector<float> auto_out(kVectorCount);
  std::vector<float> scalar_out(kVectorCount);
  auto_ip(block.data(), query.data(), kVectorCount, kDimension,
          auto_out.data());
  scalar_ip(block.data(), query.data(), kVectorCount, kDimension,
            scalar_out.data());
  for (size_t i = 0; i < kVectorCount; ++i) {
    EXPECT_NEAR(auto_out[i], scalar_out[i], 1e-4f);
  }
}

// Kernel families whose batch path needs a preprocessed query must not get
// a synthesized contiguous fallback.
TEST(TurboContiguousDistance, PreprocessedFamiliesReturnNull) {
  auto fn = get_contiguous_batch_distance_func(MetricType::kSquaredEuclidean,
                                               DataType::kInt8,
                                               QuantizeType::kUniformUint8);
  EXPECT_FALSE(fn);
}
