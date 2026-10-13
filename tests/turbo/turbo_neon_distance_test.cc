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

#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/turbo/turbo.h>

namespace zvec::turbo {
namespace {

using RawDistanceFn = DistanceFunc;

// Only used by the AArch64 branches of the tests below.
[[maybe_unused]] void CompareDistance(RawDistanceFn expected_fn,
                                      RawDistanceFn actual_fn, const void *a,
                                      const void *b, size_t dim,
                                      float tolerance) {
  ASSERT_TRUE(expected_fn);
  ASSERT_TRUE(actual_fn);
  float expected = 0.0f;
  float actual = 0.0f;
  expected_fn(a, b, dim, &expected);
  actual_fn(a, b, dim, &actual);
  EXPECT_NEAR(expected, actual, tolerance) << "dim=" << dim;
}

TEST(NeonDistance, Fp32MatchesScalarForRemainders) {
#if !defined(AILEGO_ARM64_NEON)
  GTEST_SKIP() << "NEON distance kernels require AArch64";
#else
  if (!get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                         QuantizeType::kFp32, CpuArchType::kNEON))
    GTEST_SKIP() << "NEON backend not compiled or unavailable";
  std::vector<float> a(33);
  std::vector<float> b(33);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
    b[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.2f;
  }

  for (size_t dim = 0; dim <= a.size(); ++dim) {
    CompareDistance(
        get_distance_func(MetricType::kInnerProduct, DataType::kFp32,
                          QuantizeType::kFp32, CpuArchType::kScalar),
        get_distance_func(MetricType::kInnerProduct, DataType::kFp32,
                          QuantizeType::kFp32, CpuArchType::kNEON),
        a.data(), b.data(), dim, 1e-5f);
    CompareDistance(
        get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                          QuantizeType::kFp32, CpuArchType::kScalar),
        get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                          QuantizeType::kFp32, CpuArchType::kNEON),
        a.data(), b.data(), dim, 1e-5f);
    CompareDistance(
        get_distance_func(MetricType::kCosine, DataType::kFp32,
                          QuantizeType::kFp32, CpuArchType::kScalar),
        get_distance_func(MetricType::kCosine, DataType::kFp32,
                          QuantizeType::kFp32, CpuArchType::kNEON),
        a.data(), b.data(), dim, 1e-5f);
  }
#endif
}

TEST(NeonDistance, Fp16MatchesScalarForRemainders) {
#if !defined(AILEGO_ARM64_GNU_LIKE)
  GTEST_SKIP() << "FP16 NEON distance kernels require GNU-like AArch64";
#else
  if (!get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                         QuantizeType::kFp32, CpuArchType::kNEON))
    GTEST_SKIP() << "NEON backend not compiled or unavailable";
  std::vector<float> a_fp32(33);
  std::vector<float> b_fp32(33);
  for (size_t i = 0; i < a_fp32.size(); ++i) {
    a_fp32[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
    b_fp32[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.2f;
  }
  std::vector<uint16_t> a(a_fp32.size());
  std::vector<uint16_t> b(b_fp32.size());
  ailego::FloatHelper::ToFP16(a_fp32.data(), a_fp32.size(), a.data());
  ailego::FloatHelper::ToFP16(b_fp32.data(), b_fp32.size(), b.data());

  for (size_t dim = 0; dim <= a.size(); ++dim) {
    CompareDistance(
        get_distance_func(MetricType::kInnerProduct, DataType::kFp16,
                          QuantizeType::kFp16, CpuArchType::kScalar),
        get_distance_func(MetricType::kInnerProduct, DataType::kFp16,
                          QuantizeType::kFp16, CpuArchType::kNEON),
        a.data(), b.data(), dim, 2e-3f);
    CompareDistance(
        get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                          QuantizeType::kRaw, CpuArchType::kScalar),
        get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                          QuantizeType::kRaw, CpuArchType::kNEON),
        a.data(), b.data(), dim, 2e-3f);
    CompareDistance(get_distance_func(MetricType::kCosine, DataType::kFp16,
                                      QuantizeType::kRaw, CpuArchType::kScalar),
                    get_distance_func(MetricType::kCosine, DataType::kFp16,
                                      QuantizeType::kRaw, CpuArchType::kNEON),
                    a.data(), b.data(), dim, 2e-3f);
  }
#endif
}

TEST(NeonDistance, Fp16AccumulatesInFp32) {
#if !defined(AILEGO_ARM64_GNU_LIKE)
  GTEST_SKIP() << "FP16 NEON distance kernels require GNU-like AArch64";
#else
  if (!get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                         QuantizeType::kFp32, CpuArchType::kNEON))
    GTEST_SKIP() << "NEON backend not compiled or unavailable";
  constexpr size_t dim = 8;
  std::vector<float> a_fp32(dim, 256.0f);
  std::vector<float> b_fp32(dim, 256.0f);
  std::vector<float> zero_fp32(dim, 0.0f);
  std::vector<uint16_t> a(dim);
  std::vector<uint16_t> b(dim);
  std::vector<uint16_t> zero(dim);
  ailego::FloatHelper::ToFP16(a_fp32.data(), dim, a.data());
  ailego::FloatHelper::ToFP16(b_fp32.data(), dim, b.data());
  ailego::FloatHelper::ToFP16(zero_fp32.data(), dim, zero.data());

  CompareDistance(get_distance_func(MetricType::kInnerProduct, DataType::kFp16,
                                    QuantizeType::kFp16, CpuArchType::kScalar),
                  get_distance_func(MetricType::kInnerProduct, DataType::kFp16,
                                    QuantizeType::kFp16, CpuArchType::kNEON),
                  a.data(), b.data(), dim, 0.0f);
  CompareDistance(
      get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                        QuantizeType::kRaw, CpuArchType::kScalar),
      get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                        QuantizeType::kRaw, CpuArchType::kNEON),
      a.data(), zero.data(), dim, 0.0f);
#endif
}

// Exercise complete encoded records instead of private integer-dot helpers.
TEST(NeonDistance, RecordIntegerProductsMatchScalarForRemainders) {
  for (DataType type : {DataType::kInt8, DataType::kInt4}) {
    const auto scalar =
        get_distance_kernels(MetricType::kInnerProduct, type,
                             QuantizeType::kRecord, CpuArchType::kScalar);
    const auto neon =
        get_distance_kernels(MetricType::kInnerProduct, type,
                             QuantizeType::kRecord, CpuArchType::kNEON);
    if (!neon.dist) GTEST_SKIP() << "NEON backend unavailable";
    ASSERT_TRUE(scalar.dist);
    for (size_t dim = 8; dim <= 104; dim += 8) {
      const size_t bytes = type == DataType::kInt8 ? dim : dim / 2;
      std::vector<float> a((bytes + 20) / 4, 0.0f), b(a.size(), 0.0f);
      auto *ac = reinterpret_cast<uint8_t *>(a.data());
      auto *bc = reinterpret_cast<uint8_t *>(b.data());
      for (size_t i = 0; i < bytes; ++i) {
        ac[i] = static_cast<uint8_t>(i * 37);
        bc[i] = static_cast<uint8_t>(i * 53);
      }
      a[bytes / 4] = 1.0f;
      b[bytes / 4] = 1.0f;  // scale; zero bias
      const size_t encoded_dim = dim + (type == DataType::kInt8 ? 20 : 32);
      CompareDistance(scalar.dist, neon.dist, a.data(), b.data(), encoded_dim,
                      0.0f);
    }
  }
}

TEST(NeonDistance, DispatchProvidesAllKernels) {
#if !defined(AILEGO_ARM64_NEON)
  GTEST_SKIP() << "NEON distance kernels require AArch64";
#else
  if (!get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                         QuantizeType::kFp32, CpuArchType::kNEON))
    GTEST_SKIP() << "NEON backend not compiled or unavailable";
  constexpr std::array<MetricType, 3> metrics = {MetricType::kSquaredEuclidean,
                                                 MetricType::kCosine,
                                                 MetricType::kInnerProduct};
  for (MetricType metric : metrics) {
    const auto fp32 = get_distance_kernels(
        metric, DataType::kFp32, QuantizeType::kFp32, CpuArchType::kNEON);
    EXPECT_TRUE(fp32.dist);
    EXPECT_TRUE(fp32.batch);

    const auto fp16 = get_distance_kernels(
        metric, DataType::kFp16, QuantizeType::kFp16, CpuArchType::kNEON);
    EXPECT_TRUE(fp16.dist);
    EXPECT_TRUE(fp16.batch);

    const auto int8 = get_distance_kernels(
        metric, DataType::kInt8, QuantizeType::kRecord, CpuArchType::kNEON);
    EXPECT_TRUE(int8.dist);
    EXPECT_TRUE(int8.batch);

    const auto int4 = get_distance_kernels(
        metric, DataType::kInt4, QuantizeType::kRecord, CpuArchType::kNEON);
    EXPECT_TRUE(int4.dist);
    EXPECT_TRUE(int4.batch);
  }
#endif
}

}  // namespace
}  // namespace zvec::turbo
