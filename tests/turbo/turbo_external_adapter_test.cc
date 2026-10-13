// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <gtest/gtest.h>
#include <turbo/turbo.h>
#include <zvec/turbo/turbo.h>

namespace {
namespace zt = zvec::turbo;
namespace kt = ::turbo;

TEST(TurboExternalAdapter, ForwardsActualKernelPointersAndPreprocessTogether) {
  for (auto arch : {zt::CpuArchType::kAuto, zt::CpuArchType::kScalar,
                    zt::CpuArchType::kSSE2, zt::CpuArchType::kAVX2,
                    zt::CpuArchType::kAVX512, zt::CpuArchType::kAVX512VNNI,
                    zt::CpuArchType::kNEON}) {
    for (auto type : {zt::DataType::kFp32, zt::DataType::kInt8}) {
      const auto quant = type == zt::DataType::kFp32
                             ? zt::QuantizeType::kFp32
                             : zt::QuantizeType::kRecord;
      const auto actual = zt::get_distance_kernels(
          zt::MetricType::kSquaredEuclidean, type, quant, arch);
      const auto expected = kt::get_distance_kernels(
          kt::MetricType::kSquaredEuclidean, static_cast<kt::DataType>(type),
          static_cast<kt::QuantizeType>(quant),
          static_cast<kt::CpuArchType>(arch));
      ASSERT_EQ(static_cast<bool>(actual.dist), expected.dist != nullptr);
      ASSERT_EQ(static_cast<bool>(actual.batch), expected.batch != nullptr);
      EXPECT_EQ(actual.preprocess, expected.preprocess);
      if (expected.dist) {
        const auto *ptr = actual.dist.target<kt::DistanceFunc>();
        ASSERT_NE(ptr, nullptr);
        EXPECT_EQ(*ptr, expected.dist);
      }
      if (expected.batch) {
        const auto *ptr = actual.batch.target<kt::BatchDistanceFunc>();
        ASSERT_NE(ptr, nullptr);
        EXPECT_EQ(*ptr, expected.batch);
      }
    }
  }
}

TEST(TurboExternalAdapter, LegacyBatchSignatureAndCapturingAdaptersStillWork) {
  const auto kernels =
      zt::get_distance_kernels(zt::MetricType::kSquaredEuclidean,
                               zt::DataType::kFp32, zt::QuantizeType::kFp32);
  ASSERT_TRUE(kernels.dist);
  ASSERT_TRUE(kernels.batch);
  const std::array<float, 3> a = {1, 2, 3}, b = {4, 6, 3};
  const void *candidates[] = {a.data(), b.data()};
  float results[2] = {};
  kernels.batch(candidates, a.data(), 2, a.size(), results, nullptr);
  EXPECT_FLOAT_EQ(results[0], 0);
  EXPECT_FLOAT_EQ(results[1], 25);
  zt::DistanceFunc captured = [fn = kernels.dist, bias = 3.0f](
                                  const void *lhs, const void *rhs, size_t dim,
                                  float *out) {
    fn(lhs, rhs, dim, out);
    *out += bias;
  };
  captured(a.data(), b.data(), a.size(), results);
  EXPECT_FLOAT_EQ(results[0], 28);
}

TEST(TurboExternalAdapter, UnsupportedSelectorsReturnEmpty) {
  EXPECT_FALSE(zt::get_distance_kernels(zt::MetricType::kUnknown,
                                        zt::DataType::kFp32,
                                        zt::QuantizeType::kFp32)
                   .dist);
  EXPECT_FALSE(zt::get_distance_kernels(
                   zt::MetricType::kSquaredEuclidean, zt::DataType::kFp32,
                   zt::QuantizeType::kFp32, zt::CpuArchType::kSVE)
                   .dist);
  EXPECT_EQ(zt::get_pq_kernels(zt::DataType::kUnknown).asymmetric_distance,
            nullptr);
}
}  // namespace
