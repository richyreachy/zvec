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
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/pq_block_quantizer/pq_block_quantizer.h>
#include <turbo/quantizer/pq_residual_block_quantizer/pq_residual_block_quantizer.h>
#include <turbo/quantizer/sq8_block_quantizer/sq8_block_quantizer.h>
#include <turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h>
#if RABITQ_SUPPORTED
#include <rabitqlib/utils/cpu_features.hpp>
#endif

namespace zvec::turbo {
namespace {

TEST(TurboBlockQuantizerTest, SQ8DistanceErrorAndPartialBatches) {
  for (size_t dim : {1U, 3U, 17U, 128U}) {
    for (bool cosine : {false, true}) {
      SQ8BlockQuantizer codec(dim, cosine);
      std::vector<float> values(SQ8BlockQuantizer::kBatchSize * dim);
      std::vector<float> query(dim);
      for (size_t d = 0; d < dim; ++d) query[d] = std::sin(float(d + 1));
      auto normalize = [dim](float *v) {
        float norm = 0;
        for (size_t d = 0; d < dim; ++d) norm += v[d] * v[d];
        if (norm > 0)
          for (size_t d = 0; d < dim; ++d) v[d] /= std::sqrt(norm);
      };
      for (size_t i = 1; i < SQ8BlockQuantizer::kBatchSize; ++i) {
        for (size_t d = 0; d < dim; ++d)
          values[i * dim + d] = float(i) * std::cos(float(d + i));
        if (cosine) normalize(values.data() + i * dim);
      }
      if (cosine) normalize(query.data());
      SQ8BlockQuantizer::Query prepared;
      codec.prepare_query(query.data(), prepared);
      // Deliberately unaligned storage: odd dimensions make lane offsets
      // unaligned too. Sentinels catch writes beyond the advertised payload.
      std::vector<char> bytes(codec.batch_bytes() + 2, char{42});
      std::array<float, SQ8BlockQuantizer::kBatchSize> scores;
      for (size_t count : {0U, 1U, 15U, 16U}) {
        codec.encode(nullptr, values.data(), count, bytes.data() + 1);
        codec.scan(bytes.data() + 1, prepared, 123.0f, scores.data());
        EXPECT_EQ(42, bytes.front());
        EXPECT_EQ(42, bytes.back());
        for (size_t i = 0; i < SQ8BlockQuantizer::kBatchSize; ++i) {
          float max_abs = 0;
          for (size_t d = 0; d < dim && i < count; ++d)
            max_abs = std::max(max_abs, std::abs(values[i * dim + d]));
          const double error = max_abs / 254.0 + 1e-6;
          double expected = cosine ? 1 : 0;
          double bound = 1e-5;
          for (size_t d = 0; d < dim; ++d) {
            const double x = i < count ? values[i * dim + d] : 0;
            const double q = query[d];
            if (cosine) {
              expected -= q * x;
              bound += std::abs(q) * error;
            } else {
              expected += (q - x) * (q - x);
              bound += 2 * std::abs(q - x) * error + error * error;
            }
          }
          EXPECT_NEAR(expected, scores[i], bound);
        }
      }
      std::fill(query.begin(), query.end(), 0.0f);
      codec.prepare_query(query.data(), prepared);
      codec.scan(bytes.data() + 1, prepared, 0, scores.data());
      EXPECT_FLOAT_EQ(cosine ? 1 : 0, scores[0]);
    }
  }
}

TEST(TurboSymphonyQGTest, CosineFactorsHandleZeroCentersNeighborsAndQueries) {
  const std::array<std::array<float, 2>, 4> vectors{
      {{{1, 0}}, {{0, 1}}, {{-1, 0}}, {{0, 0}}}};
  std::vector<float> packed;
  for (const auto &vector : vectors)
    packed.insert(packed.end(), vector.begin(), vector.end());
  for (const auto &center : vectors) {
    std::array<float, 4> add, scale;
    const float center_norm = center[0] * center[0] + center[1] * center[1];
    for (size_t i = 0; i < vectors.size(); ++i) {
      const auto &x = vectors[i];
      // Exact L2 representation using t = dot(query, x - center).
      add[i] = x[0] * x[0] + x[1] * x[1] - center_norm;
      scale[i] = -2;
    }
    ConvertSymphonyQGCosineFactors(center.data(), packed.data(), vectors.size(),
                                   2, add.data(), scale.data());
    for (const auto &query : vectors) {
      const float g = 1 - query[0] * center[0] - query[1] * center[1];
      for (size_t i = 0; i < vectors.size(); ++i) {
        const auto &x = vectors[i];
        const float t =
            query[0] * (x[0] - center[0]) + query[1] * (x[1] - center[1]);
        EXPECT_FLOAT_EQ(1 - query[0] * x[0] - query[1] * x[1],
                        add[i] + g + scale[i] * t);
      }
    }
  }
}

TEST(TurboSymphonyQGRotationTest, PreservesDistancesAndRebuildsIdentically) {
  for (size_t dim : {1U, 16U, 63U, 64U, 65U, 128U, 1025U, 4096U}) {
    SCOPED_TRACE(dim);
    SymphonyQGRotation rotation(dim), reopened(dim);
    std::vector<float> a(dim), b(dim), ra, rb, repeat;
    double original = 0;
    for (size_t i = 0; i < dim; ++i) {
      a[i] = std::sin(static_cast<float>(i));
      b[i] = std::cos(static_cast<float>(i) * 0.7f);
      original += (a[i] - b[i]) * (a[i] - b[i]);
    }
    rotation.rotate(a.data(), ra);
    rotation.rotate(b.data(), rb);
    reopened.rotate(a.data(), repeat);
    EXPECT_EQ(ra, repeat);
    double transformed = 0;
    for (size_t i = 0; i < ra.size(); ++i) {
      transformed += (ra[i] - rb[i]) * (ra[i] - rb[i]);
    }
    EXPECT_NEAR(original, transformed, 1e-5 * std::max(1.0, original));
  }
}


// Lock the symmetric lane ABI: [FP32 scale][D signed bytes], including
// half-away-from-zero rounding and zero-filled inactive lanes.
TEST(TurboBlockQuantizerTest, SQ8PreservesLaneBytesAndExactScores) {
  constexpr size_t dim = 5;
  const std::array<float, dim * 2> values{127.0f, -127.0f, 0.5f, -0.5f, 0.0f,
                                          0,      0,       0,    0,     0};
  const std::array<float, dim> query{1, -1, 2, -2, 0};
  for (bool cosine : {false, true}) {
    SQ8BlockQuantizer codec(dim, cosine);
    EXPECT_EQ(16U * (sizeof(float) + dim), codec.batch_bytes());
    std::vector<char> bytes(codec.batch_bytes() + 2, 42);
    codec.encode(nullptr, values.data(), 2, bytes.data() + 1);
    std::vector<char> expected(codec.batch_bytes(), 0);
    const float scale = 1.0f, zero_scale = std::numeric_limits<float>::min();
    const std::array<int8_t, dim> codes{127, -127, 1, -1, 0};
    std::memcpy(expected.data(), &scale, sizeof(scale));
    std::memcpy(expected.data() + sizeof(float), codes.data(), dim);
    std::memcpy(expected.data() + sizeof(float) + dim, &zero_scale,
                sizeof(zero_scale));
    EXPECT_EQ(0,
              std::memcmp(expected.data(), bytes.data() + 1, expected.size()));
    EXPECT_EQ(42, bytes.front());
    EXPECT_EQ(42, bytes.back());
    SQ8BlockQuantizer::Query prepared;
    codec.prepare_query(query.data(), prepared);
    std::array<float, 16> scores;
    codec.scan(bytes.data() + 1, prepared, 999, scores.data());
    EXPECT_EQ(cosine ? -257.0f : 31754.0f, scores[0]);
    for (size_t i = 1; i < scores.size(); ++i)
      EXPECT_EQ(cosine ? 1.0f : 10.0f, scores[i]);
  }
}

// Compare the block facade with the original PQ4 primitive, including every
// byte of padded lanes and query LUTs, not just nearest-neighbor identities.
TEST(TurboBlockQuantizerTest, PQPreservesPackedBytesLutsAndScores) {
  for (size_t dim : {1U, 17U, 128U}) {
    for (bool cosine : {false, true}) {
      SCOPED_TRACE(testing::Message() << dim << ":" << cosine);
      PQBlockQuantizer codec(dim, cosine);
      auto holder = std::make_shared<
          core::MultiPassIndexHolder<core::IndexMeta::DataType::DT_FP32>>(
          codec.encoded_dim());
      std::vector<float> transformed, values, raw(dim + 1);
      for (size_t i = 0; i < 32; ++i) {
        for (size_t d = 0; d < dim; ++d)
          raw[d] =
              i == 0 ? 0 : std::sin(float(i * 7 + d)) / std::sqrt(float(dim));
        raw[dim] =
            999;  // FP32 norm metadata must not enter the padded samples.
        codec.transform(raw.data(), transformed);
        for (size_t d = dim; d < transformed.size(); ++d)
          EXPECT_EQ(0, transformed[d]);
        ailego::NumericalVector<float> vector(codec.encoded_dim());
        std::copy(transformed.begin(), transformed.end(), vector.data());
        holder->emplace(i, std::move(vector));
        values.insert(values.end(), transformed.begin(), transformed.end());
      }
      ASSERT_EQ(0, codec.train(holder, 2));
      ASSERT_TRUE(codec.trained());
      // KMC2 initialization is intentionally random. Reuse the same immutable
      // codebook so this compares layouts and arithmetic, not training runs.
      const auto &reference = *codec.model();
      const size_t stride = reference.quantized_datapoint_vector_length();
      std::vector<char> plain(32 * stride);
      for (size_t i = 0; i < 32; ++i)
        reference.quantize_data(values.data() + i * codec.encoded_dim(),
                                plain.data() + i * stride);
      for (bool zero_query : {false, true}) {
        if (zero_query) std::fill(raw.begin(), raw.end(), 0);
        PQBlockQuantizer::Query actual_query;
        codec.prepare_query(raw.data(), actual_query);
        codec.transform(raw.data(), transformed);
        std::vector<char> expected_query(
            reference.quantized_query_vector_length());
        reference.quantize_query(transformed.data(), expected_query.data());
        EXPECT_EQ(expected_query, actual_query);
        for (size_t count : {0U, 1U, 17U, 31U, 32U}) {
          std::vector<char> actual(codec.batch_bytes() + 2, 42);
          std::vector<char> expected(codec.batch_bytes());
          codec.encode(nullptr, values.data(), count, actual.data() + 1);
          ASSERT_EQ(0, reference.pack_codes(plain.data(), count, stride,
                                            expected.data()));
          EXPECT_EQ(0, std::memcmp(expected.data(), actual.data() + 1,
                                   expected.size()));
          EXPECT_EQ(42, actual.front());
          EXPECT_EQ(42, actual.back());
          std::array<float, 32> scores, expected_scores;
          codec.scan(actual.data() + 1, actual_query, 999, scores.data());
          reference.calc_distance_packed_block(expected.data(), 32,
                                               expected_query.data(),
                                               expected_scores.data());
          if (cosine)
            for (auto &score : expected_scores) score += 1.0f;
          EXPECT_EQ(0, std::memcmp(scores.data(), expected_scores.data(),
                                   sizeof(scores)));
        }
      }
      codec.reset_training();
      EXPECT_FALSE(codec.trained());
      EXPECT_EQ(nullptr, codec.model());
    }
  }
}


TEST(TurboBlockQuantizerTest, PQRejectsIncompatibleTrainingSamples) {
  PQBlockQuantizer codec(17);
  EXPECT_EQ(core::IndexError_Mismatch, codec.train(nullptr, 1));
  auto wrong_dim = std::make_shared<
      core::MultiPassIndexHolder<core::IndexMeta::DataType::DT_FP32>>(17);
  auto empty = std::make_shared<
      core::MultiPassIndexHolder<core::IndexMeta::DataType::DT_FP32>>(24);
  auto wrong_type = std::make_shared<
      core::MultiPassIndexHolder<core::IndexMeta::DataType::DT_INT8>>(24);
  EXPECT_EQ(core::IndexError_Mismatch, codec.train(wrong_dim, 1));
  EXPECT_EQ(core::IndexError_Mismatch, codec.train(empty, 1));
  EXPECT_EQ(core::IndexError_Mismatch, codec.train(wrong_type, 1));
  EXPECT_FALSE(codec.trained());
}

TEST(TurboBlockQuantizerTest, ResidualPQLayoutAndReconstructionDistances) {
  for (size_t dim : {1U, 17U, 128U, 4096U}) {
    for (bool cosine : {false, true}) {
      SCOPED_TRACE(testing::Message() << dim << ":" << cosine);
      PQResidualBlockQuantizer codec(dim, cosine);
      const size_t padded = codec.encoded_dim();
      auto holder = std::make_shared<
          core::MultiPassIndexHolder<core::IndexMeta::DT_FP32>>(padded);
      std::vector<float> residuals(32 * padded, 0);
      for (size_t i = 0; i < 32; ++i) {
        ailego::NumericalVector<float> v(padded);
        for (size_t d = 0; d < padded; ++d) {
          const float value = d < dim ? float(int((i + d) % 16) - 8) /
                                            (8 * std::sqrt(float(dim)))
                                      : 0;
          v[d] = residuals[i * padded + d] = value;
        }
        holder->emplace(i, std::move(v));
      }
      ASSERT_EQ(0, codec.train(holder, 2));
      const auto &model = *codec.model();
      EXPECT_EQ("InnerProduct", model.meta().metric_name());
      const size_t packed_bytes = fast_scan_packed_block_size(padded / 8);
      EXPECT_EQ(packed_bytes + (cosine ? 0 : 128), codec.batch_bytes());
      const size_t stride = model.quantized_datapoint_vector_length();
      std::vector<char> plain(32 * stride);
      std::vector<float> center(padded, 0), vectors(32 * padded, 0),
          query(dim + 1), reconstructed(padded);
      for (size_t d = 0; d < dim; ++d)
        center[d] = std::cos(float(d)) / std::sqrt(float(dim));
      for (size_t i = 0; i < 32; ++i) {
        for (size_t d = 0; d < padded; ++d)
          vectors[i * padded + d] = center[d] + residuals[i * padded + d];
        // Use the actual FP32 subtraction done during encoding.
        for (size_t d = 0; d < padded; ++d)
          residuals[i * padded + d] = vectors[i * padded + d] - center[d];
        model.quantize_data(residuals.data() + i * padded,
                            plain.data() + i * stride);
      }
      PQResidualBlockQuantizer::Query prepared;
      for (bool zero_query : {false, true}) {
        for (size_t d = 0; d < dim; ++d)
          query[d] =
              zero_query ? 0 : std::sin(float(d + 1)) / std::sqrt(float(dim));
        query[dim] = 999;  // Norm metadata is not a coordinate.
        codec.prepare_query(query.data(), prepared);
        float delta;
        std::memcpy(&delta,
                    prepared.data() + prepared.size() - 2 * sizeof(float),
                    sizeof(float));
        float center_distance = cosine ? 1 : 0;
        for (size_t d = 0; d < dim; ++d) {
          const float diff = query[d] - center[d];
          center_distance += cosine ? -query[d] * center[d] : diff * diff;
        }
        for (size_t count : {0U, 1U, 17U, 31U, 32U}) {
          std::vector<char> codes(codec.batch_bytes() + 2, 42),
              expected(packed_bytes);
          codec.encode(center.data(), vectors.data(), count, codes.data() + 1);
          ASSERT_EQ(0, model.pack_codes(plain.data(), count, stride,
                                        expected.data()));
          EXPECT_EQ(
              0, std::memcmp(codes.data() + 1, expected.data(), packed_bytes));
          EXPECT_EQ(42, codes.front());
          EXPECT_EQ(42, codes.back());
          std::array<float, 32> scores;
          codec.scan(codes.data() + 1, prepared, center_distance,
                     scores.data());
          for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(0, model.reconstruct_plain_fp32(plain.data() + i * stride,
                                                      reconstructed.data()));
            double expected_distance = cosine ? 1 : 0, correction = 0;
            for (size_t d = 0; d < dim; ++d) {
              const double x = double(center[d]) + reconstructed[d];
              expected_distance += cosine ? -double(query[d]) * x
                                          : (query[d] - x) * (query[d] - x);
              const double r = reconstructed[d];
              correction += r * r + 2.0 * center[d] * r;
            }
            const double bound =
                (padded / 8) * delta * (cosine ? 0.5 : 1) + 5e-5;
            EXPECT_NEAR(expected_distance, scores[i], bound);
            if (!cosine) {
              float stored;
              std::memcpy(&stored,
                          codes.data() + 1 + packed_bytes + i * sizeof(float),
                          sizeof(float));
              EXPECT_FLOAT_EQ(float(correction), stored);
            }
            if (zero_query && cosine) EXPECT_FLOAT_EQ(1, scores[i]);
          }
        }
      }
    }
  }
}

TEST(TurboBlockQuantizerTest, ResidualPQSubtractsCenterAndExcludesMetadata) {
  PQResidualBlockQuantizer codec(3);
  const float center[]{100, -100, 2, 999}, neighbor[]{101, -102, 2.5f, -999};
  std::vector<float> residual;
  ASSERT_EQ(0, codec.transform_residual(center, neighbor, residual));
  EXPECT_EQ((std::vector<float>{1, -2, .5f, 0, 0, 0, 0, 0}), residual);
  float bad[]{INFINITY, 0, 0};
  EXPECT_EQ(core::IndexError_Mismatch,
            codec.transform_residual(center, bad, residual));
  const float huge[]{std::numeric_limits<float>::max(), 0, 0};
  const float negative[]{-std::numeric_limits<float>::max(), 0, 0};
  EXPECT_EQ(core::IndexError_Mismatch,
            codec.transform_residual(huge, negative, residual));
  PqFastQuantizer untrained;
  uint8_t code = 0;
  EXPECT_NE(0, untrained.reconstruct_plain_fp32(&code, residual.data()));
}

#if RABITQ_SUPPORTED
TEST(TurboSymphonyQGTest, CosineCodecMatchesL2ConversionIncludingZeroVectors) {
  if (!rabitqlib::cpu::has_avx2() && !rabitqlib::cpu::has_avx512_core())
    GTEST_SKIP();
  for (size_t dim : {1U, 128U, 4096U}) {
    SymphonyQGBlockQuantizer l2(dim), cosine(dim, true);
    const size_t padded = l2.encoded_dim();
    constexpr size_t count = 5;
    for (bool zero_center : {false, true}) {
      std::vector<float> center(dim + 1, 0);
      center[0] = zero_center ? 0 : 1;
      center[dim] = 1000;  // original norm, excluded by transform()
      std::vector<float> rotated_center, vectors(padded * count), scratch;
      l2.transform(center.data(), rotated_center);
      std::array<float, count> norms{};
      for (size_t i = 0; i < count; ++i) {
        std::vector<float> vector(dim + 1, 0);
        if (i != 0) vector[(i - 1) % dim] = i % 2 == 0 ? -1 : 1;
        vector[dim] = 17 * i;
        norms[i] = i == 0 ? 0 : 1;
        l2.transform(vector.data(), scratch);
        std::copy(scratch.begin(), scratch.end(), vectors.begin() + i * padded);
      }
      std::vector<char> l2_codes(l2.batch_bytes()),
          cos_codes(cosine.batch_bytes());
      l2.encode(rotated_center.data(), vectors.data(), count, l2_codes.data());
      cosine.encode(rotated_center.data(), vectors.data(), count,
                    cos_codes.data());
      for (float q : {-1.0f, 0.0f, 1.0f}) {
        std::vector<float> query(dim + 1, 0);
        query[0] = q;
        query[dim] = 9999;
        SymphonyQGBlockQuantizer::Query l2_query, cos_query;
        l2.prepare_query(query.data(), l2_query);
        cosine.prepare_query(query.data(), cos_query);
        std::array<float, SymphonyQGBlockQuantizer::kBatchSize> l2_dist,
            cos_dist;
        l2.scan(l2_codes.data(), l2_query, (q - center[0]) * (q - center[0]),
                l2_dist.data());
        cosine.scan(cos_codes.data(), cos_query, 1 - q * center[0],
                    cos_dist.data());
        for (size_t i = 0; i < count; ++i) {
          EXPECT_TRUE(std::isfinite(cos_dist[i]));
          EXPECT_NEAR(1 + 0.5f * (l2_dist[i] - q * q - norms[i]), cos_dist[i],
                      1e-5);
        }
      }
    }
  }
}

TEST(TurboSymphonyQGTest,
     FastScanHandlesPartialBlocksDuplicatesAndLargeDimensions) {
  if (!rabitqlib::cpu::has_avx2() && !rabitqlib::cpu::has_avx512_core()) {
    GTEST_SKIP() << "RaBitQ requires AVX2/FMA or AVX512";
  }
  for (size_t dim : {64U, 128U, 1024U, 2048U, 4096U}) {
    for (size_t count : {1U, 17U, 32U}) {
      SCOPED_TRACE(std::to_string(dim) + "/" + std::to_string(count));
      // Constant positive query deliberately produces large LUT sums at 4096D.
      std::vector<float> query(dim, 1.0f), center(dim, 0.5f);
      std::vector<float> data(count * dim);
      for (size_t i = 0; i < count; ++i) {
        std::fill(data.begin() + i * dim, data.begin() + (i + 1) * dim,
                  0.5f + i * 0.125f);
      }
      std::vector<char> codes(rabitqlib::QGBatchDataMap<float>::data_bytes(dim),
                              0);
      rabitqlib::quant::quantize_qg_batch(data.data(), center.data(), count,
                                          dim, codes.data(),
                                          rabitqlib::METRIC_L2);
      SymQuery q;
      q.reset(query.data(), dim);
      q.set_g_add(0.25f * dim);
      std::array<float, 32> result;
      ScanSymphonyQGBatch(codes.data(), q, dim, result.data());
      for (size_t i = 0; i < count; ++i) {
        const float diff = 0.5f + i * 0.125f - 1.0f;
        EXPECT_NEAR(diff * diff * dim, result[i], 0.01f * dim);
        EXPECT_TRUE(std::isfinite(result[i]));
      }
    }
  }
}

TEST(TurboSymphonyQGTest,
     FusedScanMatchesReferenceAcrossSlicesAndUnalignedInputs) {
  if (!rabitqlib::cpu::has_avx512_core()) GTEST_SKIP();
  for (size_t dim : {16U, 64U, 1024U, 1040U, 2048U, 4096U}) {
    for (bool maximum_lut : {false, true}) {
      SCOPED_TRACE(std::to_string(dim) + "/" + std::to_string(maximum_lut));
      // Deliberately misalign both byte inputs. A maximal LUT exercises
      // unsigned 16-bit sums and widening beyond the first 1024-D slice.
      std::vector<uint8_t> codes(dim * 4 + 1), lut(dim * 4 + 1);
      for (size_t i = 1; i < codes.size(); ++i) {
        codes[i] = static_cast<uint8_t>(i * 79 + 13);
        lut[i] = maximum_lut ? 255 : static_cast<uint8_t>(i * 37 + 5);
      }
      std::array<uint32_t, 32> sums{};
      using Accumulator = std::conditional_t<
          std::is_invocable_v<decltype(&rabitqlib::fastscan::accumulate),
                              const uint8_t *, const uint8_t *, int32_t *,
                              size_t>,
          int32_t, uint16_t>;
      std::array<Accumulator, 32> partial;
      for (size_t offset = 0; offset < dim; offset += 1024) {
        rabitqlib::fastscan::accumulate(
            codes.data() + 1 + offset * 4, lut.data() + 1 + offset * 4,
            partial.data(), std::min(size_t{1024}, dim - offset));
        for (size_t i = 0; i < 32; ++i) sums[i] += partial[i];
      }
      std::array<float, 33> add, scale, actual;
      for (size_t i = 0; i < 33; ++i) {
        add[i] = static_cast<float>(i) * 0.713f - 7.91f;
        scale[i] = 0.331f - static_cast<float>(i) * 0.087f;
        actual[i] = -123.0f;
      }
      if (!symqg_scan(codes.data() + 1, lut.data() + 1, dim, add.data() + 1,
                      scale.data() + 1, 3.13f, 0.0173f, -1.237f, 2.731f,
                      actual.data() + 1)) {
        GTEST_SKIP() << "SIMD kernels were not compiled for this platform";
      }
      EXPECT_EQ(-123.0f, actual[0]);
      for (size_t i = 0; i < 32; ++i) {
        // Explicit rounding boundaries make this independent of the test
        // compiler's FMA settings and protect the generic path's semantics.
        volatile float product = 0.0173f * sums[i];
        volatile float shifted = product - 1.237f;
        volatile float corrected = shifted + 2.731f;
        volatile float scaled = scale[i + 1] * corrected;
        volatile float base = add[i + 1] + 3.13f;
        EXPECT_EQ(base + scaled, actual[i + 1]);
      }
    }
  }
}

TEST(TurboSymphonyQGTest, FusedScanRejectsUnsupportedLengthsWithoutWriting) {
  std::array<uint8_t, 64> bytes{};
  std::array<float, 32> factors{}, output;
  output.fill(123.0f);
  for (size_t dim : {0U, 1U, 15U, 17U}) {
    EXPECT_FALSE(symqg_scan(bytes.data(), bytes.data(), dim, factors.data(),
                            factors.data(), 0, 0, 0, 0, output.data()));
    for (float value : output) EXPECT_EQ(123.0f, value);
  }
}

TEST(TurboSymphonyQGTest, PackedFactorsSupportUnalignedBatchStorage) {
  constexpr size_t dim = 64, count = 17;
  std::vector<float> center(dim, 0.25f), data(count * dim), query(dim, 0.5f);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(i % 29) / 16.0f;
  }
  const size_t bytes = rabitqlib::QGBatchDataMap<float>::data_bytes(dim);
  std::vector<char> aligned(bytes, 0), unaligned(bytes + 1, 0);
  rabitqlib::quant::quantize_qg_batch(data.data(), center.data(), count, dim,
                                      aligned.data(), rabitqlib::METRIC_L2);
  // Shift the complete encoded batch, including both FP32 factor arrays.
  std::copy(aligned.begin(), aligned.end(), unaligned.begin() + 1);
  SymQuery prepared;
  prepared.reset(query.data(), dim);
  prepared.set_g_add(4.0f);
  std::array<float, 32> expected, actual;
  ScanSymphonyQGBatch(aligned.data(), prepared, dim, expected.data());
  ScanSymphonyQGBatch(unaligned.data() + 1, prepared, dim, actual.data());
  for (size_t i = 0; i < count; ++i) {
    EXPECT_TRUE(std::isfinite(actual[i]));
    EXPECT_EQ(expected[i], actual[i]);
  }
}

TEST(TurboSymphonyQGTest, ReusedQueryMatchesReferenceLut) {
  if (!rabitqlib::cpu::has_avx2() && !rabitqlib::cpu::has_avx512_core()) {
    GTEST_SKIP() << "RaBitQ requires AVX2/FMA or AVX512";
  }
  SymQuery actual;
  for (size_t dim : {64U, 128U, 1024U, 2048U, 4096U, 64U}) {
    for (int pattern : {0, 1, 2}) {
      SCOPED_TRACE(std::to_string(dim) + "/" + std::to_string(pattern));
      std::vector<float> query(dim, 0.0f);
      for (size_t i = 0; i < dim; ++i) {
        if (pattern == 1) query[i] = 1.0f;
        // Exactly representable mixed signs keep subset sums reproducible.
        if (pattern == 2)
          query[i] = (static_cast<int>(i * 37 % 101) - 50) / 16.0f;
      }
      actual.set_g_add(123.0f);
      actual.reset(query.data(), dim);
      rabitqlib::BatchQuery<float> expected(query.data(), dim);
      EXPECT_FLOAT_EQ(0.0f, actual.g_add());
      EXPECT_FLOAT_EQ(expected.delta(), actual.delta());
      EXPECT_FLOAT_EQ(expected.sum_vl_lut(), actual.sum_vl_lut());
      EXPECT_FLOAT_EQ(expected.k1xsumq(), actual.k1xsumq());
      for (size_t i = 0; i < dim * 4; ++i) {
        // SIMD and reference rounding can differ by one quantization step.
        EXPECT_LE(std::abs(int(expected.lut()[i]) - int(actual.lut()[i])), 1);
      }
    }
  }
}

#endif

}  // namespace
}  // namespace zvec::turbo
