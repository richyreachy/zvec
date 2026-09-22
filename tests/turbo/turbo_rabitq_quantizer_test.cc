// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <cmath>
#include <cstring>
#include <random>
#include <gtest/gtest.h>
#include <turbo/quantizer/rabitq_quantizer/rabitq_quantizer.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec::turbo {
namespace {

TEST(RabitqQuantizer, BitsMetricsAsymmetricEncodingAndPersistence) {
  constexpr int dim =
      73;  // non-power-of-two rotation and non-byte-aligned codes
  std::mt19937 rng(123);
  std::normal_distribution<float> random;
  std::vector<float> x(dim), q(dim), zero(dim, 0);
  for (int i = 0; i < dim; ++i) {
    x[i] = random(rng);
    q[i] = random(rng);
  }
  for (const char *metric : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
    for (int bits = 1; bits <= 8; ++bits) {
      SCOPED_TRACE(metric);
      SCOPED_TRACE(bits);
      IndexMeta meta(IndexMeta::DT_FP32, dim);
      meta.set_metric(metric, 0, ailego::Params{});
      ailego::Params params;
      params.set(RABITQ_TOTAL_BITS, bits);
      auto quantizer = IndexFactory::CreateQuantizer("RabitqQuantizer");
      ASSERT_NE(nullptr, quantizer);
      ASSERT_EQ(0, quantizer->init(meta, params));
      // Use the same reproducible rotation across bit counts and metrics.
      // Production initialization intentionally draws fresh random signs.
      std::string initial_state;
      ASSERT_EQ(0, quantizer->serialize(&initial_state));
      std::mt19937 signs(42);
      const size_t signs_offset = sizeof(QuantizerSerHeader) +
                                  sizeof(uint32_t) + sizeof(RotatorSerHeader);
      for (size_t i = signs_offset; i < initial_state.size(); ++i)
        initial_state[i] = static_cast<char>(signs() & 0xff);
      ASSERT_EQ(0, quantizer->deserialize(initial_state));
      ASSERT_TRUE(quantizer->requires_original_vectors());
      ASSERT_FALSE(quantizer->require_train());
      IndexQueryMeta input(IndexMeta::DT_FP32, dim), dp_meta, query_meta;
      std::string dp, query, self, z;
      ASSERT_EQ(0,
                quantizer->quantize_datapoint(x.data(), input, &dp, &dp_meta));
      ASSERT_EQ(12u + (dim * bits + 7) / 8, dp.size());
      ASSERT_EQ(dp.size(), quantizer->meta().element_size());
      ASSERT_EQ(dp.size(), dp_meta.element_size());
      ASSERT_EQ(0, quantizer->quantize(q.data(), input, &query, &query_meta));
      ASSERT_EQ((dim + 1) * sizeof(float), query.size());
      ASSERT_EQ(query.size(), query_meta.element_size());
      ASSERT_EQ(0, quantizer->quantize(x.data(), input, &self, &query_meta));
      ASSERT_EQ(
          0, quantizer->quantize_datapoint(zero.data(), input, &z, &dp_meta));
      const float actual =
          quantizer->calc_distance_dp_query(dp.data(), query.data());
      EXPECT_TRUE(std::isfinite(actual));
      EXPECT_FLOAT_EQ(actual, quantizer->calc_distance_dp_query_unquantized(
                                  dp.data(), q.data()));
      const void *points[] = {dp.data(), z.data()};
      float batch[2], raw_batch[2];
      quantizer->calc_distance_dp_query_batch(points, 2, query.data(), batch);
      quantizer->calc_distance_dp_query_batch_unquantized(points, 2, q.data(),
                                                          raw_batch);
      EXPECT_FLOAT_EQ(actual, batch[0]);
      EXPECT_FLOAT_EQ(batch[0], raw_batch[0]);
      EXPECT_FLOAT_EQ(batch[1], raw_batch[1]);
      float norm2 = 0, query_norm2 = 0;
      for (int i = 0; i < dim; ++i) {
        norm2 += x[i] * x[i];
        query_norm2 += q[i] * q[i];
      }
      const bool l2 = std::strcmp(metric, "SquaredEuclidean") == 0;
      const bool cosine = std::strcmp(metric, "Cosine") == 0;
      EXPECT_NEAR(l2 || cosine ? 0 : -norm2,
                  quantizer->calc_distance_dp_query(dp.data(), self.data()),
                  1e-3f);
      EXPECT_NEAR(l2 ? query_norm2 : (cosine ? 1 : 0), batch[1], 1e-3f);
      std::string reconstructed;
      ASSERT_EQ(0, quantizer->dequantize(dp.data(), dp_meta, &reconstructed));
      double error2 = 0;
      for (int i = 0; i < dim; ++i) {
        float v;
        std::memcpy(&v, reconstructed.data() + i * sizeof(float), sizeof(v));
        EXPECT_TRUE(std::isfinite(v));
        error2 += (v - x[i]) * (v - x[i]);
      }
      if (bits >= 7) EXPECT_LT(error2 / norm2, 0.01);
      ASSERT_EQ(0, quantizer->dequantize(z.data(), dp_meta, &reconstructed));
      for (int i = 0; i < dim; ++i) {
        float value;
        std::memcpy(&value, reconstructed.data() + i * sizeof(float),
                    sizeof(value));
        EXPECT_FLOAT_EQ(0.0f, value);  // Rotation may preserve a negative zero.
      }
      std::string saved;
      ASSERT_EQ(0, quantizer->serialize(&saved));
      RabitqQuantizer restored;
      ASSERT_EQ(0, restored.init(meta, params));
      // Both the raw-buffer and string restoration entrypoints are supported.
      ASSERT_EQ(0, restored.deserialize(saved));
      std::string encoded, restored_query;
      ASSERT_EQ(
          0, restored.quantize_datapoint(x.data(), input, &encoded, &dp_meta));
      ASSERT_EQ(
          0, restored.quantize(q.data(), input, &restored_query, &query_meta));
      EXPECT_EQ(dp, encoded);
      EXPECT_EQ(query, restored_query);
      EXPECT_FLOAT_EQ(actual, restored.calc_distance_dp_query(
                                  dp.data(), restored_query.data()));
      EXPECT_NE(0, restored.deserialize(saved.data(), saved.size() - 1));
      saved[0] ^= 1;
      EXPECT_NE(0, restored.deserialize(saved));
      EXPECT_FLOAT_EQ(actual,
                      restored.calc_distance_dp_query(dp.data(), query.data()));
    }
  }
}

TEST(RabitqQuantizer, RejectsInvalidConfigurationAndMismatchedState) {
  IndexMeta meta(IndexMeta::DT_FP32, 64);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  ailego::Params params;
  RabitqQuantizer q;
  for (int bits : {-1, 0, 9}) {
    params.set(RABITQ_TOTAL_BITS, bits);
    EXPECT_NE(0, q.init(meta, params));
  }
  params.set(RABITQ_TOTAL_BITS, 7);
  ASSERT_EQ(0, q.init(meta, params));
  std::string state;
  ASSERT_EQ(0, q.serialize(&state));
  params.set(RABITQ_TOTAL_BITS, 8);
  ASSERT_EQ(0, q.init(meta, params));
  EXPECT_NE(0, q.deserialize(state));
  meta.set_dimension(0);
  EXPECT_NE(0, q.init(meta, params));
  meta.set_dimension(4096);
  EXPECT_NE(0, q.init(meta, params));
  meta.set_dimension(64);
  meta.set_metric("MipsSquaredEuclidean", 0, ailego::Params{});
  EXPECT_NE(0, q.init(meta, params));
}

}  // namespace
}  // namespace zvec::turbo
