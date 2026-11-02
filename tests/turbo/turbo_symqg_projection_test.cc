// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/symqg_quantizer/symqg_projection.h>

namespace zvec::turbo::research {
TEST(SymQGProjection, FullSpaceRadialConstraintAndMetric) {
  constexpr size_t full = 8, projected = 4, rank = 2;
  const std::array<double, full> residual{1, .3, -.5, .2, 1.2, -.4, .1, -.7};
  std::array<double, full * rank> basis;
  for (size_t i = 0; i < basis.size(); ++i) basis[i] = .07 * (int(i % 7) - 3);
  const double levels[]{-1.5104, -.4528, .4528, 1.5104};
  std::array<double, 32> words;
  for (size_t j = 0; j < 16; ++j) {
    words[2 * j] = levels[j % 4];
    words[2 * j + 1] = levels[j / 4];
  }
  double norm = 0;
  std::array<double, rank> metric{};
  for (size_t i = 0; i < full; ++i) {
    norm += residual[i] * residual[i];
    for (size_t k = 0; k < rank; ++k)
      metric[k] += residual[i] * basis[i * rank + k];
  }
  std::array<uint8_t, projected / 2> labels;
  std::array<double, projected> errors;
  std::array<double, 4> stats;
  double beta = 0;
  ASSERT_EQ(0, optimize_projected_radial(
                   1, projected, 2, rank, words.data(), basis.data(), .1, 1.25,
                   2, residual.data(), &norm, metric.data(), labels.data(),
                   &beta, errors.data(), stats.data()));
  double radial = 0, error_norm = 0;
  std::array<double, rank> projected_error{};
  for (size_t i = 0; i < full; ++i) {
    const double approx =
        i < projected ? beta * words[2 * labels[i / 2] + i % 2] : 0;
    const double error = approx - residual[i];
    radial += residual[i] * error;
    error_norm += error * error;
    for (size_t k = 0; k < rank; ++k)
      projected_error[k] += error * basis[i * rank + k];
  }
  EXPECT_NEAR(0., radial, 1e-12);
  EXPECT_NEAR(error_norm, stats[1], 1e-12);
  double expected_loss = .1 * error_norm;
  for (double value : projected_error) expected_loss += value * value;
  EXPECT_NEAR(expected_loss, stats[3], 1e-12);
  EXPECT_LE(stats[1], 1.25 * stats[0] + 1e-12);
  EXPECT_LE(stats[3], stats[2] + 1e-12);
}

TEST(SymQGProjection, IntegerProjectionMatchesScalarAndErrorEnvelope) {
  constexpr size_t columns = 128, rows = 8;
  std::mt19937 random(20261102);
  std::uniform_real_distribution<float> value(-2, 2);
  std::vector<float> matrix(rows * columns), query(columns), out;
  std::vector<uint8_t> scratch;
  for (float &x : matrix) x = value(random);
  std::fill(matrix.begin(), matrix.begin() + columns, 0.f);
  SymQGIntegerProjection projection;
  EXPECT_FALSE(projection.project(query.data(), out, scratch));
  EXPECT_THROW(projection.prepare({}, rows, columns), std::invalid_argument);
  projection.prepare(matrix, rows, columns);
  for (size_t trial = 0; trial < 5; ++trial) {
    for (float &x : query) x = trial == 0 ? 0 : trial == 1 ? 2 : value(random);
    if (!projection.project(query.data(), out, scratch)) GTEST_SKIP();
    const float minimum = *std::min_element(query.begin(), query.end());
    const float maximum = *std::max_element(query.begin(), query.end());
    const float step = maximum > minimum ? (maximum - minimum) / 255.f : 1.f;
    for (size_t i = 0; i < rows; ++i) {
      float magnitude = 0;
      for (size_t j = 0; j < columns; ++j)
        magnitude = std::max(magnitude, std::abs(matrix[i * columns + j]));
      const float scale = magnitude > 0 ? magnitude / 127.f : 1.f;
      int32_t sum = 0, dot = 0;
      double exact = 0, bound = 0;
      for (size_t j = 0; j < columns; ++j) {
        const int weight = std::clamp(
            int(std::nearbyint(matrix[i * columns + j] / scale)), -127, 127);
        const int quantized = std::clamp(
            int(std::nearbyint((query[j] - minimum) / step)), 0, 255);
        EXPECT_EQ(quantized, scratch[j]);
        sum += weight;
        dot += weight * quantized;
        exact += double(matrix[i * columns + j]) * query[j];
        const double decoded = double(scale) * weight;
        bound +=
            .5 * step * std::abs(decoded) +
            std::abs(decoded - matrix[i * columns + j]) * std::abs(query[j]);
      }
      // Target-specific FMA and scalar arithmetic may round differently.
      const double roundoff =
          4 * std::numeric_limits<float>::epsilon() * scale *
          (std::abs(double(step) * dot) + std::abs(double(minimum) * sum));
      EXPECT_NEAR(scale * (step * dot + minimum * sum), out[i], roundoff);
      EXPECT_LE(std::abs(out[i] - exact), bound + 1e-4);
    }
  }
}

TEST(SymQGProjection, RejectsMalformedAndOversizedModels) {
  auto header = [](uint32_t full, uint32_t projected) {
    const uint32_t fields[]{0x50524a31, full, projected, 0, 2};
    const double params[]{.1, 1.25};
    return std::string(reinterpret_cast<const char *>(fields), sizeof(fields)) +
           std::string(reinterpret_cast<const char *>(params), sizeof(params));
  };
  for (uint32_t projected : {0, 63, 256, 128}) {
    auto data = header(128, projected);
    std::istringstream stream(data);
    SymQGProjection codec;
    EXPECT_THROW(codec.load(stream, 128), std::runtime_error);
  }
  std::istringstream huge(header(65536, 65536));
  SymQGProjection codec;
  EXPECT_THROW(codec.load(huge, 65536), std::runtime_error);
}
}  // namespace zvec::turbo::research
