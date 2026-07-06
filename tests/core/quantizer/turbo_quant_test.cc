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

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_holder.h>
#include "core/quantizer/turbo_quant_codebook.h"
#include "core/quantizer/turbo_quant_engine.h"
#include "core/quantizer/turbo_quant_params.h"

using namespace zvec;
using namespace zvec::core;

// ---------------------------------------------------------------------------
// Codebook: verify centroids match paper's theoretical values
// ---------------------------------------------------------------------------
TEST(TurboQuantCodebook, CentroidsBit1) {
  // Paper: for b=1, centroids ≈ ±sqrt(2/(pi*d))
  const size_t d = 1024;
  auto cb = TurboQuantCodebook::get(d, 1);
  ASSERT_EQ(2u, cb->num_centroids());
  const auto &c = cb->centroids();
  ASSERT_EQ(2u, c.size());

  float expected =
      std::sqrt(2.0f / static_cast<float>(M_PI) / static_cast<float>(d));
  // Centroids should be approximately ±expected
  ASSERT_LT(c[0], 0.0f);
  ASSERT_GT(c[1], 0.0f);
  EXPECT_NEAR(std::abs(c[0]), expected, expected * 0.1f);
  EXPECT_NEAR(std::abs(c[1]), expected, expected * 0.1f);
}

TEST(TurboQuantCodebook, CentroidsBit2) {
  // Paper: for b=2, centroids ≈ ±0.453/sqrt(d), ±1.51/sqrt(d)
  const size_t d = 1024;
  auto cb = TurboQuantCodebook::get(d, 2);
  ASSERT_EQ(4u, cb->num_centroids());
  const auto &c = cb->centroids();
  ASSERT_EQ(4u, c.size());

  float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
  float expected_inner = 0.453f * inv_sqrt_d;
  float expected_outer = 1.51f * inv_sqrt_d;

  // Centroids should be sorted: [-outer, -inner, +inner, +outer]
  ASSERT_LT(c[0], c[1]);
  ASSERT_LT(c[1], c[2]);
  ASSERT_LT(c[2], c[3]);

  EXPECT_NEAR(std::abs(c[0]), expected_outer, expected_outer * 0.15f);
  EXPECT_NEAR(std::abs(c[1]), expected_inner, expected_inner * 0.15f);
  EXPECT_NEAR(std::abs(c[2]), expected_inner, expected_inner * 0.15f);
  EXPECT_NEAR(std::abs(c[3]), expected_outer, expected_outer * 0.15f);
}

TEST(TurboQuantCodebook, QuantizeDequantize) {
  auto cb = TurboQuantCodebook::get(256, 4);
  // Dequantize should return the centroid
  for (size_t i = 0; i < cb->num_centroids(); ++i) {
    EXPECT_EQ(cb->centroids()[i], cb->dequantize(i));
  }
  // Quantize then dequantize should be idempotent
  for (int trial = 0; trial < 100; ++trial) {
    float x = static_cast<float>(trial - 50) / 100.0f;
    uint32_t idx = cb->quantize(x);
    float dq = cb->dequantize(idx);
    // The dequantized value should be the nearest centroid
    float min_dist = std::abs(x - dq);
    for (size_t i = 0; i < cb->num_centroids(); ++i) {
      float d = std::abs(x - cb->centroids()[i]);
      EXPECT_LE(min_dist, d + 1e-6f);
    }
  }
}

// ---------------------------------------------------------------------------
// Bit packing: verify pack/unpack round-trip
// ---------------------------------------------------------------------------
TEST(TurboQuantPacking, RoundTrip) {
  for (int bits : {1, 2, 3, 4, 8}) {
    size_t count = 100;
    std::vector<uint32_t> original(count);
    std::mt19937 rng(42);
    uint32_t mask = (1u << bits) - 1u;
    for (size_t i = 0; i < count; ++i) {
      original[i] = rng() & mask;
    }
    size_t bytes = TurboQuantPacking::packed_bytes(count, bits);
    std::vector<uint8_t> packed(bytes, 0);
    TurboQuantPacking::pack(original.data(), count, bits, packed.data());

    std::vector<uint32_t> unpacked(count, 0);
    TurboQuantPacking::unpack(packed.data(), count, bits, unpacked.data());

    for (size_t i = 0; i < count; ++i) {
      EXPECT_EQ(original[i], unpacked[i]) << "bits=" << bits << " index=" << i;
    }
  }
}

// ---------------------------------------------------------------------------
// MSE mode: quantize/dequantize round-trip and distortion bounds
// ---------------------------------------------------------------------------
TEST(TurboQuantEngine, MseDistortion) {
  const size_t d = 512;
  const int bits = 4;
  const size_t N = 2000;

  TurboQuantEngine engine(d, bits, false /*mse*/, true /*rotate*/, 42);

  std::mt19937 rng(123);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  double total_sq_error = 0.0;
  size_t total_count = 0;

  std::vector<uint8_t> quantized(engine.total_bytes(), 0);
  std::vector<float> dequant(d, 0.0f);

  for (size_t i = 0; i < N; ++i) {
    // Generate random unit-norm vector
    std::vector<float> vec(d);
    float norm = 0.0f;
    for (size_t j = 0; j < d; ++j) {
      vec[j] = dist(rng);
      norm += vec[j] * vec[j];
    }
    norm = std::sqrt(norm);
    for (size_t j = 0; j < d; ++j) {
      vec[j] /= norm;
    }

    engine.quantize(vec.data(), quantized.data());
    engine.dequantize(quantized.data(), dequant.data());

    for (size_t j = 0; j < d; ++j) {
      float diff = vec[j] - dequant[j];
      total_sq_error += diff * diff;
    }
    total_count++;
  }

  double mse = total_sq_error / (total_count * d);
  // Paper: b=4 MSE ≈ 0.009 (per coordinate, for unit-norm vectors)
  // Allow generous tolerance for finite-dimension effects
  EXPECT_LT(mse, 0.05) << "MSE too high for b=" << bits << ", d=" << d;
  EXPECT_GT(mse, 0.0) << "MSE should be positive";
}

TEST(TurboQuantEngine, MseDistortionBit1) {
  const size_t d = 512;
  const int bits = 1;
  const size_t N = 2000;

  TurboQuantEngine engine(d, bits, false, true, 42);

  std::mt19937 rng(456);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  double total_sq_error = 0.0;
  std::vector<uint8_t> quantized(engine.total_bytes(), 0);
  std::vector<float> dequant(d, 0.0f);

  for (size_t i = 0; i < N; ++i) {
    std::vector<float> vec(d);
    float norm = 0.0f;
    for (size_t j = 0; j < d; ++j) {
      vec[j] = dist(rng);
      norm += vec[j] * vec[j];
    }
    norm = std::sqrt(norm);
    for (size_t j = 0; j < d; ++j) vec[j] /= norm;

    engine.quantize(vec.data(), quantized.data());
    engine.dequantize(quantized.data(), dequant.data());

    for (size_t j = 0; j < d; ++j) {
      float diff = vec[j] - dequant[j];
      total_sq_error += diff * diff;
    }
  }

  double mse = total_sq_error / (N * d);
  // Paper: b=1 MSE ≈ 0.36 (but this is total ||x - x̃||^2, not per-coordinate)
  // Actually the paper says D_mse = E[||x - x̃||^2] ≈ 0.36 for b=1
  // Our mse is per-coordinate, so total = mse * d
  double total_mse = mse * d;
  EXPECT_LT(total_mse, 0.6) << "Total MSE too high for b=1";
  EXPECT_GT(total_mse, 0.1) << "Total MSE suspiciously low for b=1";
}

// ---------------------------------------------------------------------------
// Prod mode: inner product unbiasedness
// ---------------------------------------------------------------------------
TEST(TurboQuantEngine, ProdUnbiasedness) {
  const size_t d = 512;
  const int bits = 3;  // Prod with b=3: MSE uses 2 bits, QJL uses 1 bit
  const size_t N = 3000;

  TurboQuantEngine engine(d, bits, true /*prod*/, true /*rotate*/, 42);

  std::mt19937 rng(789);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  // Generate random unit-norm vectors x and y
  auto gen_unit_vec = [&]() {
    std::vector<float> v(d);
    float n = 0.0f;
    for (size_t j = 0; j < d; ++j) {
      v[j] = dist(rng);
      n += v[j] * v[j];
    }
    n = std::sqrt(n);
    for (size_t j = 0; j < d; ++j) v[j] /= n;
    return v;
  };

  double total_bias = 0.0;
  double total_sq_error = 0.0;
  std::vector<uint8_t> quantized(engine.total_bytes(), 0);
  std::vector<float> dequant(d, 0.0f);

  for (size_t i = 0; i < N; ++i) {
    auto x = gen_unit_vec();
    auto y = gen_unit_vec();

    float true_ip = 0.0f;
    for (size_t j = 0; j < d; ++j) {
      true_ip += x[j] * y[j];
    }

    engine.quantize(x.data(), quantized.data());
    engine.dequantize(quantized.data(), dequant.data());

    float est_ip = 0.0f;
    for (size_t j = 0; j < d; ++j) {
      est_ip += y[j] * dequant[j];
    }

    total_bias += (est_ip - true_ip);
    total_sq_error += (est_ip - true_ip) * (est_ip - true_ip);
  }

  double mean_bias = total_bias / N;
  double mse = total_sq_error / N;

  // Unbiasedness: mean bias should be close to 0
  // For d=512, the std of the bias should be small
  EXPECT_NEAR(mean_bias, 0.0, 0.02) << "Inner product estimator is biased";

  // Distortion: D_prod ≈ 0.18/d for b=3
  // So MSE should be approximately 0.18/512 ≈ 0.00035
  // Allow generous tolerance
  EXPECT_LT(mse, 0.01) << "Inner product distortion too high";
}

// ---------------------------------------------------------------------------
// Converter + Reformer integration
// ---------------------------------------------------------------------------
TEST(TurboQuantConverter, InitTransformMSE) {
  const size_t DIM = 128;
  const size_t COUNT = 100;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIM);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());

  ailego::Params params;
  params.set(TURBO_QUANT_CONVERTER_BIT_WIDTH, 4);
  params.set(TURBO_QUANT_CONVERTER_MODE, TURBO_QUANT_MODE_MSE);

  auto converter = IndexFactory::CreateConverter("TurboQuantConverter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0u, converter->init(meta, params));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(DIM);
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIM);
    float n = 0.0f;
    for (size_t j = 0; j < DIM; ++j) {
      vec[j] = dist(rng);
      n += vec[j] * vec[j];
    }
    n = std::sqrt(n);
    for (size_t j = 0; j < DIM; ++j) vec[j] /= n;
    holder->emplace(i + 1, vec);
  }

  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));

  auto result = converter->result();
  ASSERT_TRUE(result);
  EXPECT_EQ(COUNT, result->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT8, result->data_type());

  // Verify quantized data is non-empty
  auto iter = result->create_iterator();
  ASSERT_TRUE(iter->is_valid());
  EXPECT_EQ(result->element_size(), result->dimension());
  EXPECT_GT(result->element_size(), 0u);

  // Create reformer and verify transform produces matching output
  auto reformer = IndexFactory::CreateReformer("TurboQuantReformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(converter->meta().reformer_params()));

  // Transform a query and verify output size matches
  std::string out;
  IndexQueryMeta ometa;
  const float *query =
      reinterpret_cast<const float *>(holder->create_iterator()->data());
  EXPECT_EQ(0, reformer->transform(
                   query, IndexQueryMeta(IndexMeta::DataType::DT_FP32, DIM),
                   &out, &ometa));
  EXPECT_EQ(result->element_size(), out.size());

  // Revert (dequantize) and check output size
  std::string reverted;
  EXPECT_EQ(0, reformer->revert(out.data(),
                                IndexQueryMeta(IndexMeta::DataType::DT_INT8,
                                               result->dimension()),
                                &reverted));
  EXPECT_EQ(DIM * sizeof(float), reverted.size());
}

TEST(TurboQuantConverter, InitTransformProd) {
  const size_t DIM = 128;
  const size_t COUNT = 50;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIM);
  meta.set_metric("InnerProduct", 0, ailego::Params());

  ailego::Params params;
  params.set(TURBO_QUANT_CONVERTER_BIT_WIDTH, 3);
  params.set(TURBO_QUANT_CONVERTER_MODE, TURBO_QUANT_MODE_PROD);

  auto converter = IndexFactory::CreateConverter("TurboQuantConverter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0u, converter->init(meta, params));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(DIM);
  std::mt19937 rng(99);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIM);
    float n = 0.0f;
    for (size_t j = 0; j < DIM; ++j) {
      vec[j] = dist(rng);
      n += vec[j] * vec[j];
    }
    n = std::sqrt(n);
    for (size_t j = 0; j < DIM; ++j) vec[j] /= n;
    holder->emplace(i + 1, vec);
  }

  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));

  auto result = converter->result();
  ASSERT_TRUE(result);
  EXPECT_EQ(COUNT, result->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT8, result->data_type());
  // Prod mode uses more bytes (MSE indices + QJL signs + metadata)
  EXPECT_GT(result->element_size(), 0u);

  // Verify reformer can be created
  auto reformer = IndexFactory::CreateReformer("TurboQuantReformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(converter->meta().reformer_params()));
}

// ---------------------------------------------------------------------------
// Converter/Reformer: verify transform produces matching quantized output
// ---------------------------------------------------------------------------
TEST(TurboQuantConverter, ReformerMatchesConverter) {
  const size_t DIM = 64;
  const size_t COUNT = 20;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIM);

  ailego::Params params;
  params.set(TURBO_QUANT_CONVERTER_BIT_WIDTH, 4);
  params.set(TURBO_QUANT_CONVERTER_MODE, TURBO_QUANT_MODE_MSE);
  params.set(TURBO_QUANT_CONVERTER_ENABLE_ROTATE, false);

  auto converter = IndexFactory::CreateConverter("TurboQuantConverter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0u, converter->init(meta, params));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(DIM);
  std::mt19937 rng(77);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIM);
    for (size_t j = 0; j < DIM; ++j) {
      vec[j] = dist(rng);
    }
    holder->emplace(i + 1, vec);
  }

  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto result = converter->result();

  auto reformer = IndexFactory::CreateReformer("TurboQuantReformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(converter->meta().reformer_params()));

  // For each original vector, converter and reformer should produce
  // identical quantized output (same rotator, same codebook, same seed)
  auto orig_iter = holder->create_iterator();
  auto quant_iter = result->create_iterator();
  std::string reformer_out;
  IndexQueryMeta ometa;

  for (; orig_iter->is_valid(); orig_iter->next(), quant_iter->next()) {
    ASSERT_TRUE(quant_iter->is_valid());
    EXPECT_EQ(0, reformer->transform(
                     orig_iter->data(),
                     IndexQueryMeta(IndexMeta::DataType::DT_FP32, DIM),
                     &reformer_out, &ometa));

    std::string converter_out(
        reinterpret_cast<const char *>(quant_iter->data()),
        result->element_size());
    EXPECT_EQ(converter_out, reformer_out)
        << "Converter and reformer produced different quantized output";
  }
}

// ---------------------------------------------------------------------------
// SDC distance computation: verify sanity
// ---------------------------------------------------------------------------
TEST(TurboQuantEngine, SdcDistance) {
  const size_t d = 256;
  const int bits = 4;

  TurboQuantEngine engine(d, bits, false /*mse*/, true /*rotate*/, 42);

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  // Generate two random vectors
  std::vector<float> x(d), y(d);
  float nx = 0, ny = 0;
  for (size_t i = 0; i < d; ++i) {
    x[i] = dist(rng);
    y[i] = dist(rng);
    nx += x[i] * x[i];
    ny += y[i] * y[i];
  }
  nx = std::sqrt(nx);
  ny = std::sqrt(ny);
  for (size_t i = 0; i < d; ++i) {
    x[i] /= nx;
    y[i] /= ny;
  }

  std::vector<uint8_t> qx(engine.total_bytes()), qy(engine.total_bytes());
  engine.quantize(x.data(), qx.data());
  engine.quantize(y.data(), qy.data());

  // SDC L2 distance should be non-negative
  float sdc_l2 = engine.sdc_squared_l2(qx.data(), qy.data());
  EXPECT_GE(sdc_l2, 0.0f);

  // SDC L2 distance to self should be small (not exactly 0 due to quantization)
  float self_dist = engine.sdc_squared_l2(qx.data(), qx.data());
  EXPECT_NEAR(self_dist, 0.0f, 1e-3f);

  // SDC inner product of x with itself should be close to ||x||^2 = 1
  float self_ip = engine.sdc_inner_product(qx.data(), qx.data());
  EXPECT_NEAR(self_ip, 1.0f, 0.2f);
}
