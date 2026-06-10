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

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include <zvec/turbo/turbo.h>
#include "quantizer/int8_quantizer/int8_quantizer.h"
#include "zvec/core/framework/index_factory.h"

using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;

TEST(Int8Quantizer, Int8General) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto quantizer = IndexFactory::CreateQuantizer("Int8Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
  params.set("int8_quantizer.bias", 0.0f);
  params.set("int8_quantizer.scale", 127.0f);
  ASSERT_EQ(0u, quantizer->init(meta, params));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          DIMENSION);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    holder->emplace(i + 1, vec);
  }
  EXPECT_EQ(COUNT, holder->count());
  EXPECT_EQ(IndexMeta::DataType::DT_FP32, holder->data_type());

  ASSERT_EQ(0u, quantizer->train(holder));

  auto iter = holder->create_iterator();
  std::string quant_buffer;
  std::string dequant_buffer;

  for (; iter->is_valid(); iter->next()) {
    EXPECT_TRUE(iter->data());

    IndexQueryMeta qmeta;
    quant_buffer.clear();
    EXPECT_EQ(0, quantizer->quantize(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &quant_buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(quantizer->meta().dimension(), qmeta.dimension());

    dequant_buffer.clear();
    EXPECT_EQ(
        0, quantizer->dequantize(quant_buffer.data(), qmeta, &dequant_buffer));

    const float *original_data = reinterpret_cast<const float *>(iter->data());
    const float *dequantize_data =
        reinterpret_cast<const float *>(dequant_buffer.data());
    for (size_t i = 0; i < holder->dimension(); ++i) {
      EXPECT_NEAR(original_data[i], dequantize_data[i], 1e-2);
    }
  }
}


TEST(Int8Quantizer, TestSerialize) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto quantizer = IndexFactory::CreateQuantizer("Int8Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
  ASSERT_EQ(0u, quantizer->init(meta, params));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          DIMENSION);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    holder->emplace(i + 1, vec);
  }
  EXPECT_EQ(COUNT, holder->count());
  EXPECT_EQ(IndexMeta::DataType::DT_FP32, holder->data_type());

  ASSERT_EQ(0u, quantizer->train(holder));

  std::string param_buffer;
  ASSERT_EQ(0u, quantizer->serialize(&param_buffer));

  // The serialized blob must begin with a valid self-describing header.
  ASSERT_GE(param_buffer.size(), sizeof(zvec::turbo::QuantizerSerHeader));
  const auto *header =
      reinterpret_cast<const zvec::turbo::QuantizerSerHeader *>(
          param_buffer.data());
  EXPECT_EQ(zvec::turbo::kQuantizerMagic, header->magic);
  EXPECT_EQ(zvec::turbo::kQuantizerSerVersion, header->version);
  EXPECT_EQ(sizeof(float) * 2u, header->payload_size);
  EXPECT_EQ(param_buffer.size(),
            sizeof(zvec::turbo::QuantizerSerHeader) + header->payload_size);

  // new quantizer
  auto quantizer_new = IndexFactory::CreateQuantizer("Int8Quantizer");
  ASSERT_TRUE(quantizer_new);
  zvec::ailego::Params params_new;
  ASSERT_EQ(0u, quantizer_new->init(meta, params_new));
  ASSERT_EQ(0u, quantizer_new->deserialize(param_buffer));

  // Zero-copy overload restores the same state.
  auto quantizer_zc = IndexFactory::CreateQuantizer("Int8Quantizer");
  ASSERT_TRUE(quantizer_zc);
  zvec::ailego::Params params_zc;
  ASSERT_EQ(0u, quantizer_zc->init(meta, params_zc));
  ASSERT_EQ(
      0u, quantizer_zc->deserialize(param_buffer.data(), param_buffer.size()));
  auto *int8_quantizer_zc =
      reinterpret_cast<zvec::turbo::Int8Quantizer *>(quantizer_zc.get());

  auto *int8_quantizer =
      reinterpret_cast<zvec::turbo::Int8Quantizer *>(quantizer.get());
  auto *int8_quantizer_new =
      reinterpret_cast<zvec::turbo::Int8Quantizer *>(quantizer_new.get());

  ASSERT_EQ(int8_quantizer->bias(), int8_quantizer_new->bias());
  ASSERT_EQ(int8_quantizer->scale(), int8_quantizer_new->scale());

  ASSERT_EQ(int8_quantizer->bias(), int8_quantizer_zc->bias());
  ASSERT_EQ(int8_quantizer->scale(), int8_quantizer_zc->scale());

  auto iter = holder->create_iterator();
  std::string quant_buffer;
  std::string dequant_buffer;

  for (; iter->is_valid(); iter->next()) {
    EXPECT_TRUE(iter->data());

    IndexQueryMeta qmeta;
    quant_buffer.clear();
    EXPECT_EQ(0, quantizer->quantize(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &quant_buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(quantizer->meta().dimension(), qmeta.dimension());

    dequant_buffer.clear();
    EXPECT_EQ(
        0, quantizer->dequantize(quant_buffer.data(), qmeta, &dequant_buffer));

    const float *original_data = reinterpret_cast<const float *>(iter->data());
    const float *dequantize_data =
        reinterpret_cast<const float *>(dequant_buffer.data());
    for (size_t i = 0; i < holder->dimension(); ++i) {
      EXPECT_NEAR(original_data[i], dequantize_data[i], 0.15);
    }
  }

  auto iter2 = holder->create_iterator();
  for (; iter2->is_valid(); iter2->next()) {
    EXPECT_TRUE(iter2->data());

    IndexQueryMeta qmeta;
    quant_buffer.clear();
    EXPECT_EQ(0, quantizer_new->quantize(
                     iter2->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &quant_buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(quantizer_new->meta().dimension(), qmeta.dimension());

    dequant_buffer.clear();
    EXPECT_EQ(0, quantizer_new->dequantize(quant_buffer.data(), qmeta,
                                           &dequant_buffer));

    const float *original_data = reinterpret_cast<const float *>(iter2->data());
    const float *dequantize_data =
        reinterpret_cast<const float *>(dequant_buffer.data());
    for (size_t i = 0; i < holder->dimension(); ++i) {
      EXPECT_NEAR(original_data[i], dequantize_data[i], 0.15);
    }
  }
}

TEST(Int8Quantizer, NewInterface) {
  std::mt19937 gen(20240608);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t DIMENSION = 16;
  const size_t COUNT = 8;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta.set_metric("SquaredEuclidean", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Int8Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
  ASSERT_EQ(0u, quantizer->init(meta, params));

  // ---- Scalar accessors of the new interface ----
  EXPECT_EQ(zvec::turbo::DataType::kFp32, quantizer->input_data_type());
  EXPECT_EQ(static_cast<int>(DIMENSION), quantizer->dim());
  EXPECT_TRUE(quantizer->require_train());
  EXPECT_GT(quantizer->quantized_datapoint_vector_length(), 0u);
  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(),
            quantizer->quantized_query_vector_length());

  const size_t dp_len = quantizer->quantized_datapoint_vector_length();
  const size_t q_len = quantizer->quantized_query_vector_length();

  // ---- Prepare raw fp32 datapoints and a query ----
  std::vector<std::vector<float>> raw(COUNT, std::vector<float>(DIMENSION));
  for (size_t i = 0; i < COUNT; ++i) {
    for (size_t j = 0; j < DIMENSION; ++j) {
      raw[i][j] = dist(gen);
    }
  }
  std::vector<float> raw_query(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    raw_query[j] = dist(gen);
  }

  // ---- quantize_data / quantize_query ----
  std::vector<std::string> dps(COUNT, std::string(dp_len, '\0'));
  for (size_t i = 0; i < COUNT; ++i) {
    quantizer->quantize_data(raw[i].data(), &dps[i][0]);
  }
  std::string q(q_len, '\0');
  quantizer->quantize_query(raw_query.data(), &q[0]);

  std::vector<const void *> dp_list(COUNT);
  for (size_t i = 0; i < COUNT; ++i) {
    dp_list[i] = dps[i].data();
  }

  // ---- calc_distance_dp_query vs batch ----
  std::vector<float> single(COUNT, 0.0f);
  for (size_t i = 0; i < COUNT; ++i) {
    single[i] = quantizer->calc_distance_dp_query(dps[i].data(), q.data());
    EXPECT_GE(single[i], 0.0f);
  }
  std::vector<float> batch(COUNT, 0.0f);
  quantizer->calc_distance_dp_query_batch(
      dp_list.data(), static_cast<int>(COUNT), q.data(), batch.data());
  for (size_t i = 0; i < COUNT; ++i) {
    EXPECT_NEAR(single[i], batch[i], 1e-4);
  }

  // ---- unquantized query path should match quantizing the query first ----
  std::vector<float> single_unq(COUNT, 0.0f);
  for (size_t i = 0; i < COUNT; ++i) {
    single_unq[i] = quantizer->calc_distance_dp_query_unquantized(
        dps[i].data(), raw_query.data());
    EXPECT_NEAR(single[i], single_unq[i], 1e-4);
  }
  std::vector<float> batch_unq(COUNT, 0.0f);
  quantizer->calc_distance_dp_query_batch_unquantized(
      dp_list.data(), static_cast<int>(COUNT), raw_query.data(),
      batch_unq.data());
  for (size_t i = 0; i < COUNT; ++i) {
    EXPECT_NEAR(single_unq[i], batch_unq[i], 1e-4);
  }

  // ---- dp-to-dp distance: self distance is ~0, symmetric ----
  float self_d = quantizer->calc_distance_dp_dp(dps[0].data(), dps[0].data());
  EXPECT_NEAR(0.0f, self_d, 1e-3);
  float d01 = quantizer->calc_distance_dp_dp(dps[0].data(), dps[1].data());
  float d10 = quantizer->calc_distance_dp_dp(dps[1].data(), dps[0].data());
  EXPECT_NEAR(d01, d10, 1e-4);
}
