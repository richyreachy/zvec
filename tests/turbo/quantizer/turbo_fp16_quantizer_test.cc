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
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include <zvec/turbo/turbo.h>
#include "zvec/core/framework/index_factory.h"

using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;

TEST(Fp16Quantizer, General) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
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
    EXPECT_EQ(IndexMeta::DataType::DT_FP16, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());

    dequant_buffer.clear();
    EXPECT_EQ(
        0, quantizer->dequantize(quant_buffer.data(), qmeta, &dequant_buffer));

    const float *original_data = reinterpret_cast<const float *>(iter->data());
    const float *dequantize_data =
        reinterpret_cast<const float *>(dequant_buffer.data());
    for (size_t i = 0; i < holder->dimension(); ++i) {
      EXPECT_NEAR(original_data[i], dequantize_data[i], 1e-3);
    }
  }
}
