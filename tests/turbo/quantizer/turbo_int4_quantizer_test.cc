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

TEST(Int4Quantizer, General) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("Int4Quantizer");
  ASSERT_TRUE(converter);
  zvec::ailego::Params params;
  params.set("proxima.int4_quantizer.converter.histogram_bins_count", 10000);
  ASSERT_EQ(0u, converter->init(meta, params));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          DIMENSION);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
      if (i == 0) printf(" %f", vec[j]);
    }
    if (i == 0) printf("\n");
    holder->emplace(i + 1, vec);
  }
  EXPECT_EQ(COUNT, holder->count());
  EXPECT_EQ(IndexMeta::DataType::DT_FP32, holder->data_type());

  auto two_pass_holder = IndexHelper::MakeTwoPassHolder(std::move(holder));
  ASSERT_EQ(0u, quantizer->train(two_pass_holder));

  auto iter = holder->create_iterator();
  std::string buffer;

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    EXPECT_TRUE(iter->data());

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, quantizer->quantize(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());


    EXPECT_EQ(0, quantizer->dequantize(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    EXPECT_EQ(0, quantizer->quantize(iter->data(),
                                     IndexQueryMeta(holder->data_type(),
                                                    holder->dimension() / 3),
                                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 3, qmeta.dimension());
    ASSERT_EQ(buffer, buffer2);
  }
}