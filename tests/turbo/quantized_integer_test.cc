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
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <ailego/math/distance.h>
#include <ailego/math/norm_matrix.h>
#include <ailego/math/normalizer.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>

using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;

TEST(QuantizedIntegerMetric, TestInt8InnerProduct) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1000;

  auto converter = IndexFactory::CreateConverter("Int8StreamingConverter");
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));
  auto &convert_meta = converter->meta();
  auto reformer = IndexFactory::CreateReformer(convert_meta.reformer_name());


  auto func_avx2 = turbo::get_distance_func(
      turbo::MetricType::kInnerProduct, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX2);

  auto func_sse = turbo::get_distance_func(
      turbo::MetricType::kInnerProduct, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kSSE);

  ailego::NumericalVector<float> query_vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    query_vec[j] = dist(gen);
  }

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::NumericalVector<float> doc_vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      doc_vec[j] = dist(gen);
    }

    IndexQueryMeta qmeta;
    qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
    IndexQueryMeta qmeta_reformer;

    std::string query_out;
    ASSERT_EQ(0, reformer->transform(query_vec.data(), qmeta, &query_out,
                                     &qmeta_reformer));
    ASSERT_EQ(qmeta_reformer.dimension(), convert_meta.dimension());

    std::string doc_out;
    ASSERT_EQ(0, reformer->transform(doc_vec.data(), qmeta, &doc_out,
                                     &qmeta_reformer));
    ASSERT_EQ(qmeta_reformer.dimension(), convert_meta.dimension());

    float score_float = ailego::Distance::MinusInnerProduct(
        query_vec.data(), doc_vec.data(), DIMENSION);

    float score_avx2{0.0f};
    float score_sse{0.0f};

    func_avx2(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
              &score_avx2);
    func_sse(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
             &score_sse);

    ASSERT_NEAR(score_float, score_avx2, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float, score_sse, 0.2 * DIMENSION);
    ASSERT_NEAR(score_avx2, score_sse, 0.001);
  }
}

#if 0
TEST(QuantizedIntegerMetric, TestInt4InnerProduct) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen) * 2;
  const size_t COUNT = 1000;
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("InnerProduct", 0, Params());
  auto converter = IndexFactory::CreateConverter("Int4StreamingConverter");
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));

  auto holder = GetHolder(DIMENSION, COUNT, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DT_INT4, holder2->data_type());
  auto &meta2 = converter->meta();

  auto reformer = IndexFactory::CreateReformer(meta2.reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(meta2.reformer_params()));

  ailego::NumericalVector<float> vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    vec[j] = dist(gen);
  }
  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta2;
  std::string out;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &out, &qmeta2));
  ASSERT_EQ(qmeta2.dimension(), meta2.dimension());

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  auto metric = IndexFactory::CreateMetric(meta2.metric_name());
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0, metric->init(meta2, meta2.metric_params()));
  auto compute = metric->distance();
  ASSERT_TRUE(compute);

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    const float *mf = (const float *)iter->data();
    const int8_t *mi = (const int8_t *)iter2->data();
    const int8_t *qi = reinterpret_cast<const int8_t *>(&out[0]);
    float v1 = ailego::Distance::MinusInnerProduct(mf, vec.data(),
                                                   holder->dimension());
    float v2;
    compute(mi, qi, holder2->dimension(), &v2);
    ASSERT_NEAR(v1, v2, 0.2 * DIMENSION);

    std::string out2;
    ASSERT_EQ(0, reformer->convert(iter->data(), qmeta, &out2, &qmeta2));
    ASSERT_EQ(out2.size(), holder2->element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), iter2->data(), out2.size()));
  }
}

TEST(QuantizedIntegerMetric, TestInt8Cosine) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1000;
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("Cosine", 0, Params());
  auto converter = IndexFactory::CreateConverter("CosineInt8Converter");
  ASSERT_TRUE(!!converter);
  Params converter_params;
  ASSERT_EQ(0u, converter->init(meta, converter_params));

  auto holder = GetHolder(DIMENSION, COUNT, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DT_INT8, holder2->data_type());
  auto &meta2 = converter->meta();

  auto reformer = IndexFactory::CreateReformer(meta2.reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(meta2.reformer_params()));

  ailego::NumericalVector<float> vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    vec[j] = dist(gen);
  }
  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta2;
  std::string out;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &out, &qmeta2));
  ASSERT_EQ(qmeta2.dimension(), meta2.dimension());

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  auto metric = IndexFactory::CreateMetric(meta2.metric_name());
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0, metric->init(meta2, meta2.metric_params()));
  auto compute_batch = metric->batch_distance();
  ASSERT_TRUE(compute_batch);

  int8_t *qi = reinterpret_cast<int8_t *>(&out[0]);
  if (auto query_preprocess_func = metric->get_query_preprocess_func();
      query_preprocess_func != nullptr) {
    query_preprocess_func(qi, holder2->dimension());
  }

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    const float *mf = (const float *)iter->data();
    const int8_t *mi = (const int8_t *)iter2->data();

    // normalize mf & vec
    std::vector<float> normalized_mf(DIMENSION);
    memcpy(normalized_mf.data(), mf, DIMENSION * sizeof(float));
    float norm_mf = 0.0;
    ailego::Normalizer<float>::L2((float *)normalized_mf.data(), DIMENSION,
                                  &norm_mf);
    std::vector<float> normalized_vec(DIMENSION);
    memcpy(normalized_vec.data(), vec.data(), DIMENSION * sizeof(float));
    float norm_vec = 0.0;
    ailego::Normalizer<float>::L2((float *)normalized_vec.data(), DIMENSION,
                                  &norm_vec);

    float v1 = ailego::Distance::MinusInnerProduct(
        normalized_mf.data(), normalized_vec.data(), holder->dimension());
    float v2;
    compute_batch(reinterpret_cast<const void **>(&mi), qi, 1,
                  holder2->dimension(), &v2);
    // printf("%f %f\n", v1, v2);
    ASSERT_NEAR(v1, v2, 0.2 * DIMENSION);

    std::string out2;
    ASSERT_EQ(0, reformer->convert(iter->data(), qmeta, &out2, &qmeta2));
    ASSERT_EQ(out2.size(), holder2->element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), iter2->data(), out2.size()));
  }
}

#endif