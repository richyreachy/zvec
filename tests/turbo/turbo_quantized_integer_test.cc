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

// Target Test Type: avx2, sse, scalar
TEST(QuantizedIntegerMetric, TestInt8InnerProduct) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1000;

  auto converter = IndexFactory::CreateConverter("Int8StreamingConverter");
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("InnerProduct", 0, Params());
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));
  auto &convert_meta = converter->meta();
  auto reformer = IndexFactory::CreateReformer(convert_meta.reformer_name());
  ASSERT_EQ(0, reformer->init(convert_meta.reformer_params()));

  auto func_float32 = turbo::get_distance_func(
      turbo::MetricType::kInnerProduct, turbo::DataType::kFp32,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAuto);

  auto func_avx512vnni = turbo::get_distance_func(
      turbo::MetricType::kInnerProduct, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX512VNNI);

  auto func_avx2 = turbo::get_distance_func(
      turbo::MetricType::kInnerProduct, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX2);

  auto func_sse = turbo::get_distance_func(
      turbo::MetricType::kInnerProduct, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kSSE);

  auto func_scalar = turbo::get_distance_func(
      turbo::MetricType::kInnerProduct, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kScalar);

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

    float score_float32{0.0f};
    float score_scalar{0.0f};
    float score_avx512vnni{0.0f};
    float score_avx2{0.0f};
    float score_sse{0.0f};

    func_float32(query_vec.data(), doc_vec.data(), DIMENSION, &score_float32);

    func_scalar(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
                &score_scalar);

    func_avx512vnni(doc_out.data(), query_out.data(),
                    qmeta_reformer.dimension(), &score_avx512vnni);

    func_avx2(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
              &score_avx2);

    func_sse(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
             &score_sse);

    ASSERT_NEAR(score_float32, score_avx512vnni, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_avx2, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_sse, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_scalar, 0.2 * DIMENSION);
    ASSERT_NEAR(score_scalar, score_avx2, 0.001);
    ASSERT_NEAR(score_scalar, score_sse, 0.001);
  }
}

// Target Test Type: avx2, sse, scalar
TEST(QuantizedIntegerMetric, TestInt4InnerProduct) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen) * 2;
  const size_t COUNT = 1000;

  auto converter = IndexFactory::CreateConverter("Int4StreamingConverter");
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("InnerProduct", 0, Params());
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));
  auto &convert_meta = converter->meta();
  auto reformer = IndexFactory::CreateReformer(convert_meta.reformer_name());
  ASSERT_EQ(0, reformer->init(convert_meta.reformer_params()));

  auto func_float32 = turbo::get_distance_func(
      turbo::MetricType::kInnerProduct, turbo::DataType::kFp32,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAuto);

  auto func_avx2 = turbo::get_distance_func(
      turbo::MetricType::kInnerProduct, turbo::DataType::kInt4,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX2);

  auto func_sse = turbo::get_distance_func(
      turbo::MetricType::kInnerProduct, turbo::DataType::kInt4,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kSSE);

  auto func_scalar = turbo::get_distance_func(
      turbo::MetricType::kInnerProduct, turbo::DataType::kInt4,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kScalar);

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

    float score_float32{0.0f};
    float score_scalar{0.0f};
    float score_avx2{0.0f};
    float score_sse{0.0f};

    func_float32(query_vec.data(), doc_vec.data(), DIMENSION, &score_float32);

    func_scalar(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
                &score_scalar);

    func_avx2(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
              &score_avx2);

    func_sse(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
             &score_sse);

    ASSERT_NEAR(score_float32, score_avx2, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_sse, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_scalar, 0.2 * DIMENSION);
    ASSERT_NEAR(score_scalar, score_avx2, 0.001);
    ASSERT_NEAR(score_scalar, score_sse, 0.001);
  }
}

// Target Test Type: avx2, sse, scalar
TEST(QuantizedIntegerMetric, TestInt8SquaredEuclidean) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1000;

  auto converter = IndexFactory::CreateConverter("Int8StreamingConverter");
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("SquaredEuclidean", 0, Params());
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));
  auto &convert_meta = converter->meta();
  auto reformer = IndexFactory::CreateReformer(convert_meta.reformer_name());
  ASSERT_EQ(0, reformer->init(convert_meta.reformer_params()));

  auto func_float32 = turbo::get_distance_func(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kFp32,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAuto);

  auto func_avx2 = turbo::get_distance_func(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX2);

  auto func_sse = turbo::get_distance_func(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kSSE);

  auto func_scalar = turbo::get_distance_func(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kScalar);

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

    float score_float32{0.0f};
    float score_scalar{0.0f};
    float score_avx2{0.0f};
    float score_sse{0.0f};

    func_float32(query_vec.data(), doc_vec.data(), DIMENSION, &score_float32);

    func_scalar(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
                &score_scalar);

    func_avx2(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
              &score_avx2);

    func_sse(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
             &score_sse);

    ASSERT_NEAR(score_float32, score_avx2, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_sse, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_scalar, 0.2 * DIMENSION);
    ASSERT_NEAR(score_scalar, score_avx2, 0.001);
    ASSERT_NEAR(score_scalar, score_sse, 0.001);
  }
}

// Target Test Type: avx2, sse, scalar
TEST(QuantizedIntegerMetric, TestInt4SquaredEuclidean) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen) * 2;
  const size_t COUNT = 1000;

  auto converter = IndexFactory::CreateConverter("Int4StreamingConverter");
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("SquaredEuclidean", 0, Params());
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));
  auto &convert_meta = converter->meta();
  auto reformer = IndexFactory::CreateReformer(convert_meta.reformer_name());
  ASSERT_EQ(0, reformer->init(convert_meta.reformer_params()));

  auto func_float32 = turbo::get_distance_func(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kFp32,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAuto);

  auto func_avx2 = turbo::get_distance_func(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt4,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX2);

  auto func_sse = turbo::get_distance_func(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt4,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kSSE);

  auto func_scalar = turbo::get_distance_func(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt4,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kScalar);

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

    float score_float32{0.0f};
    float score_scalar{0.0f};
    float score_avx2{0.0f};
    float score_sse{0.0f};

    func_float32(query_vec.data(), doc_vec.data(), DIMENSION, &score_float32);

    func_scalar(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
                &score_scalar);

    func_avx2(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
              &score_avx2);

    func_sse(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
             &score_sse);

    ASSERT_NEAR(score_float32, score_avx2, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_sse, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_scalar, 0.2 * DIMENSION);
    ASSERT_NEAR(score_scalar, score_avx2, 0.001);
    ASSERT_NEAR(score_scalar, score_sse, 0.001);
  }
}

// Target Test Type: avx2, sse, scalar
TEST(QuantizedIntegerMetric, TestInt8Cosine) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1000;

  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("Cosine", 0, Params());

  // fp32 converter
  auto fp32_converter = IndexFactory::CreateConverter("CosineFp32Converter");
  ASSERT_TRUE(!!fp32_converter);
  ASSERT_EQ(0u, fp32_converter->init(meta, Params()));

  auto &fp32_convert_meta = fp32_converter->meta();
  auto fp32_reformer =
      IndexFactory::CreateReformer(fp32_convert_meta.reformer_name());
  ASSERT_EQ(0, fp32_reformer->init(fp32_convert_meta.reformer_params()));

  // int8 converter
  auto converter = IndexFactory::CreateConverter("CosineInt8Converter");
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));

  auto &convert_meta = converter->meta();
  auto reformer = IndexFactory::CreateReformer(convert_meta.reformer_name());
  ASSERT_EQ(0, reformer->init(convert_meta.reformer_params()));

  auto func_float32 = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kFp32,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAuto);

  auto func_avx512vnni = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX512VNNI);

  auto func_avx2 = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX2);

  auto func_sse = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kSSE);

  auto func_scalar = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kInt8,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kScalar);

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
    IndexQueryMeta fp32_qmeta_reformer;

    float score_float32{0.0f};
    float score_scalar{0.0f};
    float score_avx512vnni{0.0f};
    float score_avx2{0.0f};
    float score_sse{0.0f};

    std::string fp32_query_out;
    ASSERT_EQ(0,
              fp32_reformer->transform(query_vec.data(), qmeta, &fp32_query_out,
                                       &fp32_qmeta_reformer));
    ASSERT_EQ(fp32_qmeta_reformer.dimension(), fp32_convert_meta.dimension());

    std::string fp32_doc_out;
    ASSERT_EQ(0, fp32_reformer->transform(doc_vec.data(), qmeta, &fp32_doc_out,
                                          &fp32_qmeta_reformer));
    ASSERT_EQ(fp32_qmeta_reformer.dimension(), fp32_convert_meta.dimension());

    func_float32(fp32_query_out.data(), fp32_doc_out.data(),
                 fp32_qmeta_reformer.dimension(), &score_float32);

    IndexQueryMeta qmeta_reformer;

    std::string query_out;
    ASSERT_EQ(0, reformer->transform(query_vec.data(), qmeta, &query_out,
                                     &qmeta_reformer));
    ASSERT_EQ(qmeta_reformer.dimension(), convert_meta.dimension());

    std::string doc_out;
    ASSERT_EQ(0, reformer->transform(doc_vec.data(), qmeta, &doc_out,
                                     &qmeta_reformer));
    ASSERT_EQ(qmeta_reformer.dimension(), convert_meta.dimension());

    func_scalar(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
                &score_scalar);

    func_avx512vnni(doc_out.data(), query_out.data(),
                    qmeta_reformer.dimension(), &score_avx512vnni);

    func_avx2(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
              &score_avx2);

    func_sse(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
             &score_sse);

    ASSERT_NEAR(score_float32, score_avx512vnni, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_avx2, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_sse, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_scalar, 0.2 * DIMENSION);
    ASSERT_NEAR(score_scalar, score_avx2, 0.001);
    ASSERT_NEAR(score_scalar, score_sse, 0.001);
  }
}

// Target Test Type: avx2, sse, scalar
TEST(QuantizedIntegerMetric, TestInt4Cosine) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen) * 2;
  const size_t COUNT = 1000;

  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("Cosine", 0, Params());

  // fp32 converter
  auto fp32_converter = IndexFactory::CreateConverter("CosineFp32Converter");
  ASSERT_TRUE(!!fp32_converter);
  ASSERT_EQ(0u, fp32_converter->init(meta, Params()));

  auto &fp32_convert_meta = fp32_converter->meta();
  auto fp32_reformer =
      IndexFactory::CreateReformer(fp32_convert_meta.reformer_name());
  ASSERT_EQ(0, fp32_reformer->init(fp32_convert_meta.reformer_params()));

  // int4 converter
  auto converter = IndexFactory::CreateConverter("CosineInt4Converter");
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));
  auto &convert_meta = converter->meta();
  auto reformer = IndexFactory::CreateReformer(convert_meta.reformer_name());
  ASSERT_EQ(0, reformer->init(convert_meta.reformer_params()));

  auto func_float32 = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kFp32,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAuto);

  auto func_avx2 = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kInt4,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX2);

  auto func_sse = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kInt4,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kSSE);

  auto func_scalar = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kInt4,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kScalar);

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
    IndexQueryMeta fp32_qmeta_reformer;

    float score_float32{0.0f};
    float score_scalar{0.0f};
    float score_avx2{0.0f};
    float score_sse{0.0f};

    std::string fp32_query_out;
    ASSERT_EQ(0,
              fp32_reformer->transform(query_vec.data(), qmeta, &fp32_query_out,
                                       &fp32_qmeta_reformer));
    ASSERT_EQ(fp32_qmeta_reformer.dimension(), fp32_convert_meta.dimension());

    std::string fp32_doc_out;
    ASSERT_EQ(0, fp32_reformer->transform(doc_vec.data(), qmeta, &fp32_doc_out,
                                          &fp32_qmeta_reformer));
    ASSERT_EQ(fp32_qmeta_reformer.dimension(), fp32_convert_meta.dimension());

    func_float32(fp32_query_out.data(), fp32_doc_out.data(),
                 fp32_qmeta_reformer.dimension(), &score_float32);

    IndexQueryMeta qmeta_reformer;

    std::string query_out;
    ASSERT_EQ(0, reformer->transform(query_vec.data(), qmeta, &query_out,
                                     &qmeta_reformer));
    ASSERT_EQ(qmeta_reformer.dimension(), convert_meta.dimension());

    std::string doc_out;
    ASSERT_EQ(0, reformer->transform(doc_vec.data(), qmeta, &doc_out,
                                     &qmeta_reformer));
    ASSERT_EQ(qmeta_reformer.dimension(), convert_meta.dimension());

    func_scalar(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
                &score_scalar);

    func_avx2(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
              &score_avx2);

    func_sse(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
             &score_sse);

    ASSERT_NEAR(score_float32, score_avx2, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_sse, 0.2 * DIMENSION);
    ASSERT_NEAR(score_float32, score_scalar, 0.2 * DIMENSION);
    ASSERT_NEAR(score_scalar, score_avx2, 0.001);
    ASSERT_NEAR(score_scalar, score_sse, 0.001);
  }
}
