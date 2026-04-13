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

// Target Test Type: avx, avx512, scalar
TEST(CosineMetric, TestFp32Cosine) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1000;

  auto converter = IndexFactory::CreateConverter("CosineFp32Converter");
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("Cosine", 0, Params());
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));
  auto &convert_meta = converter->meta();
  auto reformer = IndexFactory::CreateReformer(convert_meta.reformer_name());

  auto func_avx512 = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kFp32,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX512);

  auto func_avx = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kFp32,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX);

  auto func_scalar = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kFp32,
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

    float score_scalar{0.0f};
    float score_avx{0.0f};
    float score_avx512{0.0f};

    func_scalar(doc_vec.data(), query_vec.data(), DIMENSION, &score_scalar);

    func_avx512(doc_vec.data(), query_vec.data(), DIMENSION, &score_avx512);

    func_avx(doc_vec.data(), query_vec.data(), DIMENSION, &score_avx);

    float epsilon = 0.001;
    ASSERT_NEAR(score_scalar, score_avx512, epsilon);
    ASSERT_NEAR(score_scalar, score_avx, epsilon);
  }
}

// Target Test Type: avx, avx512, avx512fp16, scalar
TEST(CosineMetric, TestFp16Cosine) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1000;

  auto converter = IndexFactory::CreateConverter("CosineFp16Converter");
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("Cosine", 0, Params());
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));
  auto &convert_meta = converter->meta();
  auto reformer = IndexFactory::CreateReformer(convert_meta.reformer_name());

  auto func_avx512fp16 = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kFp16,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX512FP16);

  auto func_avx512 = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kFp16,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX512);

  auto func_avx = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kFp16,
      turbo::QuantizeType::kDefault, turbo::CpuArchType::kAVX);

  auto func_scalar = turbo::get_distance_func(
      turbo::MetricType::kCosine, turbo::DataType::kFp16,
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

    float score_avx512fp16{0.0f};
    float score_avx512{0.0f};
    float score_avx{0.0f};
    float score_scalar{0.0f};

    func_avx512fp16(doc_out.data(), query_out.data(),
                    qmeta_reformer.dimension(), &score_avx512fp16);

    func_avx512(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
                &score_avx512);

    func_avx(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
             &score_avx);

    func_scalar(doc_out.data(), query_out.data(), qmeta_reformer.dimension(),
                &score_scalar);

    float epsilon = 0.2;
    ASSERT_NEAR(score_scalar, score_avx512fp16, epsilon);
    ASSERT_NEAR(score_scalar, score_avx512, epsilon);
    ASSERT_NEAR(score_scalar, score_avx, epsilon);
  }
}
