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
using namespace zvec::turbo;

// Target Test Type: avx, avx512, scalar
TEST(SquaredEuclideanMetric, TestFp32SquaredEuclidean) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1024;

  auto func_avx512 =
      get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                        QuantizeType::kDefault, CpuArchType::kAVX512);

  auto func_avx =
      get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                        QuantizeType::kDefault, CpuArchType::kAVX);

  auto func_scalar =
      get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                        QuantizeType::kDefault, CpuArchType::kScalar);

  ailego::NumericalVector<float> query_vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    query_vec[j] = dist(gen);
  }

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::NumericalVector<float> doc_vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      doc_vec[j] = dist(gen);
    }

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
TEST(SquaredEuclideanMetric, TestFp16SquaredEuclidean) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1024;

  auto quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("SquaredEuclidean", 0, Params());
  ASSERT_TRUE(!!quantizer);
  ASSERT_EQ(0u, quantizer->init(meta, Params()));
  auto &convert_meta = quantizer->meta();

  auto func_avx512fp16 =
      get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                        QuantizeType::kDefault, CpuArchType::kAVX512FP16);

  auto func_avx512 =
      get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                        QuantizeType::kDefault, CpuArchType::kAVX512);

  auto func_avx =
      get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                        QuantizeType::kDefault, CpuArchType::kAVX);

  auto func_scalar =
      get_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                        QuantizeType::kDefault, CpuArchType::kScalar);

  ailego::NumericalVector<float> query_vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    query_vec[j] = dist(gen);
  }

  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta_quantizer;

  std::string query_out;
  ASSERT_EQ(0, quantizer->quantize(query_vec.data(), qmeta, &query_out,
                                   &qmeta_quantizer));
  ASSERT_EQ(qmeta_quantizer.dimension(), convert_meta.dimension());

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::NumericalVector<float> doc_vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      doc_vec[j] = dist(gen);
    }

    std::string doc_out;
    ASSERT_EQ(0, quantizer->quantize(doc_vec.data(), qmeta, &doc_out,
                                     &qmeta_quantizer));
    ASSERT_EQ(qmeta_quantizer.dimension(), convert_meta.dimension());

    float score_avx512fp16{0.0f};
    float score_avx512{0.0f};
    float score_avx{0.0f};
    float score_scalar{0.0f};

    func_avx512fp16(doc_out.data(), query_out.data(),
                    qmeta_quantizer.dimension(), &score_avx512fp16);

    func_avx512(doc_out.data(), query_out.data(), qmeta_quantizer.dimension(),
                &score_avx512);

    func_avx(doc_out.data(), query_out.data(), qmeta_quantizer.dimension(),
             &score_avx);

    func_scalar(doc_out.data(), query_out.data(), qmeta_quantizer.dimension(),
                &score_scalar);

    float epsilon = 0.2;
    ASSERT_NEAR(score_scalar, score_avx512fp16, epsilon);
    ASSERT_NEAR(score_scalar, score_avx512, epsilon);
    ASSERT_NEAR(score_scalar, score_avx, epsilon);
  }
}

// Target Test Type: avx, avx512, scalar
TEST(SquaredEuclideanMetric, TestFp32SquaredEuclideanBatch) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1024;
  const size_t BATCH_SIZE = 16;

  auto batch_func_avx512 =
      get_batch_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                              QuantizeType::kDefault, CpuArchType::kAVX512);

  auto batch_func_avx =
      get_batch_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                              QuantizeType::kDefault, CpuArchType::kAVX);

  auto batch_func_scalar =
      get_batch_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                              QuantizeType::kDefault, CpuArchType::kScalar);

  ailego::NumericalVector<float> query_vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    query_vec[j] = dist(gen);
  }

  std::vector<ailego::NumericalVector<float>> doc_vecs;
  for (size_t i = 0; i < COUNT; ++i) {
    ailego::NumericalVector<float> doc_vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      doc_vec[j] = dist(gen);
    }
    doc_vecs.push_back(doc_vec);

    if (doc_vecs.size() == BATCH_SIZE) {
      std::vector<const void *> doc_ptrs(BATCH_SIZE);
      for (size_t k = 0; k < BATCH_SIZE; ++k) {
        doc_ptrs[k] = doc_vecs[k].data();
      }

      std::vector<float> score_scalar(BATCH_SIZE, 0.0f);
      std::vector<float> score_avx(BATCH_SIZE, 0.0f);
      std::vector<float> score_avx512(BATCH_SIZE, 0.0f);

      batch_func_scalar(doc_ptrs.data(), query_vec.data(), DIMENSION,
                        BATCH_SIZE, &score_scalar[0]);

      batch_func_avx512(doc_ptrs.data(), query_vec.data(), DIMENSION,
                        BATCH_SIZE, &score_avx512[0]);

      batch_func_avx(doc_ptrs.data(), query_vec.data(), DIMENSION, BATCH_SIZE,
                     &score_avx[0]);

      for (size_t j = 0; j < BATCH_SIZE; ++j) {
        float epsilon = 0.001;
        ASSERT_NEAR(score_scalar[j], score_avx512[j], epsilon);
        ASSERT_NEAR(score_scalar[j], score_avx[j], epsilon);
      }

      doc_vecs.clear();
    }
  }
}

// Target Test Type: avx, avx512, avx512fp16, scalar
TEST(SquaredEuclideanMetric, TestFp16SquaredEuclideanBatch) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1024;
  const size_t BATCH_SIZE = 16;

  auto quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("SquaredEuclidean", 0, Params());
  ASSERT_TRUE(!!quantizer);
  ASSERT_EQ(0u, quantizer->init(meta, Params()));
  auto &convert_meta = quantizer->meta();

  auto batch_func_avx512fp16 =
      get_batch_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                              QuantizeType::kDefault, CpuArchType::kAVX512FP16);

  auto batch_func_avx512 =
      get_batch_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                              QuantizeType::kDefault, CpuArchType::kAVX512);

  auto batch_func_avx =
      get_batch_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                              QuantizeType::kDefault, CpuArchType::kAVX);

  auto batch_func_scalar =
      get_batch_distance_func(MetricType::kSquaredEuclidean, DataType::kFp16,
                              QuantizeType::kDefault, CpuArchType::kScalar);

  ailego::NumericalVector<float> query_vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    query_vec[j] = dist(gen);
  }

  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta_quantizer;

  std::string query_out;
  ASSERT_EQ(0, quantizer->quantize(query_vec.data(), qmeta, &query_out,
                                   &qmeta_quantizer));
  ASSERT_EQ(qmeta_quantizer.dimension(), convert_meta.dimension());

  std::vector<ailego::NumericalVector<float>> doc_vecs;
  std::vector<std::string> doc_outs;
  for (size_t i = 0; i < COUNT; ++i) {
    ailego::NumericalVector<float> doc_vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      doc_vec[j] = dist(gen);
    }
    doc_vecs.push_back(doc_vec);

    std::string doc_out;
    ASSERT_EQ(0, quantizer->quantize(doc_vec.data(), qmeta, &doc_out,
                                     &qmeta_quantizer));
    ASSERT_EQ(qmeta_quantizer.dimension(), convert_meta.dimension());

    doc_outs.push_back(doc_out);

    if (doc_vecs.size() == BATCH_SIZE) {
      std::vector<const void *> doc_ptrs(BATCH_SIZE);
      for (size_t k = 0; k < BATCH_SIZE; ++k) {
        doc_ptrs[k] = doc_outs[k].data();
      }

      std::vector<float> score_avx512fp16(BATCH_SIZE, 0.0f);
      std::vector<float> score_avx512(BATCH_SIZE, 0.0f);
      std::vector<float> score_avx(BATCH_SIZE, 0.0f);
      std::vector<float> score_scalar(BATCH_SIZE, 0.0f);

      batch_func_avx512fp16(doc_ptrs.data(), query_out.data(),
                            qmeta_quantizer.dimension(), BATCH_SIZE,
                            &score_avx512fp16[0]);

      batch_func_avx512(doc_ptrs.data(), query_out.data(),
                        qmeta_quantizer.dimension(), BATCH_SIZE,
                        &score_avx512[0]);

      batch_func_avx(doc_ptrs.data(), query_out.data(),
                     qmeta_quantizer.dimension(), BATCH_SIZE, &score_avx[0]);

      batch_func_scalar(doc_ptrs.data(), query_out.data(),
                        qmeta_quantizer.dimension(), BATCH_SIZE,
                        &score_scalar[0]);

      for (size_t j = 0; j < BATCH_SIZE; ++j) {
        float epsilon = 0.2;
        ASSERT_NEAR(score_scalar[j], score_avx512fp16[j], epsilon);
        ASSERT_NEAR(score_scalar[j], score_avx512[j], epsilon);
        ASSERT_NEAR(score_scalar[j], score_avx[j], epsilon);
      }

      doc_vecs.clear();
      doc_outs.clear();
    }
  }
}
