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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/hnsw/hnsw_params.h"
#include "tests/test_util.h"

using namespace zvec::core_interface;

namespace {

constexpr uint32_t kDimension = 36;
constexpr size_t kVectorCount = 200;
constexpr uint32_t kTopK = 10;

std::vector<std::vector<float>> RandomVectors() {
  std::mt19937 gen(2026);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<std::vector<float>> vectors(kVectorCount,
                                          std::vector<float>(kDimension));
  for (auto &vector : vectors) {
    float norm = 0.0f;
    for (float &value : vector) {
      value = dist(gen);
      norm += value * value;
    }
    norm = std::sqrt(norm);
    for (float &value : vector) {
      value /= norm;
    }
  }
  return vectors;
}

HNSWIndexParam::Pointer MakeParam(
    MetricType metric, QuantizerType quantizer = QuantizerType::kInt8,
    bool enable_rotate = false) {
  return HNSWIndexParamBuilder()
      .with_metric_type(metric)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(kDimension)
      .with_is_sparse(false)
      .with_m(16)
      .with_ef_construction(100)
      .with_quantizer_param(QuantizerParam(quantizer, enable_rotate))
      .build();
}

HNSWIndexParam::Pointer MakeDefaultParam(MetricType metric) {
  return HNSWIndexParamBuilder()
      .with_metric_type(metric)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(kDimension)
      .with_is_sparse(false)
      .with_m(16)
      .with_ef_construction(100)
      .build();
}

std::vector<std::pair<uint32_t, float>> SearchRows(
    Index *index, const std::vector<float> &query, bool linear,
    bool fetch_vector = false) {
  auto query_param = HNSWQueryParamBuilder()
                         .with_topk(kTopK)
                         .with_ef_search(100)
                         .with_is_linear(linear)
                         .with_fetch_vector(fetch_vector)
                         .build();
  VectorData query_data{DenseVector{query.data()}};
  SearchResult result;
  EXPECT_EQ(0, index->search(query_data, query_param, &result));
  std::vector<std::pair<uint32_t, float>> rows;
  for (const auto &doc : result.doc_list_) {
    rows.emplace_back(doc.key(), doc.score());
  }
  if (fetch_vector) {
    EXPECT_EQ(rows.size(), result.reverted_vector_list_.size());
  }
  return rows;
}

void AddVectors(Index *index, const std::vector<std::vector<float>> &vectors) {
  for (size_t i = 0; i < vectors.size(); ++i) {
    VectorData vector_data{DenseVector{vectors[i].data()}};
    ASSERT_EQ(0, index->add(vector_data, static_cast<uint32_t>(i)));
  }
}

void CheckTurboAddSearchReopen(MetricType metric, QuantizerType quantizer,
                               const char *quantizer_name,
                               float fetch_tolerance, const std::string &path) {
  zvec::test_util::RemoveTestFiles(path);
  auto vectors = RandomVectors();
  auto param = MakeParam(metric, quantizer);

  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(quantizer_name, index->index_searcher()->meta().quantizer_name());
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  AddVectors(index.get(), vectors);
  ASSERT_EQ(0, index->train());

  auto linear_rows = SearchRows(index.get(), vectors[37], true, true);
  ASSERT_EQ(kTopK, linear_rows.size());
  EXPECT_EQ(37U, linear_rows[0].first);

  auto ann_rows = SearchRows(index.get(), vectors[101], false);
  ASSERT_EQ(kTopK, ann_rows.size());
  EXPECT_EQ(101U, ann_rows[0].first);

  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(37, &fetched));
  const auto *fetched_vector = reinterpret_cast<const float *>(
      std::get<DenseVectorBuffer>(fetched.vector_buffer).data.data());
  for (uint32_t i = 0; i < kDimension; ++i) {
    EXPECT_NEAR(vectors[37][i], fetched_vector[i], fetch_tolerance);
  }

  ASSERT_EQ(0, index->close());

  auto reopened = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, reopened);
  ASSERT_EQ(0,
            reopened->open(path, {StorageOptions::StorageType::kMMAP, false}));
  EXPECT_EQ(quantizer_name,
            reopened->index_searcher()->meta().quantizer_name());
  auto reopened_rows = SearchRows(reopened.get(), vectors[37], true);
  ASSERT_EQ(linear_rows.size(), reopened_rows.size());
  for (size_t i = 0; i < linear_rows.size(); ++i) {
    EXPECT_EQ(linear_rows[i].first, reopened_rows[i].first);
    EXPECT_FLOAT_EQ(linear_rows[i].second, reopened_rows[i].second);
  }
  ASSERT_EQ(0, reopened->close());
  zvec::test_util::RemoveTestFiles(path);
}

void BuildLegacyFp32Hnsw(const std::string &path,
                         const std::vector<std::vector<float>> &vectors) {
  namespace core = zvec::core;
  core::IndexMeta legacy_meta(core::IndexMeta::DT_FP32, kDimension);
  legacy_meta.set_meta_type(core::IndexMeta::MetaType::MT_DENSE);
  legacy_meta.set_metric("SquaredEuclidean", 0, zvec::ailego::Params());
  ASSERT_TRUE(legacy_meta.quantizer_name().empty());

  zvec::ailego::Params params;
  params.set(core::PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 16U);
  params.set(core::PARAM_HNSW_STREAMER_SCALING_FACTOR, 16U);
  params.set(core::PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100U);
  params.set(core::PARAM_HNSW_STREAMER_EF, 100U);
  params.set(core::PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);
  auto streamer = core::IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, streamer->init(legacy_meta, params));
  auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(zvec::ailego::Params()));
  ASSERT_EQ(0, storage->open(path, true));
  ASSERT_EQ(0, streamer->open(storage));

  auto context = streamer->create_context();
  core::IndexQueryMeta qmeta(core::IndexMeta::DT_FP32, kDimension);
  for (size_t i = 0; i < vectors.size(); ++i) {
    ASSERT_EQ(0, streamer->add_with_id_impl(static_cast<uint32_t>(i),
                                            vectors[i].data(), qmeta, context));
  }
  ASSERT_EQ(0, streamer->flush(0));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, storage->close());
}

void BuildLegacyInt8Hnsw(const std::string &path, MetricType metric,
                         const std::vector<std::vector<float>> &vectors) {
  namespace core = zvec::core;
  core::IndexMeta raw_meta(core::IndexMeta::DT_FP32, kDimension);
  raw_meta.set_meta_type(core::IndexMeta::MetaType::MT_DENSE);
  raw_meta.set_metric(
      metric == MetricType::kCosine ? "Cosine" : "SquaredEuclidean", 0,
      zvec::ailego::Params());
  const char *converter_name = metric == MetricType::kCosine
                                   ? "CosineInt8Converter"
                                   : "Int8StreamingConverter";
  raw_meta.set_converter(converter_name, 0, zvec::ailego::Params());
  auto converter = core::IndexFactory::CreateConverter(converter_name);
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(raw_meta, zvec::ailego::Params()));

  core::IndexMeta legacy_meta = converter->meta();
  ASSERT_TRUE(legacy_meta.quantizer_name().empty());
  auto reformer =
      core::IndexFactory::CreateReformer(legacy_meta.reformer_name());
  ASSERT_NE(nullptr, reformer);
  ASSERT_EQ(0, reformer->init(legacy_meta.reformer_params()));

  zvec::ailego::Params params;
  params.set(core::PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 16U);
  params.set(core::PARAM_HNSW_STREAMER_SCALING_FACTOR, 16U);
  params.set(core::PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100U);
  params.set(core::PARAM_HNSW_STREAMER_EF, 100U);
  params.set(core::PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);
  auto streamer = core::IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, streamer->init(legacy_meta, params));
  auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(zvec::ailego::Params()));
  ASSERT_EQ(0, storage->open(path, true));
  ASSERT_EQ(0, streamer->open(storage));

  auto context = streamer->create_context();
  core::IndexQueryMeta raw_qmeta(core::IndexMeta::DT_FP32, kDimension);
  for (size_t i = 0; i < vectors.size(); ++i) {
    std::string converted;
    core::IndexQueryMeta converted_meta;
    ASSERT_EQ(0, reformer->convert(vectors[i].data(), raw_qmeta, &converted,
                                   &converted_meta));
    ASSERT_EQ(0, streamer->add_with_id_impl(static_cast<uint32_t>(i),
                                            converted.data(), converted_meta,
                                            context));
  }
  ASSERT_EQ(0, streamer->flush(0));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, storage->close());
}

}  // namespace

TEST(HnswTurboQuantizerIndex, Int8CosineAddSearchReopenFetch) {
  CheckTurboAddSearchReopen(MetricType::kCosine, QuantizerType::kInt8,
                            "Int8Quantizer", 1e-2f,
                            "hnsw_turbo_int8_cosine.index");
}

TEST(HnswTurboQuantizerIndex, L2AddSearchReopenFetch) {
  CheckTurboAddSearchReopen(MetricType::kL2sq, QuantizerType::kInt8,
                            "Int8Quantizer", 1e-2f, "hnsw_turbo_int8_l2.index");
}

TEST(HnswTurboQuantizerIndex, DefaultUsesFp32TurboQuantizer) {
  for (MetricType metric :
       {MetricType::kL2sq, MetricType::kCosine, MetricType::kInnerProduct}) {
    auto index = IndexFactory::CreateAndInitIndex(*MakeDefaultParam(metric));
    ASSERT_NE(nullptr, index);
    EXPECT_EQ("Fp32Quantizer",
              index->index_searcher()->meta().quantizer_name());
  }
}

TEST(HnswTurboQuantizerIndex, InnerProductScoreAndRadiusUseCallerSpace) {
  const std::string path{"hnsw_turbo_fp32_ip_radius.index"};
  zvec::test_util::RemoveTestFiles(path);
  auto index = IndexFactory::CreateAndInitIndex(
      *MakeDefaultParam(MetricType::kInnerProduct));
  ASSERT_NE(nullptr, index);
  ASSERT_EQ("Fp32Quantizer", index->index_searcher()->meta().quantizer_name());
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));

  std::vector<std::vector<float>> vectors(3, std::vector<float>(kDimension));
  vectors[0][0] = 1.0f;
  vectors[1][0] = 0.75f;
  vectors[2][0] = 0.25f;
  AddVectors(index.get(), vectors);

  auto query_param = HNSWQueryParamBuilder()
                         .with_topk(3)
                         .with_ef_search(100)
                         .with_is_linear(true)
                         .with_radius(0.5f)
                         .build();
  std::vector<float> query(kDimension);
  query[0] = 1.0f;
  SearchResult result;
  ASSERT_EQ(0, index->search(VectorData{DenseVector{query.data()}}, query_param,
                             &result));
  ASSERT_EQ(2U, result.doc_list_.size());
  EXPECT_EQ(0U, result.doc_list_[0].key());
  EXPECT_FLOAT_EQ(1.0f, result.doc_list_[0].score());
  EXPECT_EQ(1U, result.doc_list_[1].key());
  EXPECT_FLOAT_EQ(0.75f, result.doc_list_[1].score());

  ASSERT_EQ(0, index->close());
  zvec::test_util::RemoveTestFiles(path);
}

TEST(HnswTurboQuantizerIndex, SupportedQuantizersUseTurboForAllMetrics) {
  const std::vector<std::pair<QuantizerType, const char *>> quantizers{
      {QuantizerType::kNone, "Fp32Quantizer"},
      {QuantizerType::kFP16, "Fp16Quantizer"},
      {QuantizerType::kInt8, "Int8Quantizer"},
      {QuantizerType::kInt4, "Int4Quantizer"},
  };
  for (const auto &[quantizer, quantizer_name] : quantizers) {
    for (MetricType metric :
         {MetricType::kL2sq, MetricType::kCosine, MetricType::kInnerProduct}) {
      auto index =
          IndexFactory::CreateAndInitIndex(*MakeParam(metric, quantizer));
      ASSERT_NE(nullptr, index);
      EXPECT_EQ(quantizer_name,
                index->index_searcher()->meta().quantizer_name());
    }
  }
}

TEST(HnswTurboQuantizerIndex, AdditionalTurboQuantizersAddSearchReopenFetch) {
  CheckTurboAddSearchReopen(MetricType::kL2sq, QuantizerType::kNone,
                            "Fp32Quantizer", 1e-6f, "hnsw_turbo_fp32_l2.index");
  CheckTurboAddSearchReopen(MetricType::kCosine, QuantizerType::kFP16,
                            "Fp16Quantizer", 1e-3f,
                            "hnsw_turbo_fp16_cosine.index");
  CheckTurboAddSearchReopen(MetricType::kInnerProduct, QuantizerType::kInt8,
                            "Int8Quantizer", 1e-2f, "hnsw_turbo_int8_ip.index");
  CheckTurboAddSearchReopen(MetricType::kL2sq, QuantizerType::kInt4,
                            "Int4Quantizer", 2e-1f, "hnsw_turbo_int4_l2.index");
}

TEST(HnswTurboQuantizerIndex, UnsupportedCombinationsUseLegacyPipeline) {
  auto rotated = IndexFactory::CreateAndInitIndex(
      *MakeParam(MetricType::kCosine, QuantizerType::kInt8, true));
  ASSERT_NE(nullptr, rotated);
  EXPECT_TRUE(rotated->index_searcher()->meta().quantizer_name().empty());

  auto mips = IndexFactory::CreateAndInitIndex(
      *MakeParam(MetricType::kMIPSL2sq, QuantizerType::kNone));
  ASSERT_NE(nullptr, mips);
  EXPECT_TRUE(mips->index_searcher()->meta().quantizer_name().empty());
}

TEST(HnswTurboQuantizerIndex, MergePreservesTurboLayout) {
  const std::string source_path{"hnsw_turbo_int8_merge_source.index"};
  const std::string target_path{"hnsw_turbo_int8_merge_target.index"};
  zvec::test_util::RemoveTestFiles(source_path);
  zvec::test_util::RemoveTestFiles(target_path);
  auto vectors = RandomVectors();
  auto param = MakeParam(MetricType::kL2sq, QuantizerType::kInt8);

  auto source = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, source);
  ASSERT_EQ(
      0, source->open(source_path, {StorageOptions::StorageType::kMMAP, true}));
  AddVectors(source.get(), vectors);

  auto target = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, target);
  ASSERT_EQ(
      0, target->open(target_path, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(0, target->merge({source}, IndexFilter()));
  EXPECT_EQ(kVectorCount, target->get_doc_count());
  EXPECT_EQ("Int8Quantizer", target->index_searcher()->meta().quantizer_name());

  auto rows = SearchRows(target.get(), vectors[73], true);
  ASSERT_EQ(kTopK, rows.size());
  EXPECT_EQ(73U, rows[0].first);

  ASSERT_EQ(0, target->close());
  ASSERT_EQ(0, source->close());
  zvec::test_util::RemoveTestFiles(source_path);
  zvec::test_util::RemoveTestFiles(target_path);
}

TEST(HnswTurboQuantizerIndex, LegacyLayoutReopenFallsBack) {
  const std::string path{"hnsw_int8_legacy_layout.index"};
  zvec::test_util::RemoveTestFiles(path);
  auto vectors = RandomVectors();
  BuildLegacyInt8Hnsw(path, MetricType::kL2sq, vectors);
  if (::testing::Test::HasFatalFailure()) {
    return;
  }

  auto index = IndexFactory::CreateAndInitIndex(
      *MakeParam(MetricType::kL2sq, QuantizerType::kInt8));
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, false}));
  EXPECT_TRUE(index->index_searcher()->meta().quantizer_name().empty());
  auto rows = SearchRows(index.get(), vectors[37], true);
  ASSERT_EQ(kTopK, rows.size());
  EXPECT_EQ(37U, rows[0].first);

  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(37, &fetched));
  const auto *fetched_vector = reinterpret_cast<const float *>(
      std::get<DenseVectorBuffer>(fetched.vector_buffer).data.data());
  for (uint32_t i = 0; i < kDimension; ++i) {
    EXPECT_NEAR(vectors[37][i], fetched_vector[i], 5e-2f);
  }
  ASSERT_EQ(0, index->close());
  zvec::test_util::RemoveTestFiles(path);
}

TEST(HnswTurboQuantizerIndex, LegacyFp32LayoutReopenFallsBack) {
  const std::string path{"hnsw_fp32_legacy_layout.index"};
  zvec::test_util::RemoveTestFiles(path);
  auto vectors = RandomVectors();
  BuildLegacyFp32Hnsw(path, vectors);
  if (::testing::Test::HasFatalFailure()) {
    return;
  }

  auto index = IndexFactory::CreateAndInitIndex(
      *MakeParam(MetricType::kL2sq, QuantizerType::kNone));
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, false}));
  EXPECT_TRUE(index->index_searcher()->meta().quantizer_name().empty());
  auto rows = SearchRows(index.get(), vectors[37], true);
  ASSERT_EQ(kTopK, rows.size());
  EXPECT_EQ(37U, rows[0].first);

  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(37, &fetched));
  const auto *fetched_vector = reinterpret_cast<const float *>(
      std::get<DenseVectorBuffer>(fetched.vector_buffer).data.data());
  for (uint32_t i = 0; i < kDimension; ++i) {
    EXPECT_FLOAT_EQ(vectors[37][i], fetched_vector[i]);
  }
  ASSERT_EQ(0, index->close());
  zvec::test_util::RemoveTestFiles(path);
}
