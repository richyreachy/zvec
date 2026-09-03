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

constexpr uint32_t kDimension = 35;
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

HNSWIndexParam::Pointer MakeParam(MetricType metric,
                                  bool enable_rotate = false) {
  return HNSWIndexParamBuilder()
      .with_metric_type(metric)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(kDimension)
      .with_is_sparse(false)
      .with_m(16)
      .with_ef_construction(100)
      .with_quantizer_param(QuantizerParam(QuantizerType::kInt8, enable_rotate))
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

void CheckTurboAddSearchReopen(MetricType metric, const std::string &path) {
  zvec::test_util::RemoveTestFiles(path);
  auto vectors = RandomVectors();
  auto param = MakeParam(metric);

  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ("Int8Quantizer", index->index_searcher()->meta().quantizer_name());
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
    EXPECT_NEAR(vectors[37][i], fetched_vector[i], 1e-2f);
  }

  ASSERT_EQ(0, index->close());

  auto reopened = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, reopened);
  ASSERT_EQ(0,
            reopened->open(path, {StorageOptions::StorageType::kMMAP, false}));
  EXPECT_EQ("Int8Quantizer",
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

TEST(HnswTurboInt8Index, CosineAddSearchReopenFetch) {
  CheckTurboAddSearchReopen(MetricType::kCosine,
                            "hnsw_turbo_int8_cosine.index");
}

TEST(HnswTurboInt8Index, L2AddSearchReopenFetch) {
  CheckTurboAddSearchReopen(MetricType::kL2sq, "hnsw_turbo_int8_l2.index");
}

TEST(HnswTurboInt8Index, UnsupportedCombinationsUseLegacyPipeline) {
  for (const auto &[metric, rotate] : std::vector<std::pair<MetricType, bool>>{
           {MetricType::kCosine, true}, {MetricType::kInnerProduct, false}}) {
    auto index = IndexFactory::CreateAndInitIndex(*MakeParam(metric, rotate));
    ASSERT_NE(nullptr, index);
    EXPECT_TRUE(index->index_searcher()->meta().quantizer_name().empty());
  }
}

TEST(HnswTurboInt8Index, MergePreservesTurboLayout) {
  const std::string source_path{"hnsw_turbo_int8_merge_source.index"};
  const std::string target_path{"hnsw_turbo_int8_merge_target.index"};
  zvec::test_util::RemoveTestFiles(source_path);
  zvec::test_util::RemoveTestFiles(target_path);
  auto vectors = RandomVectors();
  auto param = MakeParam(MetricType::kL2sq);

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

TEST(HnswTurboInt8Index, LegacyLayoutReopenFallsBack) {
  const std::string path{"hnsw_int8_legacy_layout.index"};
  zvec::test_util::RemoveTestFiles(path);
  auto vectors = RandomVectors();
  BuildLegacyInt8Hnsw(path, MetricType::kL2sq, vectors);
  if (::testing::Test::HasFatalFailure()) {
    return;
  }

  auto index = IndexFactory::CreateAndInitIndex(*MakeParam(MetricType::kL2sq));
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
