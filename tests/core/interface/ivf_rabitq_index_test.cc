// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <cmath>
#include <random>
#include <gtest/gtest.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "tests/test_util.h"

namespace zvec::core_interface {
namespace {

IVFIndexParam::Pointer RabitqParam(MetricType metric, int bits = 7) {
  return IVFIndexParamBuilder()
      .with_dimension(128)
      .with_metric_type(metric)
      .with_data_type(DataType::DT_FP32)
      .with_n_list(4)
      .with_n_iters(3)
      .with_total_bits(bits)
      .with_sample_count(80)
      .with_quantizer_param(QuantizerParam(QuantizerType::kRabitq))
      .build();
}

TEST(IVFRabitqIntegration, ParamsRoundTrip) {
  auto param = RabitqParam(MetricType::kCosine, 9);
  auto decoded = std::dynamic_pointer_cast<IVFIndexParam>(
      IndexFactory::DeserializeIndexParamFromJson(param->serialize_to_json()));
  ASSERT_NE(nullptr, decoded);
  EXPECT_EQ(IndexType::kIVF, decoded->index_type);
  EXPECT_EQ(QuantizerType::kRabitq, decoded->quantizer_param->type);
  EXPECT_EQ(4, decoded->nlist);
  EXPECT_EQ(3, decoded->niters);
  EXPECT_EQ(9, decoded->total_bits);
  EXPECT_EQ(80, decoded->sample_count);
}

#if RABITQ_SUPPORTED
class IVFRabitqIntegrationTest
    : public testing::TestWithParam<std::tuple<MetricType, int>> {};

TEST_P(IVFRabitqIntegrationTest, TrainSearchAndReopen) {
  const auto metric = std::get<0>(GetParam());
  const auto bits = std::get<1>(GetParam());
  auto param = RabitqParam(metric, bits);
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, std::dynamic_pointer_cast<IVFIndex>(index));
  const std::string path = "ivf_integrated_rabitq.index";
  test_util::RemoveTestFiles(path);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  std::mt19937 rng(2026);
  std::normal_distribution<float> normal;
  std::vector<std::vector<float>> vectors(80, std::vector<float>(128));
  for (size_t i = 0; i < vectors.size(); ++i) {
    float norm = 0;
    for (auto &v : vectors[i]) {
      v = normal(rng);
      norm += v * v;
    }
    for (auto &v : vectors[i]) v /= std::sqrt(norm);
    // Sparse public IDs exercise persisted key mapping.
    ASSERT_EQ(0, index->add(VectorData{DenseVector{vectors[i].data()}}, i * 3));
  }
  ASSERT_EQ(0, index->train());
  ASSERT_EQ(0, index->train());
  EXPECT_EQ(80, index->get_doc_count());
  auto query = IVFQueryParamBuilder().with_topk(5).with_nprobe(4).build();
  SearchResult before;
  auto input = VectorData{DenseVector{vectors[7].data()}};
  ASSERT_EQ(0, index->search(input, query, &before));
  ASSERT_EQ(5, before.doc_list_.size());
  EXPECT_EQ(21, before.doc_list_[0].key());
  if (bits >= 7) {
    EXPECT_NEAR(metric == MetricType::kInnerProduct ? 1.0f : 0.0f,
                before.doc_list_[0].score(), 0.1f);
  }
  // Invalid probes must not silently reuse the previous thread-local setting.
  for (int nprobe : {-1, 0}) {
    query->nprobe = nprobe;
    SearchResult invalid;
    EXPECT_EQ(core::IndexError_InvalidArgument,
              index->search(input, query, &invalid));
  }
  query->nprobe = 4;
  auto group_query = IVFQueryParamBuilder().with_topk(8).with_nprobe(4).build();
  group_query->group_by_param = std::make_shared<GroupByParam>();
  group_query->group_by_param->group_count = 4;
  group_query->group_by_param->group_topk = 2;
  group_query->group_by_param->group_by = [](uint64_t key) {
    return std::to_string(key % 4);
  };
  SearchResult grouped;
  ASSERT_EQ(0, index->search(input, group_query, &grouped));
  ASSERT_EQ(4, grouped.group_doc_list_.size());
  for (const auto &group : grouped.group_doc_list_) {
    ASSERT_EQ(2, group.docs().size());
    for (const auto &doc : group.docs()) {
      EXPECT_EQ(std::to_string(doc.key() % 4), group.group_id());
    }
  }
  auto plain_param = RabitqParam(metric);
  plain_param->quantizer_param = std::make_shared<QuantizerParam>();
  auto plain = IndexFactory::CreateAndInitIndex(*plain_param);
  ASSERT_NE(nullptr, plain);
  const std::string plain_path = path + ".plain";
  test_util::RemoveTestFiles(plain_path);
  ASSERT_EQ(
      0, plain->open(plain_path, {StorageOptions::StorageType::kMMAP, true}));
  for (size_t i = 0; i < vectors.size(); ++i) {
    ASSERT_EQ(0, plain->add(VectorData{DenseVector{vectors[i].data()}}, i * 3));
  }
  ASSERT_EQ(0, plain->train());
  SearchResult rejected_group;
  EXPECT_EQ(core::IndexError_Unsupported,
            plain->search(input, group_query, &rejected_group));
  // Alternate backends in one thread: both use the public IVF context slot.
  for (int attempt = 0; attempt < 2; ++attempt) {
    SearchResult result;
    ASSERT_EQ(0, plain->search(input, query, &result));
    ASSERT_EQ(5, result.doc_list_.size());
    EXPECT_EQ(21, result.doc_list_[0].key());
    ASSERT_EQ(0, index->search(input, group_query, &result));
    ASSERT_EQ(4, result.group_doc_list_.size());
    ASSERT_EQ(0, index->search(input, query, &result));
    ASSERT_EQ(5, result.doc_list_.size());
    EXPECT_EQ(21, result.doc_list_[0].key());
  }
  // Collection optimize builds IVF from raw/decoded Flat or IVF providers.
  const std::string merge_path = path + ".merge";
  test_util::RemoveTestFiles(merge_path);
  auto merged = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, merged);
  ASSERT_EQ(
      0, merged->open(merge_path, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(0, merged->merge({plain}, IndexFilter{}));
  SearchResult merged_result;
  ASSERT_EQ(0, merged->search(input, query, &merged_result));
  ASSERT_EQ(5, merged_result.doc_list_.size());
  EXPECT_EQ(7, merged_result.doc_list_[0].key());
  ASSERT_EQ(0, merged->close());
  test_util::RemoveTestFiles(merge_path);
  ASSERT_EQ(0, plain->close());
  test_util::RemoveTestFiles(plain_path);
  query->fetch_vector = true;
  SearchResult unsupported;
  EXPECT_NE(0, index->search(input, query, &unsupported));
  query->fetch_vector = false;
  ASSERT_EQ(0, index->close());
  index.reset();

  // Detect the persisted RaBitQ postings even with default quantization.
  param->quantizer_param = std::make_shared<QuantizerParam>();
  auto reopened = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, reopened);
  for (int attempt = 0; attempt < 2; ++attempt) {
    ASSERT_EQ(
        0, reopened->open(path, {StorageOptions::StorageType::kMMAP, false}));
    SearchResult after;
    ASSERT_EQ(0, reopened->search(input, query, &after));
    ASSERT_EQ(before.doc_list_.size(), after.doc_list_.size());
    for (size_t i = 0; i < before.doc_list_.size(); ++i) {
      EXPECT_EQ(before.doc_list_[i].key(), after.doc_list_[i].key());
      EXPECT_FLOAT_EQ(before.doc_list_[i].score(), after.doc_list_[i].score());
    }
    ASSERT_EQ(0, reopened->search(input, group_query, &after));
    ASSERT_EQ(4, after.group_doc_list_.size());
    ASSERT_EQ(0, reopened->close());
  }
  // Both public APIs must return identical results from the same postings.
  auto legacy_param =
      IVFRabitqIndexParamBuilder()
          .with_dimension(128)
          .with_data_type(DataType::DT_FP32)
          .with_metric_type(metric)
          .with_n_list(4)
          .with_total_bits(bits)
          .with_quantizer_param(QuantizerParam(QuantizerType::kRabitq))
          .build();
  auto legacy = IndexFactory::CreateAndInitIndex(*legacy_param);
  ASSERT_NE(nullptr, legacy);
  ASSERT_EQ(0, legacy->open(path, {StorageOptions::StorageType::kMMAP, false}));
  auto legacy_query = std::make_shared<IVFRabitqQueryParam>();
  legacy_query->nprobe = 4;
  legacy_query->topk = 5;
  SearchResult legacy_result;
  ASSERT_EQ(0, legacy->search(input, legacy_query, &legacy_result));
  ASSERT_EQ(before.doc_list_.size(), legacy_result.doc_list_.size());
  for (size_t i = 0; i < before.doc_list_.size(); ++i) {
    EXPECT_EQ(before.doc_list_[i].key(), legacy_result.doc_list_[i].key());
    EXPECT_FLOAT_EQ(before.doc_list_[i].score(),
                    legacy_result.doc_list_[i].score());
  }
  legacy_query->topk = group_query->topk;
  legacy_query->group_by_param = group_query->group_by_param;
  ASSERT_EQ(0, legacy->search(input, legacy_query, &legacy_result));
  ASSERT_EQ(grouped.group_doc_list_.size(),
            legacy_result.group_doc_list_.size());
  for (size_t g = 0; g < grouped.group_doc_list_.size(); ++g) {
    const auto &expected = grouped.group_doc_list_[g];
    const auto &actual = legacy_result.group_doc_list_[g];
    EXPECT_EQ(expected.group_id(), actual.group_id());
    ASSERT_EQ(expected.docs().size(), actual.docs().size());
    for (size_t i = 0; i < expected.docs().size(); ++i) {
      EXPECT_EQ(expected.docs()[i].key(), actual.docs()[i].key());
      EXPECT_FLOAT_EQ(expected.docs()[i].score(), actual.docs()[i].score());
    }
  }
  ASSERT_EQ(0, legacy->close());
  test_util::RemoveTestFiles(path);
}

INSTANTIATE_TEST_SUITE_P(
    MetricsAndBits, IVFRabitqIntegrationTest,
    testing::Combine(testing::Values(MetricType::kL2sq,
                                     MetricType::kInnerProduct,
                                     MetricType::kCosine),
                     testing::Values(1, 7, 9)));
#else
TEST(IVFRabitqIntegration, UnsupportedPlatform) {
  EXPECT_EQ(nullptr,
            IndexFactory::CreateAndInitIndex(*RabitqParam(MetricType::kL2sq)));
}
#endif
}  // namespace
}  // namespace zvec::core_interface
