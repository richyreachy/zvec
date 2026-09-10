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
#include <cstring>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/framework/index_segment_storage.h>
#include "ivf_builder.h"
#include "ivf_searcher.h"
#include "ivf_streamer.h"

namespace zvec {
namespace core {
namespace {

// The dimension produces INT4 records with a byte count not divisible by four;
// queries and encoded postings have different strides and block padding.
constexpr uint32_t kDimension = 18;
constexpr size_t kVectorCount = 67;
constexpr uint32_t kTopK = 11;
constexpr uint32_t kQueryCount = 3;

uint64_t Key(size_t i) {
  return 100 + 13 * i;
}

using TurboCase = std::tuple<const char *, const char *>;

// Preserve the file's structure while withholding the serialized codebook.
class MissingQuantizerStateDumper : public IndexDumper {
 public:
  MissingQuantizerStateDumper()
      : dumper_(IndexFactory::CreateDumper("FileDumper")) {}

  int init(const ailego::Params &params) override {
    return dumper_->init(params);
  }
  int cleanup() override {
    return dumper_->cleanup();
  }
  int create(const std::string &path) override {
    return dumper_->create(path);
  }
  int close() override {
    return dumper_->close();
  }
  int append(const std::string &id, size_t size, size_t padding,
             uint32_t crc) override {
    return dumper_->append(
        id == IVF_TURBO_QUANTIZER_SEG_ID ? "missing_state" : id, size, padding,
        crc);
  }
  size_t write(const void *data, size_t size) override {
    return dumper_->write(data, size);
  }
  uint32_t magic() const override {
    return dumper_->magic();
  }

 private:
  IndexDumper::Pointer dumper_;
};

class IVFTurboTest : public testing::TestWithParam<TurboCase> {
 protected:
  void SetUp() override {
    const auto *info = testing::UnitTest::GetInstance()->current_test_info();
    path_ = std::string("ivf_turbo_") + info->name() + ".index";
    std::replace(path_.begin(), path_.end(), '/', '_');
    ailego::File::RemovePath(path_);

    meta_.set_meta(IndexMeta::DataType::DT_FP32, kDimension);
    meta_.set_metric(std::get<1>(GetParam()), 0, ailego::Params());
    ailego::Params quantizer_params;
    if (std::string(std::get<0>(GetParam())) == "PqInt4Quantizer") {
      quantizer_params.set("num_chunk", 3U);
      quantizer_params.set("use_zero_mean", true);
      quantizer_params.set("thread_count", 1U);
    }
    meta_.set_quantizer(std::get<0>(GetParam()), 0, quantizer_params);
    quantizer_ = IndexFactory::CreateQuantizer(std::get<0>(GetParam()));
    ASSERT_NE(nullptr, quantizer_);
    ASSERT_EQ(0, quantizer_->init(meta_, quantizer_params));

    std::mt19937 random(20260910);
    std::uniform_real_distribution<float> value(-2.0f, 3.0f);
    auto holder =
        std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
            kDimension);
    for (size_t i = 0; i < kVectorCount; ++i) {
      ailego::NumericalVector<float> row(kDimension);
      for (uint32_t j = 0; j < kDimension; ++j) {
        // Vary the norms so cosine exercises normalization and its extra
        // metadata instead of accidentally receiving unit vectors.
        row[j] = value(random) * (0.4f + 0.07f * (i % 9));
      }
      vectors_.emplace_back(row.data(), row.data() + kDimension);
      holder->emplace(Key(i), row);
    }
    for (size_t i : {7U, 31U, 52U}) {
      queries_.insert(queries_.end(), vectors_[i].begin(), vectors_[i].end());
    }
    queries_[kDimension + 2] += 0.17f;

    ailego::Params params;
    params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");
    params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
    params.set(PARAM_IVF_BUILDER_THREAD_COUNT, 1U);
    params.set(PARAM_IVF_BUILDER_BLOCK_VECTOR_COUNT, 8U);
    IVFBuilder builder;
    ASSERT_EQ(0, builder.init(meta_, params, quantizer_));
    ASSERT_EQ(0, builder.train(IndexThreads::Pointer(), holder));
    ASSERT_NE(nullptr, builder.centroid_index());
    EXPECT_EQ(4U, builder.centroid_index()->centroids_count());
    EXPECT_EQ(IndexMeta::DataType::DT_FP32,
              builder.centroid_index()->meta().data_type());
    ASSERT_EQ(0, builder.build(IndexThreads::Pointer(), holder));
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, dumper->create(path_));
    ASSERT_EQ(0, builder.dump(dumper));
    ASSERT_EQ(0, dumper->close());
    EXPECT_EQ(kVectorCount, builder.stats().dumped_count());
    if (quantizer_->require_train()) {
      auto missing_state = std::make_shared<MissingQuantizerStateDumper>();
      ASSERT_EQ(0, missing_state->create(path_ + ".missing"));
      ASSERT_EQ(0, builder.dump(missing_state));
      ASSERT_EQ(0, missing_state->close());
    }

    for (const auto &row : vectors_) {
      codes_.emplace_back(quantizer_->quantized_datapoint_vector_length(),
                          '\0');
      quantizer_->quantize_data(row.data(), codes_.back().data());
    }
  }

  void TearDown() override {
    ailego::File::RemovePath(path_);
    ailego::File::RemovePath(path_ + ".missing");
  }

  ailego::Params SearchParams() const {
    ailego::Params params;
    params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
    params.set(PARAM_IVF_SEARCHER_NPROBE, 4U);
    // Force the ANN entry point to select and scan the centroid lists.
    params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1U);
    return params;
  }

  IndexStorage::Pointer OpenStorage(const char *name) {
    auto storage = IndexFactory::CreateStorage(name);
    EXPECT_NE(nullptr, storage);
    if (!storage) {
      return nullptr;
    }
    EXPECT_EQ(0, storage->init(ailego::Params()));
    EXPECT_EQ(0, storage->open(path_, false));
    return storage;
  }

  void CheckResult(const IndexDocumentList &result, const float *query,
                   bool filtered) const {
    std::vector<std::pair<uint64_t, float>> expected;
    for (size_t i = 0; i < codes_.size(); ++i) {
      if (filtered && Key(i) % 2 == 0) {
        continue;
      }
      expected.emplace_back(Key(i),
                            quantizer_->calc_distance_dp_query_unquantized(
                                codes_[i].data(), query));
    }
    std::sort(expected.begin(), expected.end(),
              [](const auto &left, const auto &right) {
                return left.second < right.second;
              });
    ASSERT_EQ(kTopK, result.size());
    std::vector<uint64_t> seen;
    for (size_t rank = 0; rank < result.size(); ++rank) {
      SCOPED_TRACE(testing::Message() << "rank=" << rank);
      const auto found = std::find_if(
          expected.begin(), expected.end(),
          [&](const auto &row) { return row.first == result[rank].key(); });
      ASSERT_NE(expected.end(), found);
      EXPECT_EQ(seen.end(),
                std::find(seen.begin(), seen.end(), result[rank].key()));
      seen.push_back(result[rank].key());
      const float tolerance = 1e-4f * std::max(1.0f, std::abs(found->second));
      EXPECT_NEAR(found->second, result[rank].score(), tolerance);
      // Compare ranks by score so legitimate quantization ties need not use
      // the same key ordering as std::sort.
      EXPECT_NEAR(expected[rank].second, result[rank].score(), tolerance);
    }
  }

  template <class Searcher>
  void CheckSearches(Searcher *searcher) const {
    const auto original_queries = queries_;
    const IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kDimension);
    auto context = searcher->create_context();
    ASSERT_NE(nullptr, context);
    context->set_topk(kTopK);
    for (bool filtered : {false, true}) {
      context->set_filter(
          [filtered](uint64_t key) { return filtered && key % 2 == 0; });
      ASSERT_EQ(0, searcher->search_bf_impl(queries_.data(), qmeta, context));
      CheckResult(context->result(0), queries_.data(), filtered);
      ASSERT_EQ(0, searcher->search_impl(queries_.data(), qmeta, context));
      CheckResult(context->result(0), queries_.data(), filtered);
      ASSERT_EQ(0, searcher->search_bf_impl(queries_.data(), qmeta, kQueryCount,
                                            context));
      for (uint32_t q = 0; q < kQueryCount; ++q) {
        CheckResult(context->result(q), queries_.data() + q * kDimension,
                    filtered);
      }
      ASSERT_EQ(0, searcher->search_impl(queries_.data(), qmeta, kQueryCount,
                                         context));
      for (uint32_t q = 0; q < kQueryCount; ++q) {
        CheckResult(context->result(q), queries_.data() + q * kDimension,
                    filtered);
      }
    }
    EXPECT_EQ(original_queries, queries_);
  }

  void CheckDecodedProvider(IndexProvider::Pointer provider) const {
    ASSERT_NE(nullptr, provider);
    EXPECT_EQ(kVectorCount, provider->count());
    EXPECT_EQ(IndexMeta::DataType::DT_FP32, provider->data_type());
    ASSERT_EQ(kDimension * sizeof(float), provider->element_size());
    IndexQueryMeta encoded_meta;
    std::string encoded;
    ASSERT_EQ(0, quantizer_->quantize(
                     vectors_[0].data(),
                     IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
                     &encoded, &encoded_meta));
    auto iterator = provider->create_iterator();
    ASSERT_NE(nullptr, iterator);
    for (size_t i = 0; i < kVectorCount; ++i) {
      ASSERT_TRUE(iterator->is_valid());
      EXPECT_EQ(Key(i), iterator->key());
      std::string decoded;
      ASSERT_EQ(
          0, quantizer_->dequantize(codes_[i].data(), encoded_meta, &decoded));
      ASSERT_EQ(provider->element_size(), decoded.size());
      EXPECT_EQ(0,
                std::memcmp(decoded.data(), iterator->data(), decoded.size()));
      iterator->next();
    }
    EXPECT_FALSE(iterator->is_valid());
  }

  IndexMeta meta_;
  turbo::Quantizer::Pointer quantizer_;
  std::vector<std::vector<float>> vectors_;
  std::vector<float> queries_;
  std::vector<std::string> codes_;
  std::string path_;
};

TEST_P(IVFTurboTest, RawQueriesAndPostingCodesSurviveReopen) {
  // No quantizer is injected: both storage implementations must reconstruct
  // it from the dedicated persisted metadata and serialized state.
  for (const char *storage_name : {"FileReadStorage", "MMapFileReadStorage"}) {
    SCOPED_TRACE(storage_name);
    auto storage = OpenStorage(storage_name);
    ASSERT_NE(nullptr, storage);
    IVFSearcher searcher;
    ASSERT_EQ(0, searcher.init(SearchParams()));
    ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
    ASSERT_NE(nullptr, searcher.quantizer());
    EXPECT_EQ(kDimension, searcher.meta().dimension());
    EXPECT_EQ(IndexMeta::DataType::DT_FP32, searcher.meta().data_type());
    CheckDecodedProvider(searcher.create_provider());
    CheckSearches(&searcher);
    auto retained_context = searcher.create_context();
    ASSERT_NE(nullptr, retained_context);
    retained_context->set_topk(kTopK);
    const IndexQueryMeta qmeta(IndexMeta::DT_FP32, kDimension);
    ASSERT_EQ(0,
              searcher.search_impl(queries_.data(), qmeta, retained_context));
    ASSERT_EQ(0, searcher.unload());
    ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
    ASSERT_EQ(0,
              searcher.search_impl(queries_.data(), qmeta, retained_context));
    CheckResult(retained_context->result(0), queries_.data(), false);
    ASSERT_EQ(0, searcher.unload());
  }

  auto storage = OpenStorage("MMapFileReadStorage");
  ASSERT_NE(nullptr, storage);
  IVFStreamer streamer;
  ASSERT_EQ(0, streamer.init(meta_, SearchParams()));
  ASSERT_EQ(0, streamer.open(storage));
  ASSERT_NE(nullptr, streamer.quantizer());
  CheckDecodedProvider(streamer.create_provider());
  for (size_t i = 0; i < codes_.size(); ++i) {
    IndexStorage::MemoryBlock block;
    ASSERT_EQ(0,
              streamer.get_vector_by_id(static_cast<uint32_t>(Key(i)), block));
    ASSERT_NE(nullptr, block.data());
    EXPECT_EQ(0, std::memcmp(codes_[i].data(), block.data(), codes_[i].size()));
  }
  CheckSearches(&streamer);
  ASSERT_EQ(0, streamer.close());

  if (quantizer_->require_train()) {
    auto missing_state = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_NE(nullptr, missing_state);
    ASSERT_EQ(0, missing_state->init(ailego::Params()));
    ASSERT_EQ(0, missing_state->open(path_ + ".missing", false));
    IVFSearcher searcher;
    ASSERT_EQ(0, searcher.init(SearchParams()));
    EXPECT_NE(0, searcher.load(missing_state, IndexMetric::Pointer()));
    IVFStreamer missing_streamer;
    ASSERT_EQ(0, missing_streamer.init(meta_, SearchParams()));
    EXPECT_NE(0, missing_streamer.open(missing_state));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ScalarFormatsAndMetrics, IVFTurboTest,
    testing::Combine(testing::Values("Fp32Quantizer", "Fp16Quantizer",
                                     "Int8Quantizer", "Int4Quantizer"),
                     testing::Values("SquaredEuclidean", "InnerProduct",
                                     "Cosine")),
    [](const testing::TestParamInfo<TurboCase> &info) {
      return std::string(std::get<0>(info.param)) + "_" +
             std::get<1>(info.param);
    });

INSTANTIATE_TEST_SUITE_P(TrainedCodebook, IVFTurboTest,
                         testing::Values(TurboCase{"PqInt4Quantizer",
                                                   "SquaredEuclidean"}));

TEST(IVFTurboConfiguration, NullOptionalQuantizerUsesLegacyInitialization) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  IVFSearcher searcher;
  EXPECT_EQ(0, searcher.init(ailego::Params(), nullptr));
  IVFStreamer streamer;
  EXPECT_EQ(0, streamer.init(meta, ailego::Params(), nullptr));
  IVFBuilder builder;
  ailego::Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2");
  EXPECT_EQ(0, builder.init(meta, params, nullptr));
}

class FixedAngularTrainer : public IndexTrainer {
 public:
  FixedAngularTrainer(IndexMeta meta, IndexBundle::Pointer bundle)
      : meta_(std::move(meta)), bundle_(std::move(bundle)) {}
  int init(const IndexMeta &, const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int train(IndexThreads::Pointer, IndexHolder::Pointer) override {
    return 0;
  }
  int load(IndexStorage::Pointer) override {
    return 0;
  }
  int dump(const IndexDumper::Pointer &) override {
    return 0;
  }
  const IndexMeta &meta() const override {
    return meta_;
  }
  const Stats &stats() const override {
    return stats_;
  }
  IndexBundle::Pointer indexes() const override {
    return bundle_;
  }

 private:
  IndexMeta meta_;
  IndexBundle::Pointer bundle_;
  Stats stats_;
};

TEST(IVFTurboCentroids, CosineNprobeUsesEveryCoordinateAndIgnoresVectorNorm) {
  constexpr uint32_t dim = 8;
  constexpr uint32_t group_size = 6;
  const std::string path = "ivf_turbo_cosine_nprobe.index";
  struct Cleanup {
    const std::string &path;
    ~Cleanup() {
      ailego::File::RemovePath(path);
    }
  } cleanup{path};
  ailego::File::RemovePath(path);
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, dim);
  meta.set_metric("Cosine", 0, ailego::Params());
  meta.set_quantizer("Int8Quantizer", 0, ailego::Params());
  auto quantizer = IndexFactory::CreateQuantizer("Int8Quantizer");
  ASSERT_NE(nullptr, quantizer);
  ASSERT_EQ(0, quantizer->init(meta, ailego::Params()));

  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
  for (uint32_t group = 0; group < 2; ++group) {
    for (uint32_t i = 0; i < group_size; ++i) {
      ailego::NumericalVector<float> vector(dim);
      std::fill(vector.data(), vector.data() + dim, 0.0f);
      const float scale = static_cast<float>(1U << i) + group * 0.25f;
      vector[0] = scale;
      // The final logical coordinate distinguishes the two angular groups.
      // Dropping it as an implicit norm tail makes their directions identical.
      vector[dim - 1] = (group == 0 ? 3.0f : -3.0f) * scale;
      holder->emplace(group * group_size + i, vector);
    }
  }
  ailego::Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2");
  params.set(PARAM_IVF_BUILDER_THREAD_COUNT, 1U);
  IVFBuilder builder;
  ASSERT_EQ(0, builder.init(meta, params, quantizer));
  // Known centers isolate centroid assignment and query transformation from
  // K-means random initialization on a corpus with only two distinct angles.
  auto converter = IndexFactory::CreateConverter("CosineFp32Converter");
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(meta, ailego::Params()));
  auto trainer_meta = converter->meta();
  trainer_meta.set_quantizer("", 0, ailego::Params());
  IndexCluster::CentroidList centroids;
  const float norm = std::sqrt(10.0f);
  for (uint32_t group = 0; group < 2; ++group) {
    ailego::NumericalVector<float> center(dim + 1);
    std::fill(center.data(), center.data() + dim + 1, 0.0f);
    center[0] = 1.0f / norm;
    center[dim - 1] = (group == 0 ? 3.0f : -3.0f) / norm;
    center[dim] = norm;
    IndexCluster::Centroid centroid;
    centroid.set_feature(center);
    centroids.emplace_back(std::move(centroid));
  }
  IndexBundle::Pointer bundle;
  ASSERT_EQ(0, IndexCluster::Serialize(trainer_meta, centroids, &bundle));
  ASSERT_EQ(0, builder.train(std::make_shared<FixedAngularTrainer>(
                   trainer_meta, std::move(bundle))));
  ASSERT_NE(nullptr, builder.centroid_index());
  ASSERT_EQ(2U, builder.centroid_index()->centroids_count());
  EXPECT_EQ("CosineFp32Converter",
            builder.centroid_index()->meta().converter_name());
  EXPECT_EQ(dim + 1, builder.centroid_index()->meta().dimension());
  ASSERT_EQ(0, builder.build(IndexThreads::Pointer(), holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(nullptr, dumper);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder.dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(path, false));
  auto segment = storage->get(IVF_CENTROID_SEG_ID);
  ASSERT_NE(nullptr, segment);
  IndexSegmentStorage centroid_storage(segment);
  ASSERT_EQ(0, centroid_storage.open("", false));
  IndexMeta centroid_meta;
  ASSERT_EQ(0, IndexHelper::DeserializeFromStorage(&centroid_storage,
                                                   &centroid_meta));
  EXPECT_EQ("CosineFp32Converter", centroid_meta.converter_name());
  EXPECT_EQ(dim + 1, centroid_meta.dimension());

  ailego::Params search_params;
  search_params.set(PARAM_IVF_SEARCHER_NPROBE, 1U);
  search_params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  search_params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 0U);
  IVFSearcher searcher;
  ASSERT_EQ(0, searcher.init(search_params));
  ASSERT_EQ(0, searcher.load(storage, IndexMetric::Pointer()));
  auto context = searcher.create_context();
  ASSERT_NE(nullptr, context);
  auto *ivf_context = dynamic_cast<IVFSearcherContext *>(context.get());
  ASSERT_NE(nullptr, ivf_context);
  EXPECT_EQ(0U, ivf_context->bruteforce_threshold());
  // BF would return all twelve records; nprobe=1 must return only six.
  context->set_topk(2 * group_size);
  std::vector<float> queries(2 * dim, 0.0f);
  queries[0] = 0.125f;
  queries[dim - 1] = 0.375f;
  queries[dim] = 40.0f;
  queries[2 * dim - 1] = -120.0f;
  ASSERT_EQ(0, searcher.search_impl(queries.data(),
                                    IndexQueryMeta(IndexMeta::DT_FP32, dim), 2,
                                    context));
  for (uint32_t group = 0; group < 2; ++group) {
    EXPECT_EQ(1U, ivf_context->centroid_searcher_ctx()->result(group).size());
    const auto &result = context->result(group);
    ASSERT_EQ(group_size, result.size());
    for (const auto &doc : result) {
      EXPECT_EQ(group, doc.key() / group_size);
      EXPECT_NEAR(0.0f, doc.score(), 0.001f);
    }
  }
  ASSERT_EQ(0, searcher.unload());
}

TEST(IVFTurboConfiguration, RejectsColumnOrderAndLegacyPostingQuantization) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  meta.set_quantizer("Int8Quantizer", 0, ailego::Params());
  auto quantizer = IndexFactory::CreateQuantizer("Int8Quantizer");
  ASSERT_NE(nullptr, quantizer);
  ASSERT_EQ(0, quantizer->init(meta, ailego::Params()));
  ailego::Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");

  {
    IVFBuilder builder;
    auto column_meta = meta;
    column_meta.set_major_order(IndexMeta::MO_COLUMN);
    EXPECT_NE(0, builder.init(column_meta, params, quantizer));
  }
  {
    IVFBuilder builder;
    auto conflicting_params = params;
    conflicting_params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS,
                           "Int8QuantizerConverter");
    EXPECT_NE(0, builder.init(meta, conflicting_params, quantizer));
  }
  {
    IVFBuilder builder;
    auto conflicting_params = params;
    conflicting_params.set(PARAM_IVF_BUILDER_QUANTIZE_BY_CENTROID, true);
    EXPECT_NE(0, builder.init(meta, conflicting_params, quantizer));
  }
}

}  // namespace
}  // namespace core
}  // namespace zvec
