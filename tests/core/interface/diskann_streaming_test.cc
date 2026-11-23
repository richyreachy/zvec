// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <random>
#include <thread>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/vamana/vamana_params.h"
#include "tests/test_util.h"

#if DISKANN_SUPPORTED && RABITQ_SUPPORTED
namespace zvec::core_interface {
namespace {
constexpr uint32_t kDim = 128;

auto parameter(bool fp16, MetricType metric) {
  return DiskAnnIndexParamBuilder()
      .with_dimension(kDim)
      .with_metric_type(metric)
      .with_data_type(DataType::DT_FP32)
      .with_quantizer_param(
          QuantizerParam(fp16 ? QuantizerType::kFP16 : QuantizerType::kNone))
      .with_max_degree(16)
      .with_list_size(40)
      .with_streaming(true)
      .build();
}

auto vector_for(uint32_t id) {
  std::mt19937 rng(id);
  std::uniform_real_distribution<float> dist(-1, 1);
  std::vector<float> data(kDim);
  for (auto &v : data) v = dist(rng);
  return data;
}

int add(Index &index, uint32_t id) {
  auto data = vector_for(id);
  return index.add(VectorData{DenseVector{data.data()}}, id);
}

void expect_nearest(Index &index, uint32_t key) {
  auto data = vector_for(key);
  auto query = std::make_shared<DiskAnnQueryParam>();
  query->list_size = 64;
  query->topk = 3;
  SearchResult result;
  ASSERT_EQ(0,
            index.search(VectorData{DenseVector{data.data()}}, query, &result));
  ASSERT_FALSE(result.doc_list_.empty());
  EXPECT_EQ(key, result.doc_list_.front().key());
  VectorDataBuffer buffer;
  ASSERT_EQ(0, index.fetch(key, &buffer));
  const auto &bytes = std::get<DenseVectorBuffer>(buffer.vector_buffer).data;
  ASSERT_EQ(kDim * sizeof(float), bytes.size());
  for (size_t i = 0; i < kDim; ++i) {
    float actual;
    std::memcpy(&actual, bytes.data() + i * sizeof(float), sizeof(actual));
    EXPECT_NEAR(data[i], actual, 0.001);
  }
}

TEST(DiskAnnStreaming, ExplicitIdsAndGeneratedIdsSurviveReopen) {
  const std::string path = "diskann_stream_id_lookup";
  test_util::RemoveTestFiles(path);
  auto cleanup =
      ailego::ScopeGuard::Make([&] { test_util::RemoveTestFiles(path); });
  core::IndexMeta meta(core::IndexMeta::DT_FP32, kDim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  core::IndexQueryMeta qm(core::IndexMeta::DT_FP32, kDim);
  constexpr uint64_t kKey = (1ULL << 40);
  constexpr uint32_t kSparseId = 4000000000U;
  constexpr uint64_t kMax32BitKey = 0xffffffffULL;
  auto first = vector_for(1), second = vector_for(2), third = vector_for(3);
  size_t allocated = 0;
  for (bool create : {true, false}) {
    auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_EQ(0, storage->init({}));
    ASSERT_EQ(0, storage->open(path, create));
    auto index = core::IndexFactory::CreateStreamer("DiskAnnMutableStreamer");
    ailego::Params params;
    params.set(core::PARAM_VAMANA_STREAMER_GET_VECTOR_ENABLE, true);
    params.set(core::PARAM_VAMANA_STREAMER_DOCS_HARD_LIMIT, size_t{6});
    ASSERT_EQ(0, index->init(meta, params));
    ASSERT_EQ(0, index->open(storage));
    auto ctx = index->create_context();
    ctx->set_topk(1);
    if (create) {
      auto fourth = vector_for(4), fifth = vector_for(5);
      ASSERT_EQ(0, index->add_with_id_impl(2, first.data(), qm, ctx));
      ASSERT_EQ(nullptr, index->get_vector_by_id(0));
      ASSERT_EQ(0, index->add_with_id_impl(0, second.data(), qm, ctx));
      // Physical ordinal 2 is already reserved as a public ID by the first row.
      ASSERT_EQ(0, index->add_impl(kKey, third.data(), qm, ctx));
      ASSERT_EQ(0, index->add_with_id_impl(kSparseId, fourth.data(), qm, ctx));
      ASSERT_EQ(0,
                index->add_with_id_impl(kSparseId - 1, fifth.data(), qm, ctx));
      auto sixth = vector_for(6);
      ASSERT_EQ(0, index->add_impl(kMax32BitKey, sixth.data(), qm, ctx));
      allocated = index->stats().index_size();
      ASSERT_EQ(core::IndexError_IndexFull,
                index->add_impl(kKey + 1, first.data(), qm, ctx));
    }
    EXPECT_EQ(allocated, index->stats().index_size());
    {
      auto provider = index->create_provider();
      ASSERT_NE(nullptr, provider);
      auto iter = provider->create_iterator();
      ASSERT_NE(nullptr, iter);
      for (uint64_t key : std::vector<uint64_t>{0, kKey, 2, kMax32BitKey,
                                                kSparseId - 1, kSparseId}) {
        ASSERT_TRUE(iter->is_valid());
        EXPECT_EQ(key, iter->key());
        iter->next();
      }
      EXPECT_FALSE(iter->is_valid());
    }
    core::IndexStorage::MemoryBlock block;
    ASSERT_EQ(0, index->get_vector_by_id(0, block));
    EXPECT_EQ(0,
              std::memcmp(second.data(), block.data(), sizeof(float) * kDim));
    const auto *vec = index->get_vector_by_id(2);
    ASSERT_NE(nullptr, vec);
    EXPECT_EQ(0, std::memcmp(first.data(), vec, sizeof(float) * kDim));
    ASSERT_EQ(0, index->search_impl(third.data(), qm, 1, ctx));
    ASSERT_EQ(1U, ctx->result().size());
    EXPECT_EQ(kKey, ctx->result().front().key());
    EXPECT_EQ(1U, ctx->result().front().index());
    ASSERT_EQ(0, index->search_bf_impl(first.data(), qm, 1, ctx));
    ASSERT_EQ(1U, ctx->result().size());
    EXPECT_EQ(2U, ctx->result().front().index());
    ASSERT_EQ(
        0, index->search_bf_by_p_keys_impl(second.data(), {{0}}, qm, 1, ctx));
    ASSERT_EQ(1U, ctx->result().size());
    EXPECT_EQ(0U, ctx->result().front().index());
    ASSERT_EQ(0, index->flush(0));
    ctx.reset();
    ASSERT_EQ(0, index->close());
    ASSERT_EQ(0, storage->close());
  }
}

TEST(DiskAnnStreaming, CodePagesShareGraphQuota) {
  const std::string path = "diskann_stream_size_quota";
  test_util::RemoveTestFiles(path);
  auto cleanup =
      ailego::ScopeGuard::Make([&] { test_util::RemoveTestFiles(path); });
  core::IndexMeta meta(core::IndexMeta::DT_FP32, kDim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  core::IndexQueryMeta qm(core::IndexMeta::DT_FP32, kDim);
  auto vec = vector_for(0);
  for (size_t quota : {128UL << 10, 512UL << 10}) {
    SCOPED_TRACE(quota);
    size_t published = 0, allocated = 0;
    for (bool create : {true, false}) {
      auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
      ASSERT_EQ(0, storage->init({}));
      ASSERT_EQ(0, storage->open(path, create));
      auto index = core::IndexFactory::CreateStreamer("DiskAnnMutableStreamer");
      ailego::Params params;
      params.set(core::PARAM_VAMANA_STREAMER_CHUNK_SIZE, size_t{4096});
      params.set(core::PARAM_VAMANA_STREAMER_MAX_INDEX_SIZE, quota);
      ASSERT_EQ(0, index->init(meta, params));
      ASSERT_EQ(0, index->open(storage));
      auto ctx = index->create_context();
      ctx->set_topk(1);
      if (create) {
        int ret = 0;
        while (ret == 0 && published < 4096) {
          ret = index->add_with_id_impl(published, vec.data(), qm, ctx);
          if (ret == 0) ++published;
        }
        ASSERT_EQ(core::IndexError_IndexFull, ret);
        allocated = index->stats().index_size();
        ASSERT_LT(allocated, quota);
        auto page = storage->get("diskann.stream.v1.codes.0");
        if (quota == (128UL << 10)) {
          EXPECT_EQ(0U, published);
          EXPECT_FALSE(page);
        } else {
          ASSERT_GT(published, 0U);
          ASSERT_TRUE(page);
          EXPECT_GT(allocated, page->capacity());
        }
      }
      EXPECT_EQ(allocated, index->stats().index_size());
      EXPECT_EQ(core::IndexError_IndexFull,
                index->add_with_id_impl(published, vec.data(), qm, ctx));
      ASSERT_EQ(0, index->search_impl(vec.data(), qm, 1, ctx));
      EXPECT_EQ(published ? 1U : 0U, ctx->result().size());
      EXPECT_EQ(0, index->flush(0));
      ctx.reset();
      ASSERT_EQ(0, index->close());
      ASSERT_EQ(0, storage->close());
    }
    test_util::RemoveTestFiles(path);
  }
}

TEST(DiskAnnStreaming, InsertSearchReopenAndAppend) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const auto previous = pool.capacity();
  ASSERT_EQ(0, pool.init(64UL << 20));
  auto restore =
      ailego::ScopeGuard::Make([&] { EXPECT_EQ(0, pool.init(previous)); });
  for (auto storage : {StorageOptions::StorageType::kMMAP,
                       StorageOptions::StorageType::kBufferPool}) {
    for (bool fp16 : {false, true}) {
      for (auto metric : {MetricType::kL2sq, MetricType::kCosine}) {
        SCOPED_TRACE(::testing::Message()
                     << int(storage) << " " << fp16 << " " << int(metric));
        const std::string path = "diskann_stream_insert_test";
        test_util::RemoveTestFiles(path);
        auto options = parameter(fp16, metric);
        auto index = IndexFactory::CreateAndInitIndex(*options);
        ASSERT_NE(nullptr, index);
        ASSERT_EQ(0, index->open(path, {storage, true}));
        ASSERT_EQ(0, add(*index, 907));
        expect_nearest(*index, 907);
        for (uint32_t i = 0; i < 100; ++i) ASSERT_EQ(0, add(*index, 3 * i));
        expect_nearest(*index, 297);
        EXPECT_NE(0, add(*index, 297));
        EXPECT_EQ(101U, index->get_doc_count());
        ASSERT_EQ(0, index->flush());
        ASSERT_EQ(0, index->close());
        index = IndexFactory::CreateAndInitIndex(*options);
        ASSERT_EQ(0, index->open(path, {storage, false}));
        expect_nearest(*index, 907);
        ASSERT_EQ(0, add(*index, 999));
        expect_nearest(*index, 999);
        ASSERT_EQ(0, index->close());
        index = IndexFactory::CreateAndInitIndex(*options);
        ASSERT_EQ(0, index->open(path, {storage, false, true}));
        EXPECT_EQ(102U, index->get_doc_count());
        EXPECT_NE(0, add(*index, 1000));
        expect_nearest(*index, 999);
        ASSERT_EQ(0, index->close());
        test_util::RemoveTestFiles(path);
      }
    }
  }
}

TEST(DiskAnnStreaming, CrossesCodeAndGraphChunksWithHeldOutQueries) {
  constexpr uint32_t kRows = 4128;
  constexpr uint32_t kQueries = 32;
  constexpr uint32_t kTopK = 10;
  const std::string path = "diskann_stream_page_boundary";
  test_util::RemoveTestFiles(path);
  auto cleanup =
      ailego::ScopeGuard::Make([&] { test_util::RemoveTestFiles(path); });
  auto options = parameter(false, MetricType::kL2sq);
  auto index = IndexFactory::CreateAndInitIndex(*options);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_TRUE(std::dynamic_pointer_cast<DiskAnnIndexParam>(index->get_param())
                  ->streaming);
  std::vector<std::vector<float>> corpus;
  for (uint32_t i = 0; i < kRows; ++i) {
    corpus.push_back(vector_for(i));
    ASSERT_EQ(0, index->add(VectorData{DenseVector{corpus.back().data()}}, i));
  }
  auto linear = std::make_shared<DiskAnnQueryParam>();
  linear->is_linear = true;
  linear->topk = 1;
  SearchResult result;
  ASSERT_EQ(0, index->search(VectorData{DenseVector{corpus.back().data()}},
                             linear, &result));
  ASSERT_EQ(1U, result.doc_list_.size());
  EXPECT_EQ(kRows - 1, result.doc_list_[0].key());
  ASSERT_EQ(0, index->close());
  index = IndexFactory::CreateAndInitIndex(*options);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, false}));
  ASSERT_EQ(kRows, index->get_doc_count());
  ASSERT_EQ(0, add(*index, 10000));
  expect_nearest(*index, 10000);
  corpus.push_back(vector_for(10000));
  auto param = std::make_shared<DiskAnnQueryParam>();
  param->topk = kTopK;
  param->list_size = 256;
  size_t hits = 0;
  for (uint32_t q = 0; q < kQueries; ++q) {
    auto query = vector_for(20000 + q);
    std::vector<std::pair<float, uint32_t>> truth;
    for (uint32_t i = 0; i < corpus.size(); ++i) {
      float distance = 0;
      for (size_t d = 0; d < kDim; ++d) {
        float delta = query[d] - corpus[i][d];
        distance += delta * delta;
      }
      truth.emplace_back(distance, i == kRows ? 10000 : i);
    }
    std::partial_sort(truth.begin(), truth.begin() + kTopK, truth.end());
    int64_t ids[kTopK];
    ASSERT_EQ(0,
              index->search_internal_ids(VectorData{DenseVector{query.data()}},
                                         param, ids, nullptr));
    for (uint32_t k = 0; k < kTopK; ++k)
      hits += std::find(ids, ids + kTopK, truth[k].second) != ids + kTopK;
  }
  const double recall = double(hits) / (kTopK * kQueries);
  RecordProperty("held_out_recall_at_10", std::to_string(recall));
  EXPECT_GE(recall, 0.80);
  ASSERT_EQ(0, index->close());
}

TEST(DiskAnnStreaming, ConcurrentSearchAndInsert) {
  const std::string path = "diskann_stream_concurrent";
  test_util::RemoveTestFiles(path);
  auto cleanup =
      ailego::ScopeGuard::Make([&] { test_util::RemoveTestFiles(path); });
  auto index =
      IndexFactory::CreateAndInitIndex(*parameter(true, MetricType::kL2sq));
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(0, add(*index, 77));
  std::atomic<bool> start{false};
  std::thread writer([&] {
    while (!start.load()) std::this_thread::yield();
    for (uint32_t i = 100; i < 300; ++i) ASSERT_EQ(0, add(*index, i));
  });
  std::thread reader([&] {
    start.store(true);
    for (size_t i = 0; i < 64; ++i) expect_nearest(*index, 77);
  });
  writer.join();
  reader.join();
  EXPECT_EQ(201U, index->get_doc_count());
  expect_nearest(*index, 299);
  ASSERT_EQ(0, index->close());
}

TEST(DiskAnnStreaming, ZeroVectorAndRejectedInput) {
  const std::string path = "diskann_stream_zero";
  test_util::RemoveTestFiles(path);
  auto cleanup =
      ailego::ScopeGuard::Make([&] { test_util::RemoveTestFiles(path); });
  auto index =
      IndexFactory::CreateAndInitIndex(*parameter(false, MetricType::kL2sq));
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = 1;
  std::vector<float> zero(kDim, 0);
  VectorData input{DenseVector{zero.data()}};
  SearchResult result;
  ASSERT_EQ(0, index->search(input, query, &result));
  EXPECT_TRUE(result.doc_list_.empty());
  ASSERT_EQ(0, index->add(input, 1));
  ASSERT_EQ(0, add(*index, 2));
  ASSERT_EQ(0, index->search(input, query, &result));
  ASSERT_EQ(1U, result.doc_list_.size());
  EXPECT_EQ(1U, result.doc_list_[0].key());
  EXPECT_FLOAT_EQ(0, result.doc_list_[0].score());
  zero[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_NE(0, index->add(input, 3));
  EXPECT_EQ(2U, index->get_doc_count());
  expect_nearest(*index, 2);
  ASSERT_EQ(0, index->close());
}

TEST(DiskAnnStreaming, RejectsUnpublishedGraph) {
  const std::string path = "diskann_stream_incomplete";
  test_util::RemoveTestFiles(path);
  auto cleanup =
      ailego::ScopeGuard::Make([&] { test_util::RemoveTestFiles(path); });
  auto options = parameter(false, MetricType::kL2sq);
  auto index = IndexFactory::CreateAndInitIndex(*options);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(0, add(*index, 19));
  ASSERT_EQ(0, index->close());
  auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path, false));
  auto published = storage->get("diskann.stream.v1.published");
  uint32_t old_count = 0;
  ASSERT_NE(nullptr, published);
  ASSERT_EQ(sizeof(old_count),
            published->write(0, &old_count, sizeof(old_count)));
  ASSERT_EQ(0, storage->flush());
  published.reset();
  ASSERT_EQ(0, storage->close());
  index = IndexFactory::CreateAndInitIndex(*options);
  EXPECT_NE(
      0, index->open(path, {StorageOptions::StorageType::kMMAP, false, true}));
}

TEST(DiskAnnStreaming, FilteredMergeUsesDocumentOrderAfterReopen) {
  const std::string source_path = "diskann_merge_rows_source";
  const std::string target_path = "diskann_merge_rows_target";
  auto cleanup = ailego::ScopeGuard::Make([&] {
    test_util::RemoveTestFiles(source_path);
    test_util::RemoveTestFiles(target_path);
  });
  auto param = parameter(false, MetricType::kL2sq);
  auto source = IndexFactory::CreateAndInitIndex(*param);
  auto target = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, source);
  ASSERT_NE(nullptr, target);
  ASSERT_EQ(
      0, source->open(source_path, {StorageOptions::StorageType::kMMAP, true}));
  // A parallel writer can acquire the graph lock in any document order.
  for (auto id : {4U, 1U, 3U, 0U, 5U, 2U}) ASSERT_EQ(0, add(*source, id));
  ASSERT_EQ(0, source->close());
  source = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_EQ(0, source->open(source_path,
                            {StorageOptions::StorageType::kMMAP, false, true}));
  ASSERT_EQ(
      0, target->open(target_path, {StorageOptions::StorageType::kMMAP, true}));
  IndexFilter deleted;
  deleted.set([](uint64_t id) { return id == 1 || id == 5; });
  ASSERT_EQ(0, target->merge({source}, deleted, MergeOptions{4}));
  uint32_t row = 0;
  for (auto original : {0U, 2U, 3U, 4U}) {
    SCOPED_TRACE(original);
    VectorDataBuffer fetched;
    ASSERT_EQ(0, target->fetch(row++, &fetched));
    const auto &bytes = std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
    ASSERT_EQ(kDim * sizeof(float), bytes.size());
    EXPECT_EQ(0, std::memcmp(vector_for(original).data(), bytes.data(),
                             bytes.size()));
  }
  EXPECT_EQ(row, target->get_doc_count());
  EXPECT_EQ(0, target->close());
  EXPECT_EQ(0, source->close());
}

TEST(DiskAnnStreaming, MergeThenAppendAndReopenLegacyParameters) {
  DiskAnnIndexParam decoded;
  auto target = parameter(true, MetricType::kL2sq);
  ASSERT_TRUE(decoded.deserialize_from_json(target->serialize_to_json()));
  ASSERT_TRUE(decoded.streaming);
  ASSERT_TRUE(
      decoded.deserialize_from_json(DiskAnnIndexParam().serialize_to_json()));
  EXPECT_FALSE(decoded.streaming);
  const std::string source_path = "diskann_stream_merge_source",
                    path = "diskann_stream_merge_target";
  test_util::RemoveTestFiles(source_path);
  test_util::RemoveTestFiles(path);
  auto cleanup = ailego::ScopeGuard::Make([&] {
    test_util::RemoveTestFiles(source_path);
    test_util::RemoveTestFiles(path);
  });
  auto flat = FlatIndexParamBuilder()
                  .with_dimension(kDim)
                  .with_metric_type(MetricType::kL2sq)
                  .with_data_type(DataType::DT_FP32)
                  .build();
  auto source = IndexFactory::CreateAndInitIndex(*flat);
  auto index = IndexFactory::CreateAndInitIndex(*target);
  ASSERT_NE(nullptr, source);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(
      0, source->open(source_path, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  for (uint32_t i = 0; i < 100; ++i) ASSERT_EQ(0, add(*source, i));
  ASSERT_EQ(0, index->merge({source}, {}));
  expect_nearest(*index, 30);
  ASSERT_EQ(0, add(*index, 5432));
  expect_nearest(*index, 5432);
  ASSERT_EQ(0, source->close());
  ASSERT_EQ(0, index->close());
}

TEST(DiskAnnStreaming, AlternatesWithStaticDiskAnnOnSameThread) {
  auto mutable_param = parameter(false, MetricType::kL2sq);
  auto static_param = parameter(false, MetricType::kL2sq);
  static_param->streaming = false;
  static_param->pq_chunk_num = 8;
  const std::string mutable_path = "diskann_stream_mixed_graph",
                    static_path = "diskann_stream_mixed_static";
  test_util::RemoveTestFiles(mutable_path);
  test_util::RemoveTestFiles(static_path);
  auto cleanup = ailego::ScopeGuard::Make([&] {
    test_util::RemoveTestFiles(mutable_path);
    test_util::RemoveTestFiles(static_path);
  });
  auto online = IndexFactory::CreateAndInitIndex(*mutable_param);
  auto offline = IndexFactory::CreateAndInitIndex(*static_param);
  ASSERT_NE(nullptr, online);
  ASSERT_NE(nullptr, offline);
  ASSERT_EQ(0, online->open(mutable_path,
                            {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(0, offline->open(static_path,
                             {StorageOptions::StorageType::kMMAP, true}));
  for (uint32_t i = 0; i < 72; ++i) {
    ASSERT_EQ(0, add(*offline, i));
    ASSERT_EQ(0, add(*online, 1000 + i));
  }
  ASSERT_EQ(0, offline->train());
  expect_nearest(*offline, 13);
  expect_nearest(*online, 1013);
  expect_nearest(*offline, 4);
  ASSERT_EQ(0, add(*online, 1072));
  expect_nearest(*online, 1072);
  ASSERT_EQ(0, offline->close());
  ASSERT_EQ(0, online->close());
}
}  // namespace
}  // namespace zvec::core_interface
#endif
