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
#include <cmath>
#include <cstdio>
#include <random>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_framework.h>
#include "diskann_params.h"
#include "diskann_rabitq_quantizer.h"
#include "diskann_util.h"

namespace zvec::core {
#if RABITQ_SUPPORTED
TEST(DiskAnnRabitqTest, CodeRoundTripAndValidation) {
  for (uint32_t dim : {1u, 10u, 65u, 768u}) {
    IndexMeta meta(IndexMeta::DT_FP32, dim);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    for (uint32_t bits = 1; bits <= 9; ++bits) {
      auto quantizer = CreateDiskAnnRabitqQuantizer();
      ailego::Params params;
      params.set("total_bits", bits);
      ASSERT_EQ(0, quantizer->init(meta, params));
      std::vector<float> row(dim), query(dim);
      for (uint32_t i = 0; i < dim; ++i) {
        row[i] = std::sin(i + 0.5f);
        query[i] = std::cos(i + 0.2f);
      }
      std::string code(quantizer->quantized_datapoint_vector_length(), '\0');
      std::string q(quantizer->quantized_query_vector_length(), '\0');
      quantizer->quantize_data(row.data(), code.data());
      quantizer->quantize_query(query.data(), q.data());
      float before = quantizer->calc_distance_dp_query(code.data(), q.data());
      ASSERT_TRUE(std::isfinite(before));
      std::string blob;
      ASSERT_EQ(0, quantizer->serialize(&blob));
      auto loaded = DiskAnnUtil::create_quantizer_from_meta_buffer(
          blob, meta, code.size(), false);
      ASSERT_NE(nullptr, loaded);
      loaded->quantize_query(query.data(), q.data());
      EXPECT_FLOAT_EQ(before,
                      loaded->calc_distance_dp_query(code.data(), q.data()));
      EXPECT_EQ(nullptr, DiskAnnUtil::create_quantizer_from_meta_buffer(
                             blob, meta, code.size() + 1, false));
      auto truncated = blob.substr(0, blob.size() - 1);
      EXPECT_NE(0, loaded->deserialize(truncated));
      std::fill(row.begin(), row.end(), 0);
      quantizer->quantize_data(row.data(), code.data());
      EXPECT_TRUE(std::isfinite(
          quantizer->calc_distance_dp_query(code.data(), q.data())));
      quantizer->quantize_query(row.data(), q.data());
      EXPECT_NEAR(0, quantizer->calc_distance_dp_query(code.data(), q.data()),
                  1e-6);
    }
  }
}

TEST(DiskAnnRabitqTest, BuildReloadSearchAndStreamer) {
  constexpr uint32_t dim = 65, count = 128;
  for (uint32_t bits : {1u, 4u, 9u}) {
    IndexMeta meta(IndexMeta::DT_FP32, dim);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    auto holder =
        std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
    std::mt19937 rng(42);
    std::normal_distribution<float> normal;
    std::vector<std::vector<float>> rows(count, std::vector<float>(dim));
    for (uint32_t i = 0; i < count; ++i) {
      ailego::NumericalVector<float> row(dim);
      for (uint32_t j = 0; j < dim; ++j) row[j] = rows[i][j] = normal(rng);
      ASSERT_TRUE(holder->emplace(i, row));
    }
    ailego::Params params;
    params.set(PARAM_DISKANN_BUILDER_QUANTIZER, "rabitq");
    params.set(PARAM_DISKANN_BUILDER_RABITQ_BITS, bits);
    params.set(PARAM_DISKANN_BUILDER_MAX_DEGREE, 16);
    params.set(PARAM_DISKANN_BUILDER_LIST_SIZE, 64);
    params.set(PARAM_DISKANN_BUILDER_THREAD_COUNT, 2);
    auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
    ASSERT_NE(nullptr, builder);
    ASSERT_EQ(0, builder->init(meta, params));
    ASSERT_EQ(0, builder->train(holder));
    ASSERT_EQ(0, builder->build(holder));
    const std::string path = "rabitq-test-" + std::to_string(bits) + ".index";
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(0, builder->dump(dumper));
    ASSERT_EQ(0, dumper->close());
    ASSERT_EQ(0, builder->cleanup());
    for (uint32_t cache : {0u, 32u}) {
      auto storage = IndexFactory::CreateStorage("FileReadStorage");
      ASSERT_EQ(0, storage->open(path, false));
      ailego::Params search_params;
      search_params.set(PARAM_DISKANN_SEARCHER_LIST_SIZE, count);
      search_params.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, cache);
      auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
      ASSERT_EQ(0, searcher->init(search_params));
      ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
      auto context = searcher->create_context();
      ASSERT_NE(nullptr, context);
      context->set_topk(1);
      IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
      for (uint32_t id : {0u, 42u, 127u}) {
        ASSERT_EQ(0, searcher->search_impl(rows[id].data(), qmeta, context));
        ASSERT_EQ(1u, context->result().size());
        EXPECT_EQ(id, context->result()[0].key());
        EXPECT_NEAR(0, context->result()[0].score(), 1e-5);
      }
      auto streamer = IndexFactory::CreateStreamer("DiskAnnStreamer");
      ASSERT_EQ(0, streamer->init(meta, search_params));
      auto stream_storage = IndexFactory::CreateStorage("FileReadStorage");
      ASSERT_EQ(0, stream_storage->open(path, false));
      ASSERT_EQ(0, streamer->open(stream_storage));
      auto stream_ctx = streamer->create_context();
      ASSERT_NE(nullptr, stream_ctx);
      stream_ctx->set_topk(1);
      ASSERT_EQ(0, streamer->search_impl(rows[42].data(), qmeta, stream_ctx));
      ASSERT_EQ(1u, stream_ctx->result().size());
      EXPECT_EQ(42u, stream_ctx->result()[0].key());
      ASSERT_EQ(0, streamer->close());
      ASSERT_EQ(0, searcher->unload());
      ASSERT_EQ(0, storage->close());
    }
    std::remove(path.c_str());
  }
}
#endif

TEST(DiskAnnRabitqTest, InvalidOptions) {
  IndexMeta meta(IndexMeta::DT_FP32, 64);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ailego::Params params;
  params.set(PARAM_DISKANN_BUILDER_QUANTIZER, "invalid");
  EXPECT_NE(0, builder->init(meta, params));
  for (uint32_t bits : {0u, 10u}) {
    params.set(PARAM_DISKANN_BUILDER_QUANTIZER, "rabitq");
    params.set(PARAM_DISKANN_BUILDER_RABITQ_BITS, bits);
    EXPECT_NE(0, builder->init(meta, params));
  }
}
}  // namespace zvec::core
