// Copyright 2026-present the zvec project
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
#include "algorithm/quantized_graph/quantized_graph_builder.h"
#include <gtest/gtest.h>

#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
#include "algorithm/quantized_graph/quantized_graph_index.h"
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_flow.h>
#include "algorithm/quantized_graph/quantized_graph_query.h"

namespace zvec::core {
namespace {

std::vector<float> vectors(size_t count, size_t dimension) {
  std::mt19937 random(714);
  std::normal_distribution<float> normal;
  std::vector<float> data(count * dimension);
  for (float &v : data) v = normal(random);
  return data;
}

QuantizedGraphBuilder::Options options() {
  QuantizedGraphBuilder::Options out;
  out.search_list_size = 64;
  out.passes = 2;
  out.encoding.training_samples = 64;
  out.router_anchors = 32;
  out.router_dimension = 16;
  out.page_params.set(quantized_graph_pages::kPolicy, std::string("hint"));
  return out;
}

void check_graph(const QuantizedGraphData &data, size_t degree) {
  ASSERT_EQ(data.offsets.size(), data.size() + 1);
  ASSERT_EQ(data.offsets.front(), 0u);
  ASSERT_EQ(data.offsets.back(), data.edges.size());
  for (uint32_t id = 0; id < data.size(); ++id) {
    const auto row = data.neighbors(id);
    std::set<uint32_t> seen;
    for (size_t i = 0; i < row.size(); ++i) {
      EXPECT_TRUE(data.valid(row[i]));
      EXPECT_NE(row[i], id);
      EXPECT_TRUE(seen.insert(row[i]).second);
    }
    EXPECT_LE(row.size(), degree);
    if (data.size() > 1) {
      EXPECT_EQ(seen.count((id + 1) % data.size()), 1u);
    }
  }
}

void check_queries(const QuantizedGraphBuilder::Result &built) {
  const auto &data = *built.data;
  auto distance = [&data](const void *a, const void *b, size_t dimension,
                          float *out) {
    const auto *x = static_cast<const float *>(a);
    const auto *y = static_cast<const float *>(b);
    float value = 0;
    for (size_t d = 0; d < dimension; ++d) {
      const float delta = x[d] - y[d];
      value += data.cosine ? x[d] * y[d] : delta * delta;
    }
    *out = data.cosine ? 1.f - value : value;
  };
  QuantizedGraphQuery context(data.dimension, distance);
  QuantizedGraphQuery::Options search;
  search.margin = 1e6f;
  search.max_expansions = data.size() * 2;
  const size_t k = std::min(size_t{10}, data.size());
  for (size_t q = 0; q < std::min(size_t{8}, data.size()); ++q) {
    const float *query = data.vectors.data() + q * data.dimension;
    std::vector<float> expected;
    for (size_t i = 0; i < data.size(); ++i) {
      float value;
      distance(data.vectors.data() + i * data.dimension, query, data.dimension,
               &value);
      expected.push_back(value);
    }
    std::sort(expected.begin(), expected.end());
    for (unsigned beam : {0u, 1u}) {
      search.route_beam = beam;
      ASSERT_EQ(0, context.reset(query, k, data.size(), search));
      ASSERT_EQ(0, built.index->search(data, context));
      ASSERT_GE(context.results().size(), k);
      std::set<uint32_t> ids;
      for (size_t i = 0; i < k; ++i) {
        const auto &hit = context.results()[i];
        EXPECT_TRUE(data.valid(hit.first));
        EXPECT_TRUE(ids.insert(hit.first).second);
        EXPECT_NEAR(hit.second, expected[i], 2e-4f);
      }
    }
  }
}

TEST(QuantizedGraphFramework, ValidateBuildOptions) {
  if (!QuantizedGraphSupported()) GTEST_SKIP();
  QuantizedGraphBuilder builder;
  IndexMeta meta(IndexMeta::DT_FP32, 32);
  auto init = [&](const char *key, auto value) {
    ailego::Params params;
    params.set(std::string("zvec.quantized_graph.builder.") + key, value);
    return builder.init(meta, params);
  };
  EXPECT_EQ(0, builder.init(meta, {}));
  EXPECT_EQ(0, init("seed", UINT32_MAX));
  EXPECT_EQ(0, init("encoding.center_weight", 0.5));
  EXPECT_EQ(0, init("encoding.center_weight", 1));
  EXPECT_EQ(0, init("encoding.codebook", std::string("gaussian")));
  EXPECT_EQ(IndexError_InvalidArgument, init("threads", std::string("2")));
  EXPECT_EQ(IndexError_InvalidArgument, init("threads", 2.0));
  EXPECT_EQ(IndexError_InvalidArgument, init("threads", 0));
  EXPECT_EQ(IndexError_InvalidArgument, init("seed", -1));
  EXPECT_EQ(IndexError_InvalidArgument, init("seed", uint64_t{1} << 32));
  EXPECT_EQ(IndexError_InvalidArgument, init("encoding.center_weight", true));
  EXPECT_EQ(IndexError_InvalidArgument, init("encoding.rotation_view", 1));
  EXPECT_EQ(IndexError_InvalidArgument, init("encoding.codec", std::string("unknown")));
  ailego::Params old;
  old.set("options", std::string("{}"));
  EXPECT_EQ(IndexError_InvalidArgument, builder.init(meta, old));
}

TEST(QuantizedGraphFramework, BuildDumpAndQueryThroughBothReaders) {
  if (!QuantizedGraphSupported()) GTEST_SKIP();
  for (bool cosine : {false, true}) {
    const size_t n = 48, dim = 16;
    const uint64_t base = uint64_t{1} << 40;
    IndexMeta meta(IndexMeta::DT_FP32, dim);
    if (cosine) meta.set_metric("Cosine", 0, ailego::Params{});
    auto x = vectors(n, dim);
    auto holder = std::make_shared<RandomAccessIndexHolder>(meta);
    for (size_t i = n; i--;) holder->emplace(base + i, x.data() + i * dim);
    auto b = std::dynamic_pointer_cast<QuantizedGraphBuilder>(
        IndexFactory::CreateBuilder("QuantizedGraphBuilder"));
    ASSERT_TRUE(b);
    ailego::Params params;
    params.set("zvec.quantized_graph.builder.threads", 1);
    params.set("zvec.quantized_graph.builder.router_anchors", 8);
    params.set("zvec.quantized_graph.builder.router_dimension", 8);
    const double weight = 0.12345678901234566;
    params.set("zvec.quantized_graph.builder.encoding.center_weight", weight);
    ASSERT_EQ(0, b->init(meta, params));
    ASSERT_EQ(0, b->train(holder));
    ASSERT_EQ(0, b->build(holder));
    EXPECT_EQ(n, b->stats().built_count());
    char path[] = "/tmp/zvec-qg-framework-XXXXXX";
    int fd = mkstemp(path);
    ASSERT_GE(fd, 0);
    close(fd);
    struct Remove {
      const char *path;
      ~Remove() {
        std::remove(path);
      }
    } cleanup{path};
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_TRUE(dumper);
    ASSERT_EQ(0, dumper->init({}));
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(0, b->dump(dumper));
    ASSERT_EQ(0, dumper->close());
    QuantizedGraphBuilder::Result restored;
    std::vector<uint64_t> keys;
    EXPECT_EQ(IndexError_InvalidFormat, restored.open(path));
    ASSERT_EQ(0, restored.open(path, {}, &keys));
    ASSERT_EQ(n, keys.size());
    for (size_t i = 0; i < n; ++i) EXPECT_EQ(base + i, keys[i]);
    auto permuted = keys;
    std::swap(permuted.front(), permuted.back());
    ASSERT_EQ(0, restored.save(path, permuted));
    QuantizedGraphBuilder::Result reread;
    std::vector<uint64_t> mapped;
    ASSERT_EQ(0, reread.open(path, {}, &mapped));
    EXPECT_EQ(permuted, mapped);
    permuted.front() = permuted.back();
    EXPECT_EQ(IndexError_InvalidArgument, restored.save(path, permuted));
    ASSERT_EQ(0, restored.save(path, keys));
    IndexFlow flow;
    ASSERT_EQ(0, flow.load(path));
    ASSERT_EQ("QuantizedGraphSearcher", flow.searcher()->name());
    EXPECT_EQ("QuantizedGraphBuilder", flow.meta().builder_name());
    const auto &saved = flow.meta().builder_params();
    EXPECT_FALSE(saved.has("options"));
    EXPECT_EQ(1u, saved.get_as_uint32("zvec.quantized_graph.builder.threads"));
    EXPECT_EQ(weight, saved.get_as_double(
                          "zvec.quantized_graph.builder.encoding.center_weight"));
    auto ctx = flow.create_context();
    ctx->set_topk(3);
    ASSERT_EQ(0, flow.search<IndexMeta::DT_FP32>(x.data(), dim, 2, ctx));
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    ASSERT_TRUE(storage);
    ASSERT_EQ(0, storage->init({}));
    ASSERT_EQ(0, storage->open(path, false));
    auto stream = IndexFactory::CreateStreamer("QuantizedGraphStreamer");
    ASSERT_TRUE(stream);
    ASSERT_EQ(0, stream->init(meta, {}));
    ASSERT_EQ(0, stream->open(storage, true));
    auto sc = stream->create_context();
    ASSERT_TRUE(sc);
    sc->set_topk(3);
    IndexQueryMeta qm(IndexMeta::DT_FP32, dim);
    ASSERT_EQ(0, stream->search_impl(x.data(), qm, 2, sc));
    for (size_t i = 0; i < 2; ++i) {
      ASSERT_EQ(3u, sc->result(i).size());
      ASSERT_EQ(3u, ctx->result(i).size());
      EXPECT_EQ(base + i, ctx->result(i)[0].key());
      EXPECT_NEAR(0.f, ctx->result(i)[0].score(), 1e-5f);
      for (size_t j = 0; j < 3; ++j) {
        EXPECT_EQ(ctx->result(i)[j].key(), sc->result(i)[j].key());
        EXPECT_FLOAT_EQ(ctx->result(i)[j].score(), sc->result(i)[j].score());
      }
    }
    sc->set_filter([base](uint64_t key) { return key == base; });
    sc->set_threshold(1e-5f);
    ASSERT_EQ(0, stream->search_impl(x.data(), qm, sc));
    EXPECT_TRUE(sc->result().empty());
    ASSERT_EQ(0, stream->close());
    EXPECT_NE(0, stream->search_impl(x.data(), qm, sc));
  }
}
class RawQuantizedGraph : public testing::TestWithParam<unsigned> {};

TEST_P(RawQuantizedGraph, BuildsAndSearchesWithoutAnotherIndex) {
  if (!QuantizedGraphSupported()) GTEST_SKIP();
  const auto variant = GetParam();
  auto config = options();
  if (variant & 2) {
    config.encoding.codec = QuantizedGraphEncodingOptions::Codec::LowRank;
    config.encoding.block_dimension = 8;
    config.encoding.codebook = QuantizedGraphEncodingOptions::Codebook::Gaussian;
    config.encoding.rotation_views = 2;
  }
  if (variant & 4) config.router_anchors = 0;
  QuantizedGraphBuilder::Result result;
  ASSERT_EQ(0, QuantizedGraphBuilder::build(vectors(96, 32), 32, variant & 1,
                                            config, result));
  ASSERT_TRUE(result.index);
  ASSERT_TRUE(result.data);
  check_graph(*result.data, config.encoding.max_neighbors);
  check_queries(result);
}

INSTANTIATE_TEST_SUITE_P(MetricsCodecsAndRouting, RawQuantizedGraph,
                         testing::Range(0u, 8u));

TEST(QuantizedGraphBuilder, DuplicateVectorsAndSmallCorpora) {
  if (!QuantizedGraphSupported()) GTEST_SKIP();
  for (size_t count : {1u, 2u, 7u, 65u}) {
    QuantizedGraphBuilder::Result result;
    ASSERT_EQ(0, QuantizedGraphBuilder::build(std::vector<float>(count * 32, 1),
                                              32, true, options(), result));
    check_graph(*result.data, 32);
    check_queries(result);
  }
}

TEST(QuantizedGraphBuilder, DeterministicSingleThreadConstruction) {
  if (!QuantizedGraphSupported()) GTEST_SKIP();
  auto config = options();
  config.threads = 1;
  QuantizedGraphBuilder::Result first, second;
  ASSERT_EQ(0, QuantizedGraphBuilder::build(vectors(128, 32), 32, false, config,
                                            first));
  ASSERT_EQ(0, QuantizedGraphBuilder::build(vectors(128, 32), 32, false, config,
                                            second));
  EXPECT_EQ(first.data->edges, second.data->edges);
  EXPECT_EQ(first.data->offsets, second.data->offsets);
  EXPECT_EQ(first.router, second.router);
  EXPECT_EQ(first.codec, second.codec);
}

TEST(QuantizedGraphBuilder, RejectsInvalidInputsWithoutReplacingResult) {
  if (!QuantizedGraphSupported()) GTEST_SKIP();
  QuantizedGraphBuilder::Result result;
  auto config = options();
  ASSERT_EQ(0, QuantizedGraphBuilder::build(vectors(16, 32), 32, false, config,
                                            result));
  const auto original = result.index;
  EXPECT_EQ(IndexError_InvalidArgument,
            QuantizedGraphBuilder::build({}, 32, false, config, result));
  EXPECT_EQ(
      IndexError_InvalidArgument,
      QuantizedGraphBuilder::build(vectors(3, 32), 0, false, config, result));
  EXPECT_EQ(
      IndexError_InvalidArgument,
      QuantizedGraphBuilder::build(vectors(3, 32), 31, false, config, result));
  EXPECT_EQ(IndexError_InvalidArgument,
            QuantizedGraphBuilder::build(std::vector<float>(64, 0), 32, true,
                                         config, result));
  auto invalid = vectors(3, 32);
  invalid[5] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(IndexError_InvalidArgument,
            QuantizedGraphBuilder::build(invalid, 32, false, config, result));
  invalid[5] = std::numeric_limits<float>::max();
  EXPECT_EQ(IndexError_InvalidArgument,
            QuantizedGraphBuilder::build(invalid, 32, false, config, result));
  config.threads = 0;
  EXPECT_EQ(
      IndexError_InvalidArgument,
      QuantizedGraphBuilder::build(vectors(3, 32), 32, false, config, result));
  EXPECT_EQ(result.index, original);
}

}  // namespace
}  // namespace zvec::core
#else
namespace zvec::core {
TEST(QuantizedGraphBuilder, ReportsUnsupportedBackend) {
  QuantizedGraphBuilder::Result result;
  QuantizedGraphBuilder::Options config;
  EXPECT_EQ(IndexError_Unsupported,
            QuantizedGraphBuilder::build({}, 32, false, config, result));
}
}  // namespace zvec::core
#endif
