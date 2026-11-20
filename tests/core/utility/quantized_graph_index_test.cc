// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include "algorithm/quantized_graph/quantized_graph_index.h"
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "algorithm/quantized_graph/quantized_graph_query.h"

namespace zvec::core {
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
namespace {

// A plain adjacency list and vector array: no HNSW classes, includes or links.
struct ArrayGraph {
  struct Vector {
    const void *value = nullptr;
    const void *data() const {
      return value;
    }
  };
  size_t dimension = 65, count = 24;
  bool fp16, cosine, deleted = true;
  std::vector<float> floats;
  std::vector<uint16_t> halves;
  std::vector<std::vector<uint32_t>> edges;

  ArrayGraph(bool half, bool angular) : fp16(half), cosine(angular) {
    std::mt19937 rng(718);
    std::uniform_real_distribution<float> uniform(-1.f, 1.f);
    floats.resize(count * dimension);
    edges.resize(count);
    for (size_t id = 0; id < count; ++id) {
      double norm = 0;
      for (size_t d = 0; d < dimension; ++d) {
        float &v = floats[id * dimension + d];
        v = uniform(rng);
        norm += double(v) * v;
      }
      if (cosine)
        for (size_t d = 0; d < dimension; ++d)
          floats[id * dimension + d] /= float(std::sqrt(norm));
      for (uint32_t neighbor = 0; neighbor < count; ++neighbor)
        if (neighbor != id) edges[id].push_back(neighbor);
    }
    halves.resize(floats.size());
    ailego::FloatHelper::ToFP16(floats.data(), floats.size(), halves.data());
  }

  bool valid(uint32_t id) const {
    return id < count && (!deleted || id != 3);
  }
  const std::vector<uint32_t> &neighbors(uint32_t id) const {
    return edges[id];
  }

  int get_vector(uint32_t id, Vector &out) const {
    if (!valid(id)) return IndexError_NoExist;
    out.value = fp16
                    ? static_cast<const void *>(halves.data() + id * dimension)
                    : static_cast<const void *>(floats.data() + id * dimension);
    return 0;
  }

  void distance(const void *data, const void *query, size_t n,
                float *out) const {
    std::array<float, 4096> decoded;
    const auto *v = static_cast<const float *>(data);
    if (fp16) {
      ailego::FloatHelper::ToFP32(static_cast<const uint16_t *>(data), n,
                                  decoded.data());
      v = decoded.data();
    }
    const auto *q = static_cast<const float *>(query);
    float value = 0;
    for (size_t d = 0; d < n; ++d) {
      const float delta = v[d] - q[d];
      value += cosine ? v[d] * q[d] : delta * delta;
    }
    *out = cosine ? 1.f - value : value;
  }
};

struct File {
  std::string name;
  File() {
    char pattern[] = "/tmp/quantized-graph-test-XXXXXX";
    int fd = mkstemp(pattern);
    if (fd < 0) throw std::runtime_error("cannot create test file");
    ::close(fd);
    name = pattern;
  }
  ~File() {
    std::remove(name.c_str());
  }
};

class IndependentQuantizedGraph : public testing::TestWithParam<unsigned> {};

TEST_P(IndependentQuantizedGraph, BuildQueryAndSnapshotWithoutHnsw) {
  if (!QuantizedGraphSupported()) GTEST_SKIP();
  const unsigned variant = GetParam();
  ArrayGraph source(variant & 1, variant & 2);
  QuantizedGraphBuildOptions options;
  options.threads = 2;
  options.training_samples = 16;
  if (variant & 4) {
    options.codec = QuantizedGraphBuildOptions::Codec::LowRank;
    options.block_dimension = 8;
    options.rotation_views = 2;
  }
  std::string codec, router;
  ASSERT_EQ(0, QuantizedGraphIndex::train_codec(source, source.count,
                                                source.dimension, source.cosine,
                                                source.fp16, options, codec));
  const size_t bytes = source.dimension * (source.fp16 ? 2 : 4);
  QuantizedGraphIndex index(source.dimension, 32, source.cosine, bytes, router,
                            source.fp16, &codec);
  ailego::Params pages;
  pages.set(quantized_graph_pages::kPolicy, std::string("hint"));
  ASSERT_EQ(0, index.prebuild(source, source.count, 2, pages));

  auto exact = [&source](const void *v, const void *q, size_t n, float *out) {
    source.distance(v, q, n, out);
  };
  QuantizedGraphQuery context(source.dimension, exact);
  QuantizedGraphSearchOptions query_options;
  query_options.route_beam = 0;
  query_options.max_expansions = 128;
  query_options.margin = 1e6f;
  auto run = [&](QuantizedGraphIndex &active, bool filtered) {
    for (size_t row = 0; row < 5; ++row) {
      const float *q = source.floats.data() + row * source.dimension;
      auto exclude = [filtered](uint32_t id) { return filtered && id % 2; };
      EXPECT_EQ(0, context.reset(q, 5, 20, query_options, 0, exclude));
      EXPECT_EQ(0, active.search(source, context));
      std::vector<std::pair<float, uint32_t>> expected;
      for (uint32_t id = 0; id < source.count; ++id) {
        if (!source.valid(id) || exclude(id)) continue;
        ArrayGraph::Vector v;
        EXPECT_EQ(0, source.get_vector(id, v));
        float distance;
        exact(v.data(), q, source.dimension, &distance);
        expected.emplace_back(distance, id);
      }
      std::sort(expected.begin(), expected.end());
      const auto &result = context.results();
      EXPECT_GE(result.size(), 5u);
      for (size_t i = 0; i < std::min(size_t{5}, result.size()); ++i) {
        EXPECT_EQ(expected[i].second, result[i].first);
        EXPECT_FLOAT_EQ(expected[i].first, result[i].second);
      }
      EXPECT_GT(context.expanded(), 0u);
    }
  };
  run(index, false);
  run(index, true);

  File file;
  auto dump = IndexFactory::CreateDumper("FileDumper");
  ASSERT_TRUE(dump);
  ASSERT_EQ(0, dump->init({}));
  ASSERT_EQ(0, dump->create(file.name));
  ASSERT_EQ(0, index.dump(dump));
  ASSERT_EQ(0, dump->close());
  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_TRUE(storage);
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(file.name, false));
  auto segment = storage->get(QuantizedGraphIndex::kSnapshotSegment);
  ASSERT_TRUE(segment);
  std::shared_ptr<QuantizedGraphIndex> loaded;
  ASSERT_EQ(0, QuantizedGraphIndex::load(source, source.count, source.dimension,
                                         source.cosine, bytes, router,
                                         source.fp16, segment, loaded, pages));
  ASSERT_TRUE(loaded);
  run(*loaded, false);
  run(*loaded, true);

  auto previous = loaded;
  EXPECT_EQ(IndexError_InvalidFormat,
            QuantizedGraphIndex::load(
                source, source.count, source.dimension + 1, source.cosine,
                bytes, router, source.fp16, segment, loaded, pages));
  EXPECT_EQ(previous, loaded);
  EXPECT_EQ(IndexError_InvalidFormat,
            QuantizedGraphIndex::load(source, source.count, source.dimension,
                                      source.cosine, bytes, "wrong router",
                                      source.fp16, segment, loaded, pages));
  EXPECT_EQ(previous, loaded);
  EXPECT_EQ(IndexError_NoExist,
            QuantizedGraphIndex::load(source, source.count, source.dimension,
                                      source.cosine, bytes, router, source.fp16,
                                      nullptr, loaded, pages));

  pages.set(quantized_graph_pages::kPolicy, std::string("require"));
  pages.set(quantized_graph_pages::kMinimum, 1.01);
  EXPECT_EQ(IndexError_InvalidArgument,
            QuantizedGraphIndex::load(source, source.count, source.dimension,
                                      source.cosine, bytes, router, source.fp16,
                                      segment, loaded, pages));
  EXPECT_EQ(previous, loaded);

  for (uint64_t budget : {1u, 2u, 3u, 5u, 9u}) {
    ASSERT_EQ(
        0, context.reset(source.floats.data(), 10, 20, query_options, budget));
    ASSERT_EQ(0, index.search(source, context));
    EXPECT_LE(context.comparisons(), budget);
  }
  for (uint32_t cap : {1u, 2u, 3u, 7u}) {
    auto limited = query_options;
    limited.max_expansions = cap;
    ASSERT_EQ(0, context.reset(source.floats.data(), 10, 20, limited, 0,
                               [](uint32_t) { return true; }));
    ASSERT_EQ(0, index.search(source, context));
    EXPECT_LE(context.expanded(), cap);
    EXPECT_EQ(0u, context.comparisons());
    EXPECT_EQ(0u, context.results().size());
  }

  // A valid outer container must not bless a modified block payload.
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(segment->data_size(),
            segment->read(0, block, segment->data_size()));
  std::string damaged(static_cast<const char *>(block.data()),
                      segment->data_size());
  damaged.back() ^= 1;
  File bad_file;
  auto bad_dump = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, bad_dump->init({}));
  ASSERT_EQ(0, bad_dump->create(bad_file.name));
  ASSERT_EQ(damaged.size(), bad_dump->write(damaged.data(), damaged.size()));
  ASSERT_EQ(0, bad_dump->append(QuantizedGraphIndex::kSnapshotSegment,
                                damaged.size(), 0, segment->data_crc()));
  ASSERT_EQ(0, bad_dump->close());
  auto bad_storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, bad_storage->init({}));
  ASSERT_EQ(0, bad_storage->open(bad_file.name, false));
  ailego::Params hint;
  hint.set(quantized_graph_pages::kPolicy, std::string("hint"));
  EXPECT_EQ(
      IndexError_InvalidFormat,
      QuantizedGraphIndex::load(
          source, source.count, source.dimension, source.cosine, bytes, router,
          source.fp16, bad_storage->get(QuantizedGraphIndex::kSnapshotSegment),
          loaded, hint));
  EXPECT_EQ(previous, loaded);
  ASSERT_EQ(0, bad_storage->close());

  ASSERT_EQ(0, context.reset(source.floats.data(), 10, 10, query_options, 1));
  ASSERT_EQ(0, index.search(source, context));
  EXPECT_LE(context.comparisons(), 1u);
  ASSERT_EQ(0, context.reset(source.floats.data(), 0, 0, query_options));
  EXPECT_EQ(0, index.search(source, context));
  EXPECT_EQ(0u, context.results().size());
  EXPECT_EQ(0, storage->close());
}

TEST_P(IndependentQuantizedGraph, RouterTrainingWithoutHnsw) {
  if (!QuantizedGraphSupported()) GTEST_SKIP();
  ArrayGraph source(GetParam() & 1, GetParam() & 2);
  source.deleted = false;
  const size_t views = GetParam() & 4 ? 2 : 1;
  std::string router;
  ASSERT_EQ(0, QuantizedGraphIndex::train_router(
                   source, source.count, source.dimension, source.cosine,
                   source.count, 32, 718, router, source.fp16, views));
  ASSERT_FALSE(router.empty());
  QuantizedGraphBuildOptions build;
  build.rotation_views = views;
  std::string codec;
  ASSERT_EQ(0, QuantizedGraphIndex::train_codec(source, source.count,
                                                source.dimension, source.cosine,
                                                source.fp16, build, codec));
  QuantizedGraphIndex index(source.dimension, 32, source.cosine,
                            source.dimension * (source.fp16 ? 2 : 4), router,
                            source.fp16, &codec);
  ailego::Params pages;
  pages.set(quantized_graph_pages::kPolicy, std::string("hint"));
  ASSERT_EQ(0, index.prebuild(source, source.count, 2, pages));
  QuantizedGraphQuery query(
      source.dimension,
      [&source](const void *v, const void *q, size_t n, float *out) {
        source.distance(v, q, n, out);
      });
  QuantizedGraphSearchOptions options;
  options.margin = 1e6f;
  for (uint32_t beam : {0u, 1u, 4u}) {
    options.route_beam = beam;
    ASSERT_EQ(0, query.reset(source.floats.data(), 5, source.count, options));
    ASSERT_EQ(0, index.search(source, query));
    ASSERT_GE(query.results().size(), 5u);
    EXPECT_EQ(0u, query.results()[0].first);
  }
}

INSTANTIATE_TEST_SUITE_P(StorageMetricCodec, IndependentQuantizedGraph,
                         testing::Values(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u));

TEST(QuantizedGraphQuery, InvalidOptionsClearPreviousResults) {
  float q[8]{};
  QuantizedGraphQuery ctx(
      8, [](const void *, const void *, size_t, float *out) { *out = 0; });
  ASSERT_EQ(0, ctx.reset(q, 3, 3));
  ctx.reset_results().emplace(1, 0);
  QuantizedGraphSearchOptions options;
  options.route_beam = 5;
  EXPECT_EQ(IndexError_InvalidArgument, ctx.reset(q, 3, 3, options));
  EXPECT_EQ(0u, ctx.results().size());
  EXPECT_EQ(0u, ctx.topk());
  EXPECT_EQ(IndexError_InvalidArgument, ctx.reset(nullptr, 3, 3));
}

TEST(QuantizedGraphQuery, GeometryAndMetricDimensionsRemainDistinct) {
  float q[9]{};
  size_t seen = 0;
  QuantizedGraphQuery query(
      8,
      [&seen](const void *, const void *, size_t dim, float *out) {
        seen = dim;
        *out = 42;
      },
      9);
  ASSERT_EQ(0, query.reset(q, 1, 1));
  EXPECT_EQ(42, query.distance(q));
  EXPECT_EQ(9u, seen);
}

TEST(QuantizedGraphPages, IndependentParamsOverrideLegacyKeys) {
  ailego::Params params;
  params.set("proxima.hnsw.quantized_graph.thp_policy", std::string("typo"));
  params.set(quantized_graph_pages::kPolicy, std::string("require"));
  params.set(quantized_graph_pages::kMinimum, 0.95);
  params.set(quantized_graph_pages::kFloor, 0.8);
  EXPECT_EQ(0, quantized_graph_pages::prepare_arena(nullptr, 0, params));
  params.set(quantized_graph_pages::kFloor, 0.96);
  EXPECT_EQ(IndexError_InvalidArgument,
            quantized_graph_pages::prepare_arena(nullptr, 0, params));
}

}  // namespace
#endif
}  // namespace zvec::core
