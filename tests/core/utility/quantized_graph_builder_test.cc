// Copyright 2026-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include "algorithm/quantized_graph/quantized_graph_builder.h"
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <gtest/gtest.h>
#include "algorithm/quantized_graph/quantized_graph_query.h"

namespace zvec::core {
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
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
  QuantizedGraphSearchOptions search;
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
      // The QG context retains a sorted candidate heap; the caller takes Top-K.
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

class RawQuantizedGraph : public testing::TestWithParam<unsigned> {};

TEST_P(RawQuantizedGraph, BuildsAndSearchesWithoutAnotherIndex) {
  if (!QuantizedGraphSupported()) GTEST_SKIP();
  const auto variant = GetParam();
  auto config = options();
  if (variant & 2) {
    config.encoding.codec = QuantizedGraphBuildOptions::Codec::LowRank;
    config.encoding.block_dimension = 8;
    config.encoding.codebook = QuantizedGraphBuildOptions::Codebook::Gaussian;
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
#else
TEST(QuantizedGraphBuilder, ReportsUnsupportedBackend) {
  QuantizedGraphBuilder::Result result;
  QuantizedGraphBuilder::Options config;
  EXPECT_EQ(IndexError_Unsupported,
            QuantizedGraphBuilder::build({}, 32, false, config, result));
}
#endif
}  // namespace zvec::core
