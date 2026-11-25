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
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/preprocessor/quantized_graph_rotator/quantized_graph_rotator.h>
#include <turbo/quantizer/quantized_graph_quantizer/quantized_graph_quantizer.h>
#include "algorithm/quantized_graph/quantized_graph_integer_router.h"
#include "algorithm/quantized_graph/quantized_graph_router_builder.h"

namespace zvec::turbo {
namespace {
double Dot(const std::vector<float> &a, const std::vector<float> &b) {
  double result = 0;
  for (size_t i = 0; i < a.size(); ++i) result += double(a[i]) * b[i];
  return result;
}

TEST(QuantizedGraphRotatorViews,
     PreservesInnerProductsAndNormsAcrossDimensions) {
  std::mt19937 random(991);
  std::uniform_real_distribution<float> value(-1.f, 1.f);
  for (size_t dimension : {1u, 65u, 100u, 768u, 960u, 4096u}) {
    std::vector<float> a(dimension), b(dimension), ra, rb, fast;
    for (float &v : a) v = value(random);
    for (float &v : b) v = value(random);
    for (size_t views : {1u, 2u, 4u}) {
      SCOPED_TRACE(testing::Message() << dimension << ":" << views);
      QuantizedGraphRotator rotation(dimension, views);
      rotation.rotate(a.data(), ra);
      rotation.rotate(b.data(), rb);
      EXPECT_EQ(QuantizedGraphRotator(dimension).padded_dim() * views,
                ra.size());
      EXPECT_NEAR(Dot(a, a), Dot(ra, ra), 3e-6 * Dot(a, a));
      EXPECT_NEAR(Dot(a, b), Dot(ra, rb),
                  3e-6 * std::sqrt(Dot(a, a) * Dot(b, b)));
      rotation.rotate(a.data(), fast, true);
      ASSERT_EQ(ra.size(), fast.size());
      for (size_t i = 0; i < ra.size(); ++i) EXPECT_NEAR(ra[i], fast[i], 1e-5);
      QuantizedGraphRotator fresh(dimension, views);
      fresh.rotate(a.data(), fast);
      EXPECT_EQ(ra, fast);
    }
  }
}

TEST(QuantizedGraphRotatorViews, ExpandedFastScanDoesNotOverflow) {
  for (size_t dimension : {1536u, 2048u, 3072u, 8192u, 16384u}) {
    SCOPED_TRACE(dimension);
    std::vector<uint8_t> codes(4 * dimension, 0), lut(4 * dimension, 255);
    std::array<float, 32> add{}, scale, actual{};
    scale.fill(1.f);
    if (!quantized_graph_scan(codes.data(), lut.data(), dimension, add.data(),
                              scale.data(), 0, 1, 0, 0, actual.data()))
      GTEST_SKIP();
    // Every four-dimensional group contributes 255, independently of code.
    for (float score : actual) EXPECT_EQ(float(dimension / 4 * 255), score);
  }
}

TEST(QuantizedGraphRotatorViews, RejectsInvalidShape) {
  for (size_t views : {0u, 3u, 8u, 128u})
    EXPECT_THROW(QuantizedGraphRotator(100, views), std::invalid_argument);
  EXPECT_THROW(QuantizedGraphRotator(0), std::invalid_argument);
  EXPECT_THROW(QuantizedGraphRotator(4097), std::invalid_argument);
}

#if RABITQ_SUPPORTED
TEST(QuantizedGraphRotatorViews,
     RouterLoadValidatesMeanTransformAndGraphTogether) {
  if (!__builtin_cpu_supports("avx512vnni")) GTEST_SKIP();
  using Builder = core::QuantizedGraphRouterBuilder;
  struct Graph {
    bool valid(uint32_t id) const {
      return id == 0;
    }
  } graph;
  Builder::Model model;
  model.docs = 1;
  model.mean.assign(100, 0.f);
  model.mean[0] = 1.f;
  model.projected = 32;
  model.groups.resize(1);
  model.groups[0].count = 1;
  model.groups[0].centers[0] = model.mean;
  model.groups[0].targets[0] = -1;
  std::ostringstream out(std::ios::out | std::ios::binary);
  model.write(out);
  const auto bytes = out.str();
  QuantizedGraphBlockQuantizer codec(100, true);
  core::QuantizedGraphIntegerRouter router;
  std::vector<float> mean(100), rotated;
  ASSERT_TRUE(router.load_data(bytes, mean, graph, 1, codec, true));
  EXPECT_EQ(model.mean, mean);
  codec.transform(mean.data(), rotated);
  core::QuantizedGraphIntegerRouter::Query query;
  ASSERT_TRUE(router.prepare_projected(rotated.data(), query));
  EXPECT_EQ(0u, router.choose(query, 1));

  core::QuantizedGraphRouterReader reader{bytes};
  core::QuantizedGraphRouterFormat format;
  ASSERT_TRUE(format.read_bytes(reader));
  // A valid-sized but inconsistent mean must not silently change routing.
  auto inconsistent = bytes;
  const float changed = .5f;
  std::memcpy(inconsistent.data() + reader.offset, &changed, sizeof(changed));
  EXPECT_FALSE(router.load_data(inconsistent, mean, graph, 1, codec, true));
  EXPECT_TRUE(router.empty());
  EXPECT_EQ(model.mean, mean);
  for (const auto &invalid : {bytes.substr(0, bytes.size() - 1), bytes + '\0'})
    EXPECT_FALSE(router.load_data(invalid, mean, graph, 1, codec, true));
  EXPECT_FALSE(router.load_data(bytes, mean, graph, 2, codec, true));
  EXPECT_FALSE(router.load_data(bytes, mean, graph, 1, codec, false));
  // A dangling leaf must be rejected even when its file checksum is valid.
  model.groups[0].targets[0] = -2;
  std::ostringstream bad(std::ios::out | std::ios::binary);
  model.write(bad);
  EXPECT_FALSE(router.load_data(bad.str(), mean, graph, 1, codec, true));
}

TEST(QuantizedGraphRotatorViews,
     RaBitQCodecMetadataIsSelfDescribingAndValidated) {
  const std::string legacy;
  EXPECT_EQ(128u,
            QuantizedGraphBlockQuantizer(100, true, legacy).encoded_dim());
  // V2 RaBitQ descriptor: magic, version, input dimension, block=0, kind=0,
  // lambda=0 (float bits), followed by rotation views.
  const uint32_t words[]{0x51474243, 2, 100, 0, 0, 0, 2};
  const std::string model(reinterpret_cast<const char *>(words), sizeof(words));
  QuantizedGraphBlockQuantizer codec(100, true, model);
  EXPECT_EQ(2u, codec.rotation_views());
  EXPECT_EQ(256u, codec.encoded_dim());
  EXPECT_EQ(model, codec.model_data());
  for (size_t size = 1; size < model.size(); ++size) {
    const auto truncated = model.substr(0, size);
    EXPECT_THROW(QuantizedGraphBlockQuantizer(100, true, truncated),
                 std::runtime_error);
  }
  for (uint32_t views : {0u, 3u, 5u, UINT32_MAX}) {
    auto invalid = model;
    std::memcpy(invalid.data() + 24, &views, 4);
    EXPECT_THROW(QuantizedGraphBlockQuantizer(100, true, invalid),
                 std::runtime_error);
  }
  auto extra = model + '\0';
  EXPECT_THROW(QuantizedGraphBlockQuantizer(100, true, extra),
               std::runtime_error);
  EXPECT_THROW(QuantizedGraphBlockQuantizer(99, true, model),
               std::runtime_error);
}
#endif

TEST(QuantizedGraphRotatorViews,
     RouterFormatPersistsViewCountAndKeepsLegacyVersions) {
  using Builder = core::QuantizedGraphRouterBuilder;
  for (size_t dimension : {100u, 768u, 960u}) {
    for (size_t views : {1u, 2u, 4u}) {
      Builder::Model model;
      model.docs = 1;
      model.mean.assign(dimension, 0.f);
      model.mean[0] = 1.f;
      model.projected = dimension == 768 ? 384 : 32;
      model.rotation_views = views;
      model.groups.resize(1);
      model.groups[0].count = 1;
      model.groups[0].centers[0] = model.mean;
      model.groups[0].targets[0] = -1;
      std::ostringstream out(std::ios::out | std::ios::binary);
      model.write(out);
      auto bytes = out.str();
      core::QuantizedGraphRouterReader reader{bytes};
      core::QuantizedGraphRouterFormat format;
      ASSERT_TRUE(format.read_bytes(reader));
      EXPECT_EQ(views, format.rotation_views);
      EXPECT_EQ(QuantizedGraphRotator(dimension, views).padded_dim(),
                format.rotated);
      EXPECT_EQ(views == 1 ? (dimension == 768 ? 2u : 3u) : 4u, format.version);
      if (views != 1) {
        const uint32_t invalid = 3;
        std::memcpy(bytes.data() + 36, &invalid, 4);
        core::QuantizedGraphRouterReader bad{bytes};
        EXPECT_FALSE(format.read_bytes(bad));
      }
    }
  }
}
}  // namespace
}  // namespace zvec::turbo
