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
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "hnsw_symphony_qg.h"
#include "symphony_qg_beam.h"
#include "symphony_qg_rotation.h"
#if RABITQ_SUPPORTED
#include <rabitqlib/quantization/rabitq.hpp>
#include <rabitqlib/utils/cpu_features.hpp>
#include "symphony_qg_utils.h"
#endif

namespace zvec::core {
namespace {

TEST(SymphonyQGTest, BeamKeepsFullWidthIdsAndRevisitsBetterEstimates) {
  SymphonyQGBeam beam(2);
  beam.insert(0x80000001U, 5);
  beam.insert(2, 10);
  EXPECT_EQ(0x80000001U, beam.pop());
  beam.insert(3, 20);  // beyond the beam
  beam.insert(2, 1);   // a better estimate from a different center
  ASSERT_TRUE(beam.has_next());
  EXPECT_EQ(2U, beam.pop());
  EXPECT_FALSE(beam.has_next());
  beam.insert(4, 0);  // can reopen a previously exhausted beam
  EXPECT_EQ(4U, beam.pop());
  EXPECT_FALSE(beam.has_next());
}

TEST(SymphonyQGTest, BeamCapacityAndEqualEstimates) {
  SymphonyQGBeam beam(0);
  beam.insert(1, 5);
  beam.insert(2, 4);
  EXPECT_EQ(2U, beam.pop());
  EXPECT_FALSE(beam.has_next());
  beam.insert(3, 4);
  ASSERT_TRUE(beam.has_next());
  EXPECT_EQ(3U, beam.pop());
  EXPECT_FALSE(beam.has_next());
}

TEST(SymphonyQGTest, BeamLookaheadSkipsExpandedEntriesAndResetClearsState) {
  SymphonyQGBeam beam(4);
  beam.insert(0x80000001U, 1);
  beam.insert(2, 2);
  beam.insert(0xFFFFFFFEU, 3);
  EXPECT_EQ(0x80000001U, beam.pop());
  beam.insert(0x80000004U, 0);
  ASSERT_TRUE(beam.has_next_at(0));
  EXPECT_EQ(0x80000004U, beam.next_id_at(0));
  ASSERT_TRUE(beam.has_next_at(1));
  EXPECT_EQ(2U, beam.next_id_at(1));
  ASSERT_TRUE(beam.has_next_at(2));
  EXPECT_EQ(0xFFFFFFFEU, beam.next_id_at(2));
  EXPECT_FALSE(beam.has_next_at(3));
  EXPECT_EQ(0x80000004U, beam.pop());
  EXPECT_EQ(2U, beam.pop());
  EXPECT_EQ(0xFFFFFFFEU, beam.pop());
  EXPECT_FALSE(beam.has_next());
  beam.reset(1);
  EXPECT_FALSE(beam.has_next());
  beam.insert(0x80000001U, 5);
  EXPECT_EQ(0x80000001U, beam.pop());
  beam.reset(8);
  beam.insert(0x80000002U, 6);
  EXPECT_EQ(0x80000002U, beam.pop());
}

TEST(SymphonyQGRotationTest, PreservesDistancesAndRebuildsIdentically) {
  for (size_t dim : {1U, 16U, 63U, 64U, 65U, 128U, 1025U, 4096U}) {
    SCOPED_TRACE(dim);
    SymphonyQGRotation rotation(dim), reopened(dim);
    std::vector<float> a(dim), b(dim), ra, rb, repeat;
    double original = 0;
    for (size_t i = 0; i < dim; ++i) {
      a[i] = std::sin(static_cast<float>(i));
      b[i] = std::cos(static_cast<float>(i) * 0.7f);
      original += (a[i] - b[i]) * (a[i] - b[i]);
    }
    rotation.rotate(a.data(), ra);
    rotation.rotate(b.data(), rb);
    reopened.rotate(a.data(), repeat);
    EXPECT_EQ(ra, repeat);
    double transformed = 0;
    for (size_t i = 0; i < ra.size(); ++i) {
      transformed += (ra[i] - rb[i]) * (ra[i] - rb[i]);
    }
    EXPECT_NEAR(original, transformed, 1e-5 * std::max(1.0, original));
  }
}

#if RABITQ_SUPPORTED
// A fixed level-zero graph makes completion and entry selection deterministic.
class SymphonyTestEntity : public HnswEntity {
 public:
  explicit SymphonyTestEntity(std::vector<float> values)
      : values_(std::move(values)),
        keys_(values_.size()),
        links_(values_.size()) {
    *mutable_doc_cnt() = static_cast<node_id_t>(values_.size());
    set_vector_size(sizeof(float));
    for (size_t i = 0; i < keys_.size(); ++i) keys_[i] = i;
  }

  key_t get_key(node_id_t id) const override {
    return keys_.at(id);
  }
  const void *get_vector(node_id_t id) const override {
    return &values_.at(id);
  }
  int get_vector(const node_id_t *ids, uint32_t count,
                 const void **vectors) const override {
    for (uint32_t i = 0; i < count; ++i) vectors[i] = get_vector(ids[i]);
    return 0;
  }
  int get_vector(node_id_t id,
                 IndexStorage::MemoryBlock &block) const override {
    block = IndexStorage::MemoryBlock::MakeBorrowedView(
        const_cast<float *>(&values_.at(id)));
    return 0;
  }
  int get_vector(
      const node_id_t *ids, uint32_t count,
      std::vector<IndexStorage::MemoryBlock> &blocks) const override {
    blocks.resize(count);
    for (uint32_t i = 0; i < count; ++i) get_vector(ids[i], blocks[i]);
    return 0;
  }
  const Neighbors get_neighbors(level_t, node_id_t id) const override {
    const auto &links = links_.at(id);
    return Neighbors(static_cast<uint32_t>(links.size()), links.data());
  }

  std::vector<float> values_;
  std::vector<key_t> keys_;
  std::vector<std::vector<node_id_t>> links_;
};

TEST(SymphonyQGTest, PrebuildSelectsMeanOfValidVectors) {
  if (!rabitqlib::cpu::has_avx2() && !rabitqlib::cpu::has_avx512_core()) {
    GTEST_SKIP() << "RaBitQ requires AVX2/FMA or AVX512";
  }
  SymphonyTestEntity entity({1.0f, 2.0f, 10.0f});
  HnswSymphonyQG index(1);
  for (size_t threads : {1U, 3U}) {
    SCOPED_TRACE(threads);
    ASSERT_EQ(0, index.prebuild(entity, entity.doc_cnt(), threads));
    EXPECT_EQ(1U, index.entry());  // nearest 13/3, not nearest the sum 13
  }

  // Three live vectors and enough deleted slots that dividing by doc_cnt
  // instead of the live count would incorrectly select node zero.
  SymphonyTestEntity holes({1.0f, 2.0f, 10.0f, 100.0f, 100.0f, 100.0f, 100.0f,
                            100.0f, 100.0f, 100.0f});
  std::fill(holes.keys_.begin() + 3, holes.keys_.end(), kInvalidKey);
  ASSERT_EQ(0, index.prebuild(holes, holes.doc_cnt(), 3));
  EXPECT_EQ(1U, index.entry());
  std::fill(holes.keys_.begin(), holes.keys_.end(), kInvalidKey);
  ASSERT_EQ(0, index.prebuild(holes, holes.doc_cnt(), 3));
  EXPECT_EQ(kInvalidNodeId, index.entry());
  ASSERT_EQ(0, index.prebuild(entity, 0, 1));
  EXPECT_EQ(kInvalidNodeId, index.entry());
}

TEST(SymphonyQGTest, CompletionHonorsExclusionFilterAndScanBudget) {
  if (!rabitqlib::cpu::has_avx2() && !rabitqlib::cpu::has_avx512_core()) {
    GTEST_SKIP() << "RaBitQ requires AVX2/FMA or AVX512";
  }
  auto entity = std::make_shared<SymphonyTestEntity>(
      std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f});
  entity->links_[0] = {1, 2, 3};
  HnswSymphonyQG index(1);
  const float query = 1.0f;
  HnswContext ctx(1, nullptr, entity);
  ctx.set_ef(1);
  ctx.set_topk(1);
  ctx.dist_calculator().update_distance(
      [](const void *lhs, const void *rhs, size_t, float *out) {
        const float diff =
            *static_cast<const float *>(lhs) - *static_cast<const float *>(rhs);
        *out = diff * diff;
      },
      {});
  ctx.dist_calculator().reset_query(&query);

  for (bool prebuilt : {true, false}) {
    SCOPED_TRACE(prebuilt);
    if (prebuilt) {
      ASSERT_EQ(0, index.prebuild(*entity, entity->doc_cnt(), 1));
    } else {
      index.clear();
    }
    for (uint32_t budget : {2U, 100U}) {
      SCOPED_TRACE(budget);
      ctx.set_max_scan_num(budget);
      ctx.dist_calculator().clear_compare_cnt();
      ctx.set_filter([](uint64_t key) { return key != 3; });
      ASSERT_EQ(0, index.search(0, ctx));
      // Beam capacity one expands nodes 0 and 1; only completion can find 3.
      EXPECT_EQ(budget == 2 ? 0U : 1U, ctx.search_heap().size());
      ctx.search_heap().for_each([](node_id_t id, dist_t distance) {
        EXPECT_EQ(3U, id);
        EXPECT_FLOAT_EQ(4.0f, distance);
        return true;
      });
      EXPECT_LE(ctx.dist_calculator().compare_cnt(), budget);
    }
    ctx.dist_calculator().clear_compare_cnt();
    ctx.set_filter([](uint64_t) { return true; });
    ASSERT_EQ(0, index.search(0, ctx));
    EXPECT_EQ(0U, ctx.search_heap().size());

    ctx.dist_calculator().clear_compare_cnt();
    ctx.reset_filter();
    ASSERT_EQ(0, index.search(0, ctx));
    ASSERT_EQ(1U, ctx.search_heap().size());
    ctx.search_heap().for_each([](node_id_t id, dist_t distance) {
      EXPECT_EQ(1U, id);
      EXPECT_FLOAT_EQ(0.0f, distance);
      return true;
    });
  }
}

TEST(SymphonyQGTest, FastScanHandlesPartialBlocksDuplicatesAndLargeDimensions) {
  if (!rabitqlib::cpu::has_avx2() && !rabitqlib::cpu::has_avx512_core()) {
    GTEST_SKIP() << "RaBitQ requires AVX2/FMA or AVX512";
  }
  for (size_t dim : {64U, 128U, 1024U, 2048U, 4096U}) {
    for (size_t count : {1U, 17U, 32U}) {
      SCOPED_TRACE(std::to_string(dim) + "/" + std::to_string(count));
      // Constant positive query deliberately produces large LUT sums at 4096D.
      std::vector<float> query(dim, 1.0f), center(dim, 0.5f);
      std::vector<float> data(count * dim);
      for (size_t i = 0; i < count; ++i) {
        std::fill(data.begin() + i * dim, data.begin() + (i + 1) * dim,
                  0.5f + i * 0.125f);
      }
      std::vector<char> codes(rabitqlib::QGBatchDataMap<float>::data_bytes(dim),
                              0);
      rabitqlib::quant::quantize_qg_batch(data.data(), center.data(), count,
                                          dim, codes.data(),
                                          rabitqlib::METRIC_L2);
      SymQuery q;
      q.reset(query.data(), dim);
      q.set_g_add(0.25f * dim);
      std::array<float, 32> result;
      ScanSymphonyQGBatch(codes.data(), q, dim, result.data());
      for (size_t i = 0; i < count; ++i) {
        const float diff = 0.5f + i * 0.125f - 1.0f;
        EXPECT_NEAR(diff * diff * dim, result[i], 0.01f * dim);
        EXPECT_TRUE(std::isfinite(result[i]));
      }
    }
  }
}

TEST(SymphonyQGTest, ReusedQueryMatchesReferenceLut) {
  if (!rabitqlib::cpu::has_avx2() && !rabitqlib::cpu::has_avx512_core()) {
    GTEST_SKIP() << "RaBitQ requires AVX2/FMA or AVX512";
  }
  SymQuery actual;
  for (size_t dim : {64U, 128U, 1024U, 2048U, 4096U, 64U}) {
    for (int pattern : {0, 1, 2}) {
      SCOPED_TRACE(std::to_string(dim) + "/" + std::to_string(pattern));
      std::vector<float> query(dim, 0.0f);
      for (size_t i = 0; i < dim; ++i) {
        if (pattern == 1) query[i] = 1.0f;
        // Exactly representable mixed signs keep subset sums reproducible.
        if (pattern == 2)
          query[i] = (static_cast<int>(i * 37 % 101) - 50) / 16.0f;
      }
      actual.set_g_add(123.0f);
      actual.reset(query.data(), dim);
      rabitqlib::BatchQuery<float> expected(query.data(), dim);
      EXPECT_FLOAT_EQ(0.0f, actual.g_add());
      EXPECT_FLOAT_EQ(expected.delta(), actual.delta());
      EXPECT_FLOAT_EQ(expected.sum_vl_lut(), actual.sum_vl_lut());
      EXPECT_FLOAT_EQ(expected.k1xsumq(), actual.k1xsumq());
      for (size_t i = 0; i < dim * 4; ++i) {
        // SIMD and reference rounding can differ by one quantization step.
        EXPECT_LE(std::abs(int(expected.lut()[i]) - int(actual.lut()[i])), 1);
      }
    }
  }
}
#else
TEST(SymphonyQGTest, PrebuildIsLinkableOnUnsupportedPlatforms) {
  // A volatile member pointer forces the linker to resolve the symbol even
  // when this platform cannot construct or execute a RaBitQ index.
  using Prebuild = int (HnswSymphonyQG::*)(const HnswEntity &, size_t, size_t);
  volatile Prebuild prebuild = &HnswSymphonyQG::prebuild;
  EXPECT_TRUE(prebuild != nullptr);
}
#endif

}  // namespace
}  // namespace zvec::core
