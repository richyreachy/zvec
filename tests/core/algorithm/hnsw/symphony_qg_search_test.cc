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

#include <gtest/gtest.h>
#include "hnsw_symphony_qg.h"
#if RABITQ_SUPPORTED
#include <rabitqlib/utils/cpu_features.hpp>
#endif

namespace zvec::core {
namespace {

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
