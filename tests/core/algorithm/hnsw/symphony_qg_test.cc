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
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "algorithm/quantized_graph/quantized_graph.h"
#include "algorithm/quantized_graph/symphony_qg_codec.h"
#include "utility/linear_pool.h"
#include "hnsw_symphony_qg.h"
#if RABITQ_SUPPORTED
#include <rabitqlib/quantization/rabitq.hpp>
#include <rabitqlib/utils/cpu_features.hpp>
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_factory.h>
#include "hnsw_context.h"
#include "hnsw_streamer.h"
#endif

namespace zvec::core {
namespace {

// These adapters deliberately have no HNSW or RaBitQ dependencies. A three-
// vector int8 batch also catches assumptions about SymphonyQG's batch of 32.
class ScalarGraphCodec {
 public:
  static constexpr size_t kBatchSize = 3;
  using Query = std::vector<float>;
  explicit ScalarGraphCodec(size_t dimension) : dimension_(dimension) {}
  size_t encoded_dim() const {
    return dimension_;
  }
  size_t batch_bytes() const {
    return kBatchSize * dimension_;
  }
  void transform(const void *vector, std::vector<float> &out) const {
    const auto *values = static_cast<const float *>(vector);
    out.assign(values, values + dimension_);
  }
  void encode(const float *, const float *vectors, size_t count,
              char *codes) const {
    for (size_t i = 0; i < count * dimension_; ++i) {
      const int8_t value = static_cast<int8_t>(vectors[i]);
      std::memcpy(codes + i, &value, sizeof(value));
    }
  }
  void prepare_query(const void *vector, Query &query) const {
    transform(vector, query);
  }
  void scan(const char *codes, Query &query, float, float *distances) const {
    for (size_t i = 0; i < kBatchSize; ++i) {
      distances[i] = 0;
      for (size_t d = 0; d < dimension_; ++d) {
        int8_t value;
        std::memcpy(&value, codes + i * dimension_ + d, sizeof(value));
        const float diff = query[d] - value;
        distances[i] += diff * diff;
      }
    }
  }

 private:
  size_t dimension_;
};

struct PlainGraph {
  struct Vector {
    const float *values = nullptr;
    const void *data() const {
      return values;
    }
  };
  explicit PlainGraph(std::vector<std::vector<float>> input)
      : vectors(std::move(input)),
        links(vectors.size()),
        live(vectors.size(), true) {}
  bool valid(uint32_t id) const {
    return id < live.size() && live[id];
  }
  const std::vector<uint32_t> &neighbors(uint32_t id) const {
    return links.at(id);
  }
  int get_vector(uint32_t id, Vector &out) const {
    if (id == failed_id) return IndexError_ReadData;
    out.values = vectors.at(id).data();
    return 0;
  }
  std::vector<std::vector<float>> vectors;
  std::vector<std::vector<uint32_t>> links;
  std::vector<bool> live;
  uint32_t failed_id = UINT32_MAX;
};

struct PlainGraphQuery {
  struct Results {
    size_t capacity = 8;
    std::vector<std::pair<uint32_t, float>> values;
    size_t limit() const {
      return capacity;
    }
    size_t size() const {
      return values.size();
    }
    void clear() {
      values.clear();
    }
    void emplace(uint32_t id, float distance) {
      values.emplace_back(id, distance);
      std::sort(values.begin(), values.end(), [](const auto &a, const auto &b) {
        return a.second < b.second ||
               (a.second == b.second && a.first < b.first);
      });
      if (values.size() > capacity) values.resize(capacity);
    }
  } results;
  std::vector<float> vector;
  std::vector<uint32_t> exclusions;
  size_t k = 1, budget = 100, scans = 0, expansions = 0;
  Results &reset_results() {
    scans = 0;
    expansions = 0;
    return results;
  }
  size_t topk() const {
    return k;
  }
  const void *query() const {
    return vector.data();
  }
  bool reach_scan_limit() const {
    return scans >= budget;
  }
  bool excluded(uint32_t id) const {
    return std::find(exclusions.begin(), exclusions.end(), id) !=
           exclusions.end();
  }
  void on_expand() {
    ++expansions;
  }
  float distance(const void *raw) {
    ++scans;
    const auto *values = static_cast<const float *>(raw);
    float distance = 0;
    for (size_t d = 0; d < vector.size(); ++d) {
      const float diff = vector[d] - values[d];
      distance += diff * diff;
    }
    return distance;
  }
};

using PlainQuantizedGraph = QuantizedGraph<ScalarGraphCodec>;

struct SearchVisits {
  explicit SearchVisits(size_t size) : seen(size, false) {}
  bool visited(uint32_t id) const {
    return seen.at(id);
  }
  void set_visited(uint32_t id) {
    seen.at(id) = true;
  }
  std::vector<bool> seen;
};

// A small graph with caller-supplied edge scores isolates visit timing and
// batching from the codec. Delayed mode writes all three lanes, like FastScan.
template <bool Delayed>
struct TraversalScan {
  static constexpr bool kVisitOnExpansion = Delayed;
  std::vector<std::vector<uint32_t>> links;
  std::vector<std::vector<float>> scores;
  std::vector<uint32_t> expanded, scored;
  std::vector<size_t> positions;
  uint32_t current = 0, fail_begin = UINT32_MAX, fail_score = UINT32_MAX;
  size_t duplicates = 0;
  int begin(uint32_t id) {
    current = id;
    expanded.push_back(id);
    return id == fail_begin ? IndexError_ReadData : 0;
  }
  size_t neighbor_count() const {
    return links[current].size();
  }
  size_t batch_size() const {
    return Delayed ? 3 : neighbor_count();
  }
  uint32_t neighbor(size_t i) const {
    return links[current][i];
  }
  void prepare_batch(size_t capacity) {
    positions.resize(capacity);
  }
  void stage(uint32_t, size_t position, size_t slot) {
    positions[slot] = position;
  }
  void on_duplicate() {
    ++duplicates;
  }
  int score(size_t, const uint32_t *ids, size_t count, float *distances) {
    if (current == fail_score) return IndexError_ReadData;
    if constexpr (Delayed) std::fill_n(distances, 3, NAN);
    for (size_t i = 0; i < count; ++i) {
      scored.push_back(ids[i]);
      distances[i] = scores[current][positions[i]];
    }
    return 0;
  }
};

TEST(GraphSearchTest, ExactPoolScoresDiscoveriesOnceAndPreservesLaneOrder) {
  TraversalScan<false> scan;
  scan.links = {{0, 1, 1, 2, 3}, {0, 2, 4}, {3, 4}, {}, {1}};
  scan.scores = {{25, 16, 16, 9, 4}, {25, 9, 1}, {4, 1}, {}, {16}};
  SearchVisits visits(5);
  visits.set_visited(0);
  LinearPool<float> pool(5, 5);
  pool.insert(0, 25);
  GraphSearchPoolFrontier<LinearPool<float>> frontier{pool};
  GraphSearchScratch scratch;
  ASSERT_EQ(0, SearchGraph(scan, frontier, visits, scratch));
  EXPECT_EQ((std::vector<uint32_t>{0, 3, 2, 4, 1}), scan.expanded);
  EXPECT_EQ((std::vector<uint32_t>{1, 2, 3, 4}), scan.scored);
  EXPECT_EQ(7U, scan.duplicates);
  ASSERT_EQ(5, pool.size());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(4 - i, pool.id(i));
    EXPECT_FLOAT_EQ((i + 1) * (i + 1), pool.dist(i));
  }
}

TEST(GraphSearchTest, ScanFailuresStopWithoutPublishingUnscoredCandidates) {
  for (bool fail_on_begin : {true, false}) {
    SCOPED_TRACE(fail_on_begin);
    TraversalScan<false> scan;
    scan.links = {{1}, {2}, {}};
    scan.scores = {{1}, {0}, {}};
    if (fail_on_begin)
      scan.fail_begin = 1;
    else
      scan.fail_score = 1;
    SearchVisits visits(3);
    visits.set_visited(0);
    LinearPool<float> pool(3, 3);
    pool.insert(0, 2);
    GraphSearchPoolFrontier<LinearPool<float>> frontier{pool};
    GraphSearchScratch scratch;
    EXPECT_EQ(IndexError_ReadData,
              SearchGraph(scan, frontier, visits, scratch));
    EXPECT_EQ((std::vector<uint32_t>{0, 1}), scan.expanded);
    EXPECT_EQ((std::vector<uint32_t>{1}), scan.scored);
    ASSERT_EQ(2, pool.size());
    EXPECT_EQ(1, pool.id(0));
    EXPECT_FLOAT_EQ(1, pool.dist(0));
  }
}

TEST(GraphSearchTest, QuantizedBeamReadmitsRejectedNodesWithBetterEstimates) {
  TraversalScan<true> scan;
  scan.links = {{1, 2, 3}, {3, 2, 0}, {3}, {0}};
  scan.scores = {{1, 5, 20}, {0.5f, 0.25f, 0}, {0.1f}, {0}};
  SearchVisits visits(4);
  QuantizedGraphBeam beam(2);
  beam.insert(0, 100);
  struct Frontier {
    QuantizedGraphBeam &beam;
    bool has_next() const {
      return beam.has_next();
    }
    uint32_t pop() {
      return beam.pop();
    }
    bool accepts(float distance) const {
      return std::isfinite(distance) && !beam.is_full(distance);
    }
    void push(uint32_t id, float distance) {
      beam.insert(id, distance);
    }
  } frontier{beam};
  GraphSearchScratch scratch;
  ASSERT_EQ(0, SearchGraph(scan, frontier, visits, scratch));
  // Node 3 is initially rejected at 20, then admitted at 0.5 and improved to
  // 0.1 by another center. Marking it at discovery would lose the result.
  EXPECT_EQ((std::vector<uint32_t>{0, 1, 2, 3}), scan.expanded);
  EXPECT_EQ(0U, scan.duplicates);
  EXPECT_TRUE(std::all_of(visits.seen.begin(), visits.seen.end(),
                          [](bool seen) { return seen; }));
}


TEST(QuantizedGraphTest, IndependentBackendUsesMultipleAndPartialCodecBatches) {
  PlainGraph source({{0, 0}, {1, -1}, {2, -2}, {3, -3}, {4, -4}, {5, -5}});
  source.links[0] = {1, 2, 3, 4, 5};
  PlainQuantizedGraph index(2, ScalarGraphCodec(2), 6);
  PlainGraphQuery query;
  query.vector = {4, -4};
  for (bool prebuilt : {false, true}) {
    SCOPED_TRACE(prebuilt);
    if (prebuilt)
      ASSERT_EQ(0, index.prebuild(source, source.vectors.size(), 2));
    ASSERT_EQ(0, index.search(0, source, query));
    ASSERT_EQ(6U, query.results.size());
    EXPECT_EQ(4U, query.results.values.front().first);
    EXPECT_FLOAT_EQ(0, query.results.values.front().second);
    EXPECT_EQ(6U, query.expansions);
    EXPECT_EQ(6U, query.scans);
  }
}

TEST(QuantizedGraphTest, AuxiliaryNormIsCachedButExcludedFromGeometry) {
  PlainGraph source({{1, 0, 1}, {0, 1, 1000}, {-1, 0, 1}});
  source.links = {{1, 2}, {0, 2}, {0, 1}};
  PlainQuantizedGraph index(2, ScalarGraphCodec(2), 3, 3);
  ASSERT_EQ(0, index.prebuild(source, 3, 2));
  EXPECT_EQ(1U, index.entry());  // norm 1000 must not move the centroid
  struct Query : PlainGraphQuery {
    std::vector<float> norms;
    float distance(const void *raw) {
      norms.push_back(static_cast<const float *>(raw)[2]);
      return PlainGraphQuery::distance(raw);
    }
  } query;
  query.vector = {0, 1};
  for (bool lazy : {false, true}) {
    if (lazy) index.clear();
    query.norms.clear();
    ASSERT_EQ(0, index.search(0, source, query));
    ASSERT_EQ(3U, query.results.size());
    EXPECT_EQ(1U, query.results.values.front().first);
    std::sort(query.norms.begin(), query.norms.end());
    EXPECT_EQ((std::vector<float>{1, 1, 1000}), query.norms);
  }
}

TEST(SymphonyQGTest, CosineFactorsHandleZeroCentersNeighborsAndQueries) {
  const std::array<std::array<float, 2>, 4> vectors{
      {{{1, 0}}, {{0, 1}}, {{-1, 0}}, {{0, 0}}}};
  std::vector<float> packed;
  for (const auto &vector : vectors)
    packed.insert(packed.end(), vector.begin(), vector.end());
  for (const auto &center : vectors) {
    std::array<float, 4> add, scale;
    const float center_norm = center[0] * center[0] + center[1] * center[1];
    for (size_t i = 0; i < vectors.size(); ++i) {
      const auto &x = vectors[i];
      // Exact L2 representation using t = dot(query, x - center).
      add[i] = x[0] * x[0] + x[1] * x[1] - center_norm;
      scale[i] = -2;
    }
    ConvertSymphonyQGCosineFactors(center.data(), packed.data(), vectors.size(),
                                   2, add.data(), scale.data());
    for (const auto &query : vectors) {
      const float g = 1 - query[0] * center[0] - query[1] * center[1];
      for (size_t i = 0; i < vectors.size(); ++i) {
        const auto &x = vectors[i];
        const float t =
            query[0] * (x[0] - center[0]) + query[1] * (x[1] - center[1]);
        EXPECT_FLOAT_EQ(1 - query[0] * x[0] - query[1] * x[1],
                        add[i] + g + scale[i] * t);
      }
    }
  }
}

TEST(QuantizedGraphTest, DegreeBoundUsesCodecBatchSize) {
  PlainGraph source({{0}, {1}, {2}, {3}, {4}, {5}});
  source.links[0] = {5, 4, 3, 2, 1};
  PlainQuantizedGraph index(1, ScalarGraphCodec(1), 4);  // rounds down to three
  ASSERT_EQ(0, index.prebuild(source, source.vectors.size(), 2));
  PlainGraphQuery query;
  query.vector = {3};
  ASSERT_EQ(0, index.search(0, source, query));
  EXPECT_EQ(4U, query.expansions);  // entry plus three pruned neighbors
  ASSERT_EQ(4U, query.results.size());
  EXPECT_EQ(3U, query.results.values.front().first);
}

TEST(QuantizedGraphTest, QuantizedEstimatesDoNotReplaceExactScores) {
  PlainGraph source({{0.25f}, {1.75f}, {2.5f}});
  source.links[0] = {1, 2};
  PlainQuantizedGraph index(1, ScalarGraphCodec(1));
  PlainGraphQuery query;
  query.vector = {1.5f};
  ASSERT_EQ(0, index.search(0, source, query));
  ASSERT_FALSE(query.results.values.empty());
  EXPECT_EQ(1U, query.results.values.front().first);
  // The codec rounds 1.75 to 1 (estimated distance 0.25), whereas the result
  // must use the original vector (exact distance 0.0625).
  EXPECT_FLOAT_EQ(0.0625f, query.results.values.front().second);
}

TEST(QuantizedGraphTest, CompletionFiltersUnvisitedNodesAndHonorsBudget) {
  PlainGraph source({{0}, {1}, {2}, {3}});
  source.links[0] = {1, 2, 3};
  PlainQuantizedGraph index(1, ScalarGraphCodec(1));
  PlainGraphQuery query;
  query.vector = {1};
  query.results.capacity = 1;
  query.exclusions = {0, 1, 2};
  for (bool prebuilt : {false, true}) {
    SCOPED_TRACE(prebuilt);
    if (prebuilt)
      ASSERT_EQ(0, index.prebuild(source, source.vectors.size(), 2));
    query.budget = 2;
    ASSERT_EQ(0, index.search(0, source, query));
    EXPECT_EQ(0U, query.results.size());
    EXPECT_EQ(2U, query.scans);
    query.budget = 3;
    ASSERT_EQ(0, index.search(0, source, query));
    ASSERT_EQ(1U, query.results.size());
    EXPECT_EQ(3U, query.results.values.front().first);
    EXPECT_FLOAT_EQ(4, query.results.values.front().second);
    EXPECT_EQ(3U, query.scans);
    EXPECT_EQ(2U, query.expansions);
  }
}

TEST(QuantizedGraphTest, RebuildAndClearReplaceCachedVectorsAndNeighbors) {
  PlainGraph source({{0}, {1}, {9}});
  source.links[0] = {1};
  PlainQuantizedGraph index(1, ScalarGraphCodec(1));
  PlainGraphQuery query;
  query.vector = {1};
  ASSERT_EQ(0, index.search(0, source, query));  // builds a lazy generation
  ASSERT_EQ(1U, query.results.values.front().first);
  index.clear();
  source.links[0] = {2};
  source.vectors[2][0] = 1;
  source.live[1] = false;
  ASSERT_EQ(0, index.search(0, source, query));
  EXPECT_EQ(2U, query.results.values.front().first);
  index.clear();
  ASSERT_EQ(0, index.prebuild(source, source.vectors.size(), 2));
  ASSERT_EQ(0, index.search(0, source, query));
  EXPECT_EQ(2U, query.results.values.front().first);
  index.clear();
  EXPECT_EQ(PlainQuantizedGraph::kInvalidNodeId, index.entry());
}

TEST(QuantizedGraphTest, CentroidIgnoresHolesAndHandlesEmptyGraphs) {
  PlainGraph source(
      {{1}, {2}, {10}, {100}, {100}, {100}, {100}, {100}, {100}, {100}});
  std::fill(source.live.begin() + 3, source.live.end(), false);
  PlainQuantizedGraph index(1, ScalarGraphCodec(1));
  ASSERT_EQ(0, index.prebuild(source, source.vectors.size(), 3));
  EXPECT_EQ(1U, index.entry());
  std::fill(source.live.begin(), source.live.end(), false);
  ASSERT_EQ(0, index.prebuild(source, source.vectors.size(), 2));
  EXPECT_EQ(PlainQuantizedGraph::kInvalidNodeId, index.entry());
  ASSERT_EQ(0, index.prebuild(source, 0, 1));
  PlainGraphQuery query;
  query.vector = {1};
  ASSERT_EQ(0, index.search(index.entry(), source, query));
  EXPECT_EQ(0U, query.scans);
  EXPECT_EQ(0U, query.results.size());
}

TEST(QuantizedGraphTest, FailedPrebuildDoesNotPublishPartialBlocks) {
  PlainGraph source({{0}, {1}});
  source.links[0] = {1};
  source.failed_id = 1;
  PlainQuantizedGraph index(1, ScalarGraphCodec(1));
  EXPECT_EQ(IndexError_ReadData,
            index.prebuild(source, source.vectors.size(), 2));
  EXPECT_EQ(PlainQuantizedGraph::kInvalidNodeId, index.entry());
  source.failed_id = UINT32_MAX;
  ASSERT_EQ(0, index.prebuild(source, source.vectors.size(), 2));
  PlainGraphQuery query;
  query.vector = {1};
  ASSERT_EQ(0, index.search(0, source, query));
  ASSERT_EQ(1U, query.results.values.front().first);
}

TEST(QuantizedGraphTest, ConcurrentLazySearchesUseSeparateQueryState) {
  PlainGraph source({{0}, {1}, {2}, {3}});
  source.links[0] = {1, 2, 3};
  PlainQuantizedGraph index(1, ScalarGraphCodec(1));
  std::vector<std::thread> threads;
  for (uint32_t id = 0; id < 4; ++id) {
    threads.emplace_back([&, id] {
      PlainGraphQuery query;
      query.vector = {static_cast<float>(id)};
      for (size_t repeat = 0; repeat < 10; ++repeat) {
        ASSERT_EQ(0, index.search(0, source, query));
        ASSERT_FALSE(query.results.values.empty());
        EXPECT_EQ(id, query.results.values.front().first);
      }
    });
  }
  for (auto &thread : threads) thread.join();
}

TEST(SymphonyQGTest, BeamKeepsFullWidthIdsAndRevisitsBetterEstimates) {
  QuantizedGraphBeam beam(2);
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
  QuantizedGraphBeam beam(0);
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
  QuantizedGraphBeam beam(4);
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
  explicit SymphonyTestEntity(std::vector<float> values, size_t dimension = 1)
      : values_(std::move(values)),
        keys_(values_.size() / dimension),
        links_(keys_.size()),
        dimension_(dimension) {
    *mutable_doc_cnt() = static_cast<node_id_t>(keys_.size());
    set_vector_size(dimension * sizeof(float));
    for (size_t i = 0; i < keys_.size(); ++i) keys_[i] = i;
  }

  key_t get_key(node_id_t id) const override {
    return keys_.at(id);
  }
  const void *get_vector(node_id_t id) const override {
    return &values_.at(id * dimension_);
  }
  int get_vector(const node_id_t *ids, uint32_t count,
                 const void **vectors) const override {
    for (uint32_t i = 0; i < count; ++i) vectors[i] = get_vector(ids[i]);
    return 0;
  }
  int get_vector(node_id_t id,
                 IndexStorage::MemoryBlock &block) const override {
    block = IndexStorage::MemoryBlock::MakeBorrowedView(
        const_cast<float *>(&values_.at(id * dimension_)));
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
  size_t dimension_;
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

TEST(SymphonyQGTest, CosineSearchUsesNormalizedCoordinatesAndExactScores) {
  if (!rabitqlib::cpu::has_avx2() && !rabitqlib::cpu::has_avx512_core())
    GTEST_SKIP();
  // Two normalized coordinates followed by the original norm. In particular,
  // the large norm of node 1 must never participate in quantization.
  auto entity = std::make_shared<SymphonyTestEntity>(
      std::vector<float>{0, 0, 0, 1, 0, 300, 0, 1, 8, -1, 0, 4, 0.6f, 0.8f, 10},
      3);
  for (uint32_t i = 0; i < entity->doc_cnt(); ++i)
    for (uint32_t j = 0; j < entity->doc_cnt(); ++j)
      if (i != j) entity->links_[i].push_back(j);
  HnswSymphonyQG index(2, 32, true, 3);
  HnswContext ctx(3, nullptr, entity);
  ctx.set_ef(8);
  ctx.set_topk(5);
  ctx.set_max_scan_num(100);
  IndexMeta raw_meta(IndexMeta::DT_FP32, 2);
  raw_meta.set_metric("Cosine", 0, ailego::Params());
  auto quantizer = IndexFactory::CreateQuantizer("Fp32Quantizer");
  ASSERT_NE(nullptr, quantizer);
  ASSERT_EQ(0, quantizer->init(raw_meta, ailego::Params()));
  for (bool turbo : {false, true}) {
    SCOPED_TRACE(turbo);
    ctx.dist_calculator().set_dim(turbo ? 2 : 3);
    ctx.dist_calculator().update_distance(
        [turbo, quantizer](const void *lhs, const void *rhs, size_t dim,
                           float *out) {
          if (turbo) {
            *out = quantizer->calc_distance_dp_query(lhs, rhs);
          } else {
            const auto *x = static_cast<const float *>(lhs);
            const auto *q = static_cast<const float *>(rhs);
            *out = 1 - std::inner_product(x, x + dim - 1, q, 0.0f);
          }
        },
        {});
    const std::array<std::array<float, 3>, 3> queries{
        {{{1, 0, 100}}, {{0.6f, 0.8f, 2}}, {{0, 0, 0}}}};
    for (bool prebuilt : {true, false}) {
      if (prebuilt)
        ASSERT_EQ(0, index.prebuild(*entity, entity->doc_cnt(), 2));
      else
        index.clear();
      for (const auto &query : queries) {
        for (bool filtered : {false, true}) {
          ctx.dist_calculator().clear_compare_cnt();
          ctx.dist_calculator().reset_query(query.data());
          if (filtered)
            ctx.set_filter([](uint64_t id) { return id == 1; });
          else
            ctx.reset_filter();
          ASSERT_EQ(0, index.search(0, ctx));
          EXPECT_EQ(filtered ? 4U : 5U, ctx.search_heap().size());
          EXPECT_EQ(5U, ctx.dist_calculator().compare_cnt());
          ctx.search_heap().for_each([&](node_id_t id, dist_t distance) {
            if (filtered) EXPECT_NE(1U, id);
            const auto *x = static_cast<const float *>(entity->get_vector(id));
            EXPECT_NEAR(1 - query[0] * x[0] - query[1] * x[1], distance, 1e-6);
            return true;
          });
        }
      }
    }
  }
}

TEST(SymphonyQGTest, TurboFp32MetadataValidation) {
  if (!rabitqlib::cpu::has_avx2() && !rabitqlib::cpu::has_avx512_core())
    GTEST_SKIP();
  ailego::Params params;
  params.set(PARAM_HNSW_SYMPHONY_QG, true);
  for (const auto *metric : {"Cosine", "SquaredEuclidean"}) {
    for (uint32_t dimension : {1U, 4096U, 4097U}) {
      SCOPED_TRACE(testing::Message() << metric << ":" << dimension);
      IndexMeta raw(IndexMeta::DT_FP32, dimension);
      raw.set_metric(metric, 0, ailego::Params());
      auto quantizer = IndexFactory::CreateQuantizer("Fp32Quantizer");
      ASSERT_NE(nullptr, quantizer);
      ASSERT_EQ(0, quantizer->init(raw, ailego::Params()));
      auto meta = quantizer->meta();
      meta.set_quantizer("Fp32Quantizer", 0, ailego::Params());
      IndexStreamer::Pointer valid = std::make_shared<HnswStreamer>();
      EXPECT_EQ(dimension <= 4096 ? 0 : IndexError_Unsupported,
                valid->init(meta, params, quantizer));
      IndexStreamer::Pointer missing_quantizer =
          std::make_shared<HnswStreamer>();
      EXPECT_EQ(IndexError_Unsupported, missing_quantizer->init(meta, params));
      // Dropping or inventing a norm field must not cause cached-vector reads
      // to overrun storage, even when dimensions and names otherwise match.
      meta.set_extra_meta_size(meta.extra_meta_size() == 0 ? sizeof(float) : 0);
      IndexStreamer::Pointer invalid_layout = std::make_shared<HnswStreamer>();
      EXPECT_EQ(IndexError_Unsupported,
                invalid_layout->init(meta, params, quantizer));
    }
  }
  for (const auto *name : {"Fp16Quantizer", "Int8Quantizer"}) {
    IndexMeta raw(IndexMeta::DT_FP32, 128);
    raw.set_metric("SquaredEuclidean", 0, ailego::Params());
    auto quantizer = IndexFactory::CreateQuantizer(name);
    ASSERT_NE(nullptr, quantizer);
    ASSERT_EQ(0, quantizer->init(raw, ailego::Params()));
    auto meta = quantizer->meta();
    meta.set_quantizer(name, 0, ailego::Params());
    IndexStreamer::Pointer unsupported = std::make_shared<HnswStreamer>();
    EXPECT_EQ(IndexError_Unsupported,
              unsupported->init(meta, params, quantizer));
  }
}

TEST(SymphonyQGTest, CosineCodecMatchesL2ConversionIncludingZeroVectors) {
  if (!rabitqlib::cpu::has_avx2() && !rabitqlib::cpu::has_avx512_core())
    GTEST_SKIP();
  for (size_t dim : {1U, 128U, 4096U}) {
    SymphonyQGCodec l2(dim), cosine(dim, true);
    const size_t padded = l2.encoded_dim();
    constexpr size_t count = 5;
    for (bool zero_center : {false, true}) {
      std::vector<float> center(dim + 1, 0);
      center[0] = zero_center ? 0 : 1;
      center[dim] = 1000;  // original norm, excluded by transform()
      std::vector<float> rotated_center, vectors(padded * count), scratch;
      l2.transform(center.data(), rotated_center);
      std::array<float, count> norms{};
      for (size_t i = 0; i < count; ++i) {
        std::vector<float> vector(dim + 1, 0);
        if (i != 0) vector[(i - 1) % dim] = i % 2 == 0 ? -1 : 1;
        vector[dim] = 17 * i;
        norms[i] = i == 0 ? 0 : 1;
        l2.transform(vector.data(), scratch);
        std::copy(scratch.begin(), scratch.end(), vectors.begin() + i * padded);
      }
      std::vector<char> l2_codes(l2.batch_bytes()),
          cos_codes(cosine.batch_bytes());
      l2.encode(rotated_center.data(), vectors.data(), count, l2_codes.data());
      cosine.encode(rotated_center.data(), vectors.data(), count,
                    cos_codes.data());
      for (float q : {-1.0f, 0.0f, 1.0f}) {
        std::vector<float> query(dim + 1, 0);
        query[0] = q;
        query[dim] = 9999;
        SymphonyQGCodec::Query l2_query, cos_query;
        l2.prepare_query(query.data(), l2_query);
        cosine.prepare_query(query.data(), cos_query);
        std::array<float, SymphonyQGCodec::kBatchSize> l2_dist, cos_dist;
        l2.scan(l2_codes.data(), l2_query, (q - center[0]) * (q - center[0]),
                l2_dist.data());
        cosine.scan(cos_codes.data(), cos_query, 1 - q * center[0],
                    cos_dist.data());
        for (size_t i = 0; i < count; ++i) {
          EXPECT_TRUE(std::isfinite(cos_dist[i]));
          EXPECT_NEAR(1 + 0.5f * (l2_dist[i] - q * q - norms[i]), cos_dist[i],
                      1e-5);
        }
      }
    }
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
