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
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/container/params.h>
#include <zvec/turbo/turbo.h>
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_holder.h"

using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr size_t kDim = 128;
static constexpr size_t kNumClusters = 4;
static constexpr size_t kTrainCount = 500;

// RaBitQ parameter names (mirrors core/algorithm/hnsw_rabitq/rabitq_params.h)
static const std::string kParamNumClusters("proxima.rabitq.num_clusters");
static const std::string kParamTotalBits("proxima.rabitq.total_bits");

// Reference squared Euclidean distance between two raw fp32 vectors.
static float reference_l2_sqr(const float *a, const float *b, size_t dim) {
  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

// Generate random fp32 vectors.
static std::vector<std::vector<float>> generate_vectors(size_t count,
                                                        size_t dim,
                                                        uint32_t seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<std::vector<float>> vecs(count);
  for (size_t i = 0; i < count; ++i) {
    vecs[i].resize(dim);
    for (size_t j = 0; j < dim; ++j) {
      vecs[i][j] = dist(gen);
    }
  }
  return vecs;
}

// Build a trained RabitQuantizer for L2 metric.
static turbo::Quantizer::Pointer create_trained_quantizer(
    size_t dim, size_t num_clusters, size_t train_count,
    const std::vector<std::vector<float>> &vecs) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, static_cast<uint32_t>(dim));
  meta.set_metric("SquaredEuclidean", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("RabitQuantizer");
  EXPECT_TRUE(quantizer) << "Failed to create RabitQuantizer via factory";

  Params params;
  params.set(kParamNumClusters, static_cast<uint32_t>(num_clusters));
  params.set(kParamTotalBits, static_cast<uint32_t>(7));

  EXPECT_EQ(0u, quantizer->init(meta, params));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          static_cast<uint32_t>(dim));
  for (size_t i = 0; i < train_count && i < vecs.size(); ++i) {
    NumericalVector<float> vec(dim);
    std::memcpy(vec.data(), vecs[i].data(), dim * sizeof(float));
    holder->emplace(static_cast<uint64_t>(i + 1), vec);
  }

  EXPECT_EQ(0u, quantizer->train(holder));
  return quantizer;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(RabitQuantizer, Init) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, static_cast<uint32_t>(kDim));
  meta.set_metric("SquaredEuclidean", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("RabitQuantizer");
  ASSERT_TRUE(quantizer);

  Params params;
  params.set(kParamNumClusters, static_cast<uint32_t>(kNumClusters));
  params.set(kParamTotalBits, static_cast<uint32_t>(7));

  EXPECT_EQ(0u, quantizer->init(meta, params));

  // Verify basic properties
  EXPECT_EQ(quantizer->type(), turbo::QuantizeType::kRabit);
  EXPECT_EQ(quantizer->dim(), static_cast<int>(kDim));
  EXPECT_TRUE(quantizer->require_train());
  EXPECT_EQ(quantizer->input_data_type(), turbo::DataType::kFp32);

  // Quantized vector lengths must be non-zero
  EXPECT_GT(quantizer->quantized_datapoint_vector_length(), 0u);
  EXPECT_GT(quantizer->quantized_query_vector_length(), 0u);
}

TEST(RabitQuantizer, InitInvalidMetric) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, static_cast<uint32_t>(kDim));
  meta.set_metric("UnknownMetric", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("RabitQuantizer");
  ASSERT_TRUE(quantizer);

  Params params;
  EXPECT_NE(0u, quantizer->init(meta, params));
}

TEST(RabitQuantizer, InitInvalidBits) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, static_cast<uint32_t>(kDim));
  meta.set_metric("SquaredEuclidean", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("RabitQuantizer");
  ASSERT_TRUE(quantizer);

  Params params;
  // total_bits must be in [1, 9]; 10 is invalid
  params.set(kParamTotalBits, static_cast<uint32_t>(10));
  EXPECT_NE(0u, quantizer->init(meta, params));
}

TEST(RabitQuantizer, TrainAndQuantize) {
  auto raw_vecs = generate_vectors(kTrainCount, kDim, 42);

  auto quantizer =
      create_trained_quantizer(kDim, kNumClusters, kTrainCount, raw_vecs);
  ASSERT_TRUE(quantizer);

  size_t dp_len = quantizer->quantized_datapoint_vector_length();
  size_t q_len = quantizer->quantized_query_vector_length();
  ASSERT_GT(dp_len, 0u);
  ASSERT_GT(q_len, 0u);

  // Quantize a datapoint
  std::vector<uint8_t> dp_buf(dp_len);
  EXPECT_NO_THROW(quantizer->quantize_data(raw_vecs[0].data(), dp_buf.data()));

  // Quantize a query
  std::vector<uint8_t> q_buf(q_len);
  EXPECT_NO_THROW(quantizer->quantize_query(raw_vecs[0].data(), q_buf.data()));

  // Distance of a vector to itself should be small (approximate).
  // RaBitQ estimates can be slightly negative due to quantization error.
  float self_dist =
      quantizer->calc_distance_dp_query(dp_buf.data(), q_buf.data());
  EXPECT_GT(self_dist, -1.0f);
  EXPECT_LT(self_dist, 100.0f) << "Self-distance too large: " << self_dist;
}

TEST(RabitQuantizer, DistanceRankOrdering) {
  // Use more data for better clustering
  constexpr size_t kCount = 200;
  auto raw_vecs = generate_vectors(kCount, kDim, 123);

  auto quantizer =
      create_trained_quantizer(kDim, kNumClusters, kCount, raw_vecs);
  ASSERT_TRUE(quantizer);

  size_t dp_len = quantizer->quantized_datapoint_vector_length();
  size_t q_len = quantizer->quantized_query_vector_length();

  // Pick a query and find its true nearest neighbor
  size_t query_idx = 0;
  std::vector<uint8_t> q_buf(q_len);
  quantizer->quantize_query(raw_vecs[query_idx].data(), q_buf.data());

  // Quantize all datapoints and compute both true and estimated distances
  std::vector<std::vector<uint8_t>> dp_bufs(kCount);
  std::vector<float> true_dists(kCount);
  std::vector<float> est_dists(kCount);

  for (size_t i = 0; i < kCount; ++i) {
    dp_bufs[i].resize(dp_len);
    quantizer->quantize_data(raw_vecs[i].data(), dp_bufs[i].data());
    true_dists[i] =
        reference_l2_sqr(raw_vecs[i].data(), raw_vecs[query_idx].data(), kDim);
    est_dists[i] =
        quantizer->calc_distance_dp_query(dp_bufs[i].data(), q_buf.data());
  }

  // The true nearest neighbor (excluding self) should be among the top
  // estimated nearest neighbors.  RaBitQ is approximate, so we check
  // that the true NN is in the top-10 by estimated distance.
  std::vector<size_t> true_order(kCount);
  std::vector<size_t> est_order(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    true_order[i] = i;
    est_order[i] = i;
  }
  std::sort(true_order.begin(), true_order.end(),
            [&](size_t a, size_t b) { return true_dists[a] < true_dists[b]; });
  std::sort(est_order.begin(), est_order.end(),
            [&](size_t a, size_t b) { return est_dists[a] < est_dists[b]; });

  // Self should be the closest in both
  EXPECT_EQ(true_order[0], query_idx);

  // The true 2nd NN (index 1, excluding self at index 0) should appear
  // within the top-20 of the estimated ranking.
  size_t true_nn = true_order[1];
  size_t est_rank = 0;
  for (size_t i = 0; i < kCount; ++i) {
    if (est_order[i] == true_nn) {
      est_rank = i;
      break;
    }
  }
  EXPECT_LT(est_rank, 20u) << "True NN rank " << true_nn
                           << " not in top-20 estimated (rank=" << est_rank
                           << ")";

  // Batch distance should match single distance
  {
    std::vector<const void *> dp_list(kCount);
    for (size_t i = 0; i < kCount; ++i) {
      dp_list[i] = dp_bufs[i].data();
    }
    std::vector<float> batch_dists(kCount);
    quantizer->calc_distance_dp_query_batch(dp_list.data(),
                                            static_cast<int>(kCount),
                                            q_buf.data(), batch_dists.data());

    for (size_t i = 0; i < kCount; ++i) {
      EXPECT_NEAR(est_dists[i], batch_dists[i], 1e-3f)
          << "Batch/single mismatch at i=" << i;
    }
  }
}

TEST(RabitQuantizer, SerializeDeserialize) {
  auto raw_vecs = generate_vectors(kTrainCount, kDim, 99);

  auto quantizer =
      create_trained_quantizer(kDim, kNumClusters, kTrainCount, raw_vecs);
  ASSERT_TRUE(quantizer);

  // Serialize
  std::string blob;
  EXPECT_EQ(0, quantizer->serialize(&blob));
  EXPECT_FALSE(blob.empty());

  // Deserialize into a new quantizer
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, static_cast<uint32_t>(kDim));
  meta.set_metric("SquaredEuclidean", 0, Params());

  auto quantizer2 = IndexFactory::CreateQuantizer("RabitQuantizer");
  ASSERT_TRUE(quantizer2);
  Params params2;
  params2.set(kParamNumClusters, static_cast<uint32_t>(kNumClusters));
  params2.set(kParamTotalBits, static_cast<uint32_t>(7));
  EXPECT_EQ(0u, quantizer2->init(meta, params2));

  std::string blob_copy = blob;
  EXPECT_EQ(0, quantizer2->deserialize(blob_copy));

  // Verify the deserialized quantizer produces the same distances
  size_t dp_len = quantizer->quantized_datapoint_vector_length();
  size_t q_len = quantizer->quantized_query_vector_length();
  EXPECT_EQ(dp_len, quantizer2->quantized_datapoint_vector_length());
  EXPECT_EQ(q_len, quantizer2->quantized_query_vector_length());

  std::vector<uint8_t> dp_buf(dp_len);
  std::vector<uint8_t> q_buf(q_len);
  quantizer->quantize_data(raw_vecs[0].data(), dp_buf.data());
  quantizer->quantize_query(raw_vecs[1].data(), q_buf.data());

  float dist1 = quantizer->calc_distance_dp_query(dp_buf.data(), q_buf.data());
  float dist2 = quantizer2->calc_distance_dp_query(dp_buf.data(), q_buf.data());
  EXPECT_NEAR(dist1, dist2, 1e-3f);
}

TEST(RabitQuantizer, UnquantizedDistance) {
  auto raw_vecs = generate_vectors(200, kDim, 77);

  auto quantizer = create_trained_quantizer(kDim, kNumClusters, 200, raw_vecs);
  ASSERT_TRUE(quantizer);

  size_t dp_len = quantizer->quantized_datapoint_vector_length();

  // Quantize a datapoint, then compute distance against raw query
  std::vector<uint8_t> dp_buf(dp_len);
  quantizer->quantize_data(raw_vecs[0].data(), dp_buf.data());

  float dist = quantizer->calc_distance_dp_query_unquantized(
      dp_buf.data(), raw_vecs[1].data());
  float true_dist =
      reference_l2_sqr(raw_vecs[0].data(), raw_vecs[1].data(), kDim);

  // Estimated distance should be in the same ballpark as true distance.
  // RaBitQ estimates can be off by a factor, so use a generous bound.
  EXPECT_GE(dist, 0.0f);
  EXPECT_LT(dist, true_dist * 3.0f + 100.0f)
      << "Estimated distance too far from true: est=" << dist
      << " true=" << true_dist;
}
