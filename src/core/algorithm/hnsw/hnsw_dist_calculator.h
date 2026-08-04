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
#pragma once

#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_metric.h>
#include "hnsw_entity.h"

namespace zvec {
namespace core {

//! Dist calculator used by HNSW. Prefers the turbo Quantizer's
//! distance kernels when they are available for the current
//! metric/dtype; otherwise falls back to IndexMetric's distance /
//! batch_distance handles. This keeps HNSW functional for metric/dtype
//! combos that turbo does not yet implement (e.g. MipsSquaredEuclidean,
//! Cosine with cached norm, non-FP32 converter pipelines).
//!
//! Two query modes are supported:
//!  - search (`reset_query`): the query is a quantized query vector,
//!    distances go through `calc_distance_dp_query[_batch]`.
//!  - add (`reset_add_query`): the "query" is the quantized datapoint
//!    being inserted, distances go through `calc_distance_dp_dp[_batch]`.
//! Node-vs-node pairwise distances always use `calc_distance_dp_dp`.
class HnswDistCalculator {
 public:
  typedef std::shared_ptr<HnswDistCalculator> Pointer;

 public:
  enum DistType {
    DIST_NONE = 0,
    DIST_DENSE = 1,
    DIST_HYBRID = 2,
    DIST_SPARSE = 3
  };

 public:
  //! Constructor with a turbo quantizer and an IndexMetric fallback.
  //! `dim` is the dimension of the stored vectors. `qmeta_data_type`
  //! is the data type of the raw query accepted by `reset_query`.
  HnswDistCalculator(const HnswEntity *entity,
                     zvec::turbo::Quantizer::Pointer quantizer,
                     IndexMetric::Pointer metric, uint32_t dim,
                     IndexMeta::DataType qmeta_data_type)
      : entity_(entity),
        quantizer_(std::move(quantizer)),
        metric_(std::move(metric)),
        query_(nullptr),
        dim_(dim),
        compare_cnt_(0) {
    qmeta_.set_meta(qmeta_data_type, dim);
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    }
  }

  //! Constructor without dimension (for lazy init via update()).
  HnswDistCalculator(const HnswEntity *entity,
                     zvec::turbo::Quantizer::Pointer quantizer,
                     IndexMetric::Pointer metric)
      : entity_(entity),
        quantizer_(std::move(quantizer)),
        metric_(std::move(metric)),
        query_(nullptr),
        dim_(0),
        compare_cnt_(0) {
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    }
  }

  void update(const HnswEntity *entity,
              zvec::turbo::Quantizer::Pointer quantizer,
              IndexMetric::Pointer metric) {
    entity_ = entity;
    quantizer_ = std::move(quantizer);
    metric_ = std::move(metric);
    use_turbo_ = false;
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    } else {
      distance_ = nullptr;
      batch_distance_ = nullptr;
    }
  }

  void update(const HnswEntity *entity,
              zvec::turbo::Quantizer::Pointer quantizer,
              IndexMetric::Pointer metric, uint32_t dim,
              IndexMeta::DataType qmeta_data_type) {
    entity_ = entity;
    quantizer_ = std::move(quantizer);
    metric_ = std::move(metric);
    dim_ = dim;
    qmeta_.set_meta(qmeta_data_type, dim);
    use_turbo_ = false;
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    } else {
      distance_ = nullptr;
      batch_distance_ = nullptr;
    }
  }

  //! Replace the quantizer used by this calculator. Invalidates the
  //! bound turbo path; caller should follow up with reset_query or
  //! reset_add_query.
  inline void update_quantizer(zvec::turbo::Quantizer::Pointer quantizer) {
    quantizer_ = std::move(quantizer);
    use_turbo_ = false;
  }

  //! Replace the IndexMetric fallback.
  inline void update_metric(IndexMetric::Pointer metric) {
    metric_ = std::move(metric);
    if (metric_) {
      distance_ = metric_->distance();
      batch_distance_ = metric_->batch_distance();
    } else {
      distance_ = nullptr;
      batch_distance_ = nullptr;
    }
  }

  //! Reset query vector data for search. The query is expected to be
  //! already in the quantized query layout; subsequent `dist(...)` calls
  //! go through the quantizer's `calc_distance_dp_query[_batch]`. Falls
  //! back to IndexMetric's raw query when turbo does not support this
  //! metric/dtype combination.
  inline void reset_query(const void *query) {
    error_ = false;
    query_ = query;
    query_is_dp_ = false;
    use_turbo_ = quantizer_ && quantizer_->distance_available();
  }

  //! Reset query data for add. `dp` is the quantized datapoint being
  //! inserted into the graph; subsequent `dist(...)` calls go through
  //! the quantizer's `calc_distance_dp_dp[_batch]`.
  inline void reset_add_query(const void *dp) {
    error_ = false;
    query_ = dp;
    query_is_dp_ = true;
    use_turbo_ = quantizer_ && quantizer_->distance_available();
  }

  //! Returns distance between two already-quantized datapoints
  //! (node-vs-node pairwise). Always uses `calc_distance_dp_dp` when the
  //! turbo path is available; otherwise falls back to IndexMetric.
  inline dist_t dist(const void *vec_lhs, const void *vec_rhs) {
    if (ailego_unlikely(vec_lhs == nullptr || vec_rhs == nullptr)) {
      LOG_ERROR("Nullptr of dense vector");
      error_ = true;
      return 0.0f;
    }

    if (use_turbo_) {
      return quantizer_->calc_distance_dp_dp(vec_lhs, vec_rhs);
    }
    if (ailego_unlikely(!distance_)) {
      LOG_ERROR("No distance handle available");
      error_ = true;
      return 0.0f;
    }
    float score = 0.0f;
    distance_(vec_lhs, vec_rhs, dim_, &score);
    return score;
  }

  //! Returns distance between query and vec.
  inline dist_t dist(const void *vec) {
    compare_cnt_++;
    if (ailego_unlikely(vec == nullptr)) {
      LOG_ERROR("Nullptr of dense vector");
      error_ = true;
      return 0.0f;
    }
    if (use_turbo_ && query_ != nullptr) {
      return query_is_dp_ ? quantizer_->calc_distance_dp_dp(vec, query_)
                          : quantizer_->calc_distance_dp_query(vec, query_);
    }
    if (ailego_unlikely(!distance_ || query_ == nullptr)) {
      LOG_ERROR("No distance handle or query available");
      error_ = true;
      return 0.0f;
    }
    float score = 0.0f;
    distance_(vec, query_, dim_, &score);
    return score;
  }

  //! Return distance between query and node id.
  inline dist_t dist(node_id_t id) {
    compare_cnt_++;
    IndexStorage::MemoryBlock vec_block;
    int ret = entity_->get_vector(id, vec_block);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }
    const void *feat = vec_block.data();
    if (ailego_unlikely(feat == nullptr)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }
    if (use_turbo_ && query_ != nullptr) {
      return query_is_dp_ ? quantizer_->calc_distance_dp_dp(feat, query_)
                          : quantizer_->calc_distance_dp_query(feat, query_);
    }
    if (ailego_unlikely(!distance_ || query_ == nullptr)) {
      LOG_ERROR("No distance handle or query available");
      error_ = true;
      return 0.0f;
    }
    float score = 0.0f;
    distance_(feat, query_, dim_, &score);
    return score;
  }

  //! Return dist node lhs between node rhs
  inline dist_t dist(node_id_t lhs, node_id_t rhs) {
    compare_cnt_++;

    IndexStorage::MemoryBlock vec_block_feat;
    int ret = entity_->get_vector(lhs, vec_block_feat);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Get nullptr vector, id=%u", lhs);
      error_ = true;
      return 0.0f;
    }
    const void *feat = vec_block_feat.data();

    IndexStorage::MemoryBlock vec_block_query;
    ret = entity_->get_vector(rhs, vec_block_query);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Get nullptr vector, id=%u", rhs);
      error_ = true;
      return 0.0f;
    }
    const void *query = vec_block_query.data();
    if (ailego_unlikely(feat == nullptr || query == nullptr)) {
      LOG_ERROR("Get nullptr vector");
      error_ = true;
      return 0.0f;
    }

    return dist(feat, query);
  }

  dist_t operator()(const void *vec) {
    return dist(vec);
  }

  dist_t operator()(node_id_t i) {
    return dist(i);
  }

  dist_t operator()(node_id_t lhs, node_id_t rhs) {
    return dist(lhs, rhs);
  }

  void batch_dist(const void **vecs, size_t num, dist_t *distances) {
    compare_cnt_++;
    if (use_turbo_ && query_ != nullptr) {
      if (query_is_dp_) {
        quantizer_->calc_distance_dp_dp_batch(vecs, static_cast<int>(num),
                                              query_, distances);
      } else {
        quantizer_->calc_distance_dp_query_batch(vecs, static_cast<int>(num),
                                                 query_, distances);
      }
      return;
    }
    if (batch_distance_ && query_ != nullptr) {
      batch_distance_(vecs, query_, num, dim_, distances);
      return;
    }
    // Last-resort scalar fallback using whatever single-distance path
    // is available.
    for (size_t i = 0; i < num; ++i) {
      distances[i] = dist(vecs[i]);
    }
  }

  inline dist_t batch_dist(node_id_t id) {
    compare_cnt_++;

    IndexStorage::MemoryBlock vec_block;
    int ret = entity_->get_vector(id, vec_block);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }
    const void *feat = vec_block.data();
    if (ailego_unlikely(feat == nullptr)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }
    if (use_turbo_ && query_ != nullptr) {
      return query_is_dp_ ? quantizer_->calc_distance_dp_dp(feat, query_)
                          : quantizer_->calc_distance_dp_query(feat, query_);
    }
    if (batch_distance_ && query_ != nullptr) {
      dist_t score = 0;
      const void *feats[1] = {feat};
      batch_distance_(feats, query_, 1, dim_, &score);
      return score;
    }
    return dist(feat);
  }

  inline void clear() {
    compare_cnt_ = 0;
    error_ = false;
  }

  inline void clear_compare_cnt() {
    compare_cnt_ = 0;
  }

  inline bool error() const {
    return error_;
  }

  //! Get distances compute times
  inline uint32_t compare_cnt() const {
    return compare_cnt_;
  }

  inline uint32_t dimension() const {
    return dim_;
  }

  //! Expose the underlying turbo quantizer (for clients that need to
  //! reach lower-level turbo APIs).
  inline const zvec::turbo::Quantizer::Pointer &quantizer() const {
    return quantizer_;
  }

 private:
  HnswDistCalculator(const HnswDistCalculator &) = delete;
  HnswDistCalculator &operator=(const HnswDistCalculator &) = delete;

 private:
  const HnswEntity *entity_;

  zvec::turbo::Quantizer::Pointer quantizer_{};
  IndexMetric::Pointer metric_{};
  IndexQueryMeta qmeta_{};

  IndexMetric::MatrixDistance distance_{nullptr};
  IndexMetric::MatrixBatchDistance batch_distance_{nullptr};

  const void *query_;
  uint32_t dim_;

  //! Whether the turbo quantizer distance kernels are usable.
  bool use_turbo_{false};
  //! Whether query_ is a quantized datapoint (add path, dp-dp distance)
  //! rather than a quantized query (search path, dp-query distance).
  bool query_is_dp_{false};

  uint32_t compare_cnt_;  // record distance compute times
  bool error_{false};
};

}  // namespace core
}  // namespace zvec
