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
//
// TurboQuant metric: SDC (Symmetric Distance Computation) on packed
// quantized indices.  Uses precomputed centroid tables for the MSE part
// and an approximate QJL inner product for the Prod part.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <core/quantizer/turbo_quant_engine.h>
#include <core/quantizer/turbo_quant_params.h>
#include <zvec/core/framework/index_factory.h>
#include "metric_params.h"

namespace zvec {
namespace core {

/*! Index Metric for TurboQuant
 *
 * Uses SDC (Symmetric Distance Computation): both query and database
 * vectors are in the TurboQuant packed format.  The L2 distance and
 * inner product are computed using precomputed centroid tables.
 *
 * For MSE mode:
 *   L2:  sum_j (c_{idx_a[j]} - c_{idx_b[j]})^2
 *   IP:  sum_j c_{idx_a[j]} * c_{idx_b[j]}
 *
 * For Prod mode, the QJL contribution is approximated using
 * S·S^T ≈ d·I (valid for large d).
 */
class TurboQuantMetric : public IndexMetric {
 public:
  enum class OriginMetric {
    kSquaredEuclidean,
    kInnerProduct,
    kCosine,
  };

  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    if (meta.data_type() != IndexMeta::DataType::DT_INT8) {
      LOG_ERROR("TurboQuantMetric: unsupported type %d", meta.data_type());
      return IndexError_Unsupported;
    }

    int bit_width = 4;
    index_params.get(TURBO_QUANT_REFORMER_BIT_WIDTH, &bit_width);

    std::string mode = TURBO_QUANT_MODE_MSE;
    index_params.get(TURBO_QUANT_REFORMER_MODE, &mode);
    bool prod_mode = (mode == TURBO_QUANT_MODE_PROD);

    bool enable_rotate = true;
    index_params.get(TURBO_QUANT_REFORMER_ENABLE_ROTATE, &enable_rotate);

    int64_t seed_val = 42;
    index_params.get(TURBO_QUANT_REFORMER_SEED, &seed_val);

    int64_t dim_val = 0;
    index_params.get(TURBO_QUANT_REFORMER_DIMENSION, &dim_val);
    if (dim_val <= 0) {
      LOG_ERROR("TurboQuantMetric: missing dimension param");
      return IndexError_InvalidArgument;
    }
    original_dimension_ = static_cast<size_t>(dim_val);

    std::string origin_metric_name;
    index_params.get("turbo_quant.metric.origin_metric_name",
                     &origin_metric_name);
    if (origin_metric_name == "SquaredEuclidean" ||
        origin_metric_name == "Euclidean") {
      origin_metric_ = OriginMetric::kSquaredEuclidean;
    } else if (origin_metric_name == "InnerProduct") {
      origin_metric_ = OriginMetric::kInnerProduct;
    } else if (origin_metric_name == "Cosine" ||
               origin_metric_name == "NormalizedCosine") {
      origin_metric_ = OriginMetric::kCosine;
    } else if (origin_metric_name.empty()) {
      origin_metric_ = OriginMetric::kSquaredEuclidean;
    } else {
      LOG_ERROR("TurboQuantMetric: unsupported origin metric %s",
                origin_metric_name.c_str());
      return IndexError_Unsupported;
    }

    // Create engine without rotation (SDC doesn't need the rotator)
    engine_ = std::make_shared<TurboQuantEngine>(
        original_dimension_, bit_width, prod_mode, false,
        static_cast<uint64_t>(seed_val));

    meta_ = meta;
    params_ = index_params;

    LOG_INFO(
        "TurboQuantMetric initialized: dim=%zu, bits=%d, mode=%s, "
        "origin=%s",
        original_dimension_, bit_width, prod_mode ? "prod" : "mse",
        origin_metric_name.c_str());
    return 0;
  }

  int cleanup(void) override {
    return 0;
  }

  bool is_matched(const IndexMeta &meta) const override {
    return meta.data_type() == meta_.data_type() &&
           meta.unit_size() == meta_.unit_size();
  }

  bool is_matched(const IndexMeta &meta,
                  const IndexQueryMeta &qmeta) const override {
    return qmeta.data_type() == meta_.data_type() &&
           qmeta.unit_size() == meta_.unit_size() &&
           qmeta.dimension() == meta.dimension();
  }

  MatrixDistance distance(void) const override {
    return [this](const void *m, const void *q, size_t /*dim*/, float *out) {
      const uint8_t *a = reinterpret_cast<const uint8_t *>(m);
      const uint8_t *b = reinterpret_cast<const uint8_t *>(q);
      switch (origin_metric_) {
        case OriginMetric::kSquaredEuclidean:
          *out = engine_->sdc_squared_l2(a, b);
          break;
        case OriginMetric::kInnerProduct:
        case OriginMetric::kCosine:
          *out = engine_->sdc_inner_product(a, b);
          break;
      }
    };
  }

  MatrixDistance distance_matrix(size_t /*m*/, size_t /*n*/) const override {
    return distance();
  }

  const ailego::Params &params(void) const override {
    return params_;
  }

  int train(const void * /*vec*/, size_t /*dim*/) override {
    return 0;
  }
  bool support_train(void) const override {
    return false;
  }

  void normalize(float * /*score*/) const override {}
  bool support_normalize(void) const override {
    return false;
  }

  Pointer query_metric(void) const override {
    return nullptr;
  }

  DistanceBatchQueryPreprocessFunc get_query_preprocess_func() const override {
    return nullptr;
  }

 private:
  IndexMeta meta_{};
  ailego::Params params_{};
  size_t original_dimension_{0};
  OriginMetric origin_metric_{OriginMetric::kSquaredEuclidean};
  std::shared_ptr<TurboQuantEngine> engine_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(TurboQuant, TurboQuantMetric);

}  // namespace core
}  // namespace zvec
