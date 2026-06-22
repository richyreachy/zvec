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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/core/framework/index_stats.h>
#include "turbo/quantizer/quantizer.h"
#include "turbo/quantizer/rotator/rotator.h"

#if RABITQ_SUPPORTED
#include <rabitqlib/quantization/rabitq.hpp>
#endif

namespace zvec {
namespace turbo {

using namespace zvec::core;

/// RaBitQ quantizer implementing asymmetric distance estimation.
///
/// Data vectors are rotated via FHT (Fast Walsh-Hadamard Transform), then
/// quantized to multi-bit codes per dimension using the RaBitQ algorithm.
/// Distance estimation factors (f_add, f_rescale, f_error) are stored
/// alongside each quantized datapoint.
///
/// Query vectors are rotated (same FHT) and stored as fp32 with precomputed
/// sum and squared-norm for efficient asymmetric distance computation.
///
/// Quantized data layout:
///   [uint8 codes (padded_dim bytes)] [f_add (4B)] [f_rescale (4B)] [f_error
///   (4B)]
///
/// Quantized query layout:
///   [rotated fp32 (padded_dim * 4B)] [sum_query (4B)] [norm_sq_query (4B)]
///
/// Distance (L2):
///   est_dist = f_add + norm_sq_q + f_rescale * (ip(q, code) + cb * sum_q)
///
/// Distance (IP):
///   est_dist = f_add + f_rescale * (ip(q, code) + cb * sum_q)
///
class RabitqQuantizer : public Quantizer {
 public:
  RabitqQuantizer() {
    type_ = QuantizeType::kRabit;
  }

  virtual ~RabitqQuantizer() {}

 public:
  // ---- New Quantizer interface ----
  DataType input_data_type() const override {
    return DataType::kFp32;
  }

  int dim() const override {
    return static_cast<int>(original_dim_);
  }

  void train(const void *data, size_t num, size_t stride) override;

  bool require_train() const override {
    return true;
  }

  size_t quantized_datapoint_vector_length() const override {
    return padded_dim_;
  }

  size_t quantized_query_vector_length() const override {
    return padded_dim_;
  }

  void quantize_data(const void *input, void *output) const override;

  void quantize_query(const void *input, void *output) const override;

  float calc_distance_dp_query(const void *dp,
                               const void *query) const override;

  void calc_distance_dp_query_batch(const void *const *dp_list, int dp_num,
                                    const void *query,
                                    float *dist_list) const override;

  float calc_distance_dp_query_unquantized(const void *dp,
                                           const void *query) const override;

  void calc_distance_dp_query_batch_unquantized(
      const void *const *dp_list, int dp_num, const void *query,
      float *dist_list) const override;

  float calc_distance_dp_dp(const void *dp1, const void *dp2) const override;

  // ---- Legacy interface ----
  QuantizeType type() const override {
    return type_;
  }

  int init(const IndexMeta &meta, const ailego::Params &params) override;

  int train(IndexHolder::Pointer holder) override;

  const IndexMeta &meta(void) const override {
    return meta_;
  }

  int quantize(const void *query, const IndexQueryMeta &qmeta, std::string *out,
               IndexQueryMeta *ometa) const override;

  int dequantize(const void *in, const IndexQueryMeta &qmeta,
                 std::string *out) const override;

  int serialize(std::string *out) const override;

  int deserialize(std::string &in) override;

  int deserialize(const void *data, size_t len) override;

  DistanceImpl distance(const void *query,
                        const IndexQueryMeta &qmeta) const override;

  //! Number of total bits per dimension used by the RaBitQ encoding.
  uint32_t total_bits() const {
    return total_bits_;
  }

  //! The padded dimension (next power-of-two >= original_dim_).
  uint32_t padded_dim() const {
    return padded_dim_;
  }

 private:
  //! Compute the centering constant cb for the code interpretation.
  //! cb = -((1 << (total_bits - 1)) - 0.5)
  float cb() const {
    return -(static_cast<float>(1 << (total_bits_ - 1)) - 0.5f);
  }

  //! Compute the inner product between a fp32 vector and uint8 code vector.
  static float ip_float_code(const float *q, const uint8_t *code, size_t dim);

  //! Compute next power of 2 >= n.
  static uint32_t next_power_of_two(uint32_t n);

  // Configuration
  static constexpr uint32_t kDefaultTotalBits = 2;
  const std::string RABITQ_TOTAL_BITS_KEY = "rabitq_quantizer.total_bits";

  uint32_t total_bits_{kDefaultTotalBits};
  uint32_t original_dim_{0};
  uint32_t padded_dim_{0};

  MetricType metric_type_{MetricType::kSquaredEuclidean};
  IndexMeta original_meta_{};
  IndexMeta meta_{};
  IndexMeta::DataType data_type_{};

  //! FHT rotator for preprocessing (pads and rotates vectors).
  Rotator::Pointer rotator_{};

#if RABITQ_SUPPORTED
  //! RaBitQ faster config (precomputed scaling factors for multi-bit).
  rabitqlib::quant::RabitqConfig rabitq_config_{};
#endif
};

}  // namespace turbo
}  // namespace zvec
