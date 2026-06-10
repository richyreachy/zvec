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

#include <ailego/algorithm/integer_quantizer.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/core/framework/index_stats.h>
#include "quantizer/quantizer.h"
#include "quantizer/rotator/rotator.h"

namespace zvec {
namespace turbo {

using namespace zvec::core;

class Int8Quantizer : public Quantizer {
 public:
  Int8Quantizer() {
    type_ = QuantizeType::kRecordInt8;
  }

  virtual ~Int8Quantizer() {}

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
    return quantized_length();
  }

  size_t quantized_query_vector_length() const override {
    return quantized_length();
  }

  void quantize_data(const void *input, void *output) const override {
    quantize_one(input, output);
  }

  void quantize_query(const void *input, void *output) const override {
    quantize_one(input, output);
  }

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

  // ---- Retained legacy helpers ----
  QuantizeType type() const {
    return type_;
  }

  int init(const core::IndexMeta &meta, const ailego::Params &params);

  int train(core::IndexHolder::Pointer holder);

  const core::IndexMeta &meta(void) const {
    return meta_;
  }

  int quantize(const void *query, const core::IndexQueryMeta &qmeta,
               std::string *out, core::IndexQueryMeta *ometa) const;

  int dequantize(const void *in, const core::IndexQueryMeta &qmeta,
                 std::string *out) const;

  //! Attach a rotation preprocessing stage. The rotator must be dimension
  //! preserving and match the quantizer dimension
  //! (in_dim == out_dim == dim()); otherwise IndexError_InvalidArgument is
  //! returned and the rotator is not attached.
  int set_rotator(Rotator::Pointer r);

  int serialize(std::string *out) const;

  int deserialize(std::string &in);

  int deserialize(const void *data, size_t len);

  DistanceImpl distance(const void *query,
                        const core::IndexQueryMeta &qmeta) const;

  float bias() const {
    return bias_;
  }
  float scale() const {
    return scale_;
  }

 private:
  //! Byte length of a quantized vector (data + per-vector extras).
  size_t quantized_length() const {
    return static_cast<size_t>(original_dim_) + extra_meta_size_;
  }

  //! Quantize a single fp32 vector into a caller-provided buffer of
  //! quantized_length() bytes.
  void quantize_one(const void *input, void *output) const;
  static constexpr uint32_t EXTRA_META_SIZE_INT8 = 20;
  static constexpr uint32_t EXTRA_META_SIZE_COSINE = 4;
  const std::string INT8_QUANTIZER_BIAS = "int8_quantizer.bias";
  const std::string INT8_QUANTIZER_SCALE = "int8_quantizer.scale";

  //! Optional rotation applied to fp32 vectors before quantization (and
  //! inverted on dequantize). Null means no rotation.
  Rotator::Pointer rotator_{};

  mutable float bias_{0.0f};
  mutable float scale_{1.0f};
  float scale_reciprocal_{1.0f};
  bool inner_product_{false};
  bool cosine_{false};
  bool record_quantize_{false};
  MetricType origin_metric_{MetricType::kUnknown};

  mutable ailego::EntropyInt8Quantizer quantizer_;
  IndexMeta meta_{};
  uint32_t original_dim_{0};
  IndexMeta::DataType data_type_{};

  //! Cached distance dispatch (bound in init()).
  MetricType dist_metric_{MetricType::kUnknown};
  DistanceFunc dp_query_func_{};
  BatchDistanceFunc dp_query_batch_func_{};
};


}  // namespace turbo
}  // namespace zvec
