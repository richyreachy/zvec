// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <vector>
#include "preprocessor/fht_rotator/fht_rotator.h"
#include "quantizer/quantizer.h"
#include "rabitq_params.h"

namespace zvec::turbo {

// Zero-centered RaBitQ with an intrinsic random rotation. Uses the bundled
// RaBitQ library's multi-bit encoder; codes are densely packed in bit order.
// Records: [norm squared, unbiased scale, original norm][packed codes].
// Queries: [rotated FP32 coordinates][norm squared]. No training is required.
class RabitqQuantizer final : public Quantizer {
 public:
  RabitqQuantizer() : Quantizer(QuantizeType::kRabit) {}
  int init(const IndexMeta &, const ailego::Params &) override;
  const IndexMeta &meta() const override {
    return meta_;
  }
  DataType input_data_type() const override {
    return DataType::kFp32;
  }
  int dim() const override {
    return dim_;
  }
  bool require_train() const override {
    return false;
  }
  bool requires_original_vectors() const override {
    return true;
  }
  size_t quantized_datapoint_vector_length() const override {
    return 3 * sizeof(float) + (static_cast<size_t>(dim_) * bits_ + 7) / 8;
  }
  size_t quantized_query_vector_length() const override {
    return (static_cast<size_t>(dim_) + 1) * sizeof(float);
  }
  IndexQueryMeta quantized_query_meta() const override {
    IndexQueryMeta result;
    result.set_meta(IndexMeta::DT_FP32, dim_, static_cast<uint32_t>(type_),
                    sizeof(float));
    return result;
  }
  void quantize_data(const void *, void *) const override;
  void quantize_query(const void *, void *) const override;
  float calc_distance_dp_query(const void *, const void *) const override;
  void calc_distance_dp_query_batch(const void *const *, int, const void *,
                                    float *) const override;
  float calc_distance_dp_query_unquantized(const void *,
                                           const void *) const override;
  void calc_distance_dp_query_batch_unquantized(const void *const *, int,
                                                const void *,
                                                float *) const override;
  float calc_distance_dp_dp(const void *, const void *) const override;
  int quantize(const void *, const IndexQueryMeta &, std::string *,
               IndexQueryMeta *) const override;
  int quantize_datapoint(const void *, const IndexQueryMeta &, std::string *,
                         IndexQueryMeta *) const override;
  int dequantize(const void *, const IndexQueryMeta &,
                 std::string *) const override;
  DistanceImpl distance(const void *, const IndexQueryMeta &) const override;
  bool support_score_normalization() const override {
    return metric_ == MetricType::kInnerProduct;
  }
  void normalize_score(float *score) const override {
    *score = -*score;
  }
  void denormalize_score(float *score) const override {
    *score = -*score;
  }
  int serialize(std::string *) const override;
  int deserialize(std::string &in) override {
    return deserialize(in.data(), in.size());
  }
  int deserialize(const void *, size_t) override;

 private:
  bool valid_input(const IndexQueryMeta &) const;
  float rotate(const void *, std::vector<float> *) const;
  unsigned code(const void *, int) const;
  IndexMeta meta_;
  int dim_{0};
  int bits_{7};
  MetricType metric_{MetricType::kUnknown};
  double rescale_{-1};
  FhtRotator::Pointer rotator_;
};

}  // namespace zvec::turbo
