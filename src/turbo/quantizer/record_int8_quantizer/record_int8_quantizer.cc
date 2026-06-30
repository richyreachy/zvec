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

#include "quantizer/record_int8_quantizer/record_int8_quantizer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>
#include "core/quantizer/record_quantizer.h"

namespace zvec {
namespace turbo {

int RecordInt8Quantizer::init(const core::IndexMeta &meta,
                              const ailego::Params & /*params*/) {
  if (meta.data_type() != core::IndexMeta::DataType::DT_FP32 ||
      meta.unit_size() !=
          core::IndexMeta::UnitSizeof(core::IndexMeta::DataType::DT_FP32)) {
    LOG_ERROR("Unsupported type %d with unit size %u", meta.data_type(),
              meta.unit_size());
    return core::IndexError_Unsupported;
  }

  meta_ = meta;
  original_dim_ = meta.dimension();
  data_type_ = core::IndexMeta::DataType::DT_INT8;

  extra_meta_size_ = EXTRA_META_SIZE_INT8;
  if (meta.metric_name() == "Cosine") {
    cosine_ = true;
    extra_meta_size_ += EXTRA_META_SIZE_COSINE;
  }

  origin_metric_ = metric_from_name(meta.metric_name());

  // Inflate dimension by extra bytes (INT8 unit=1) so meta_.element_size()
  // reflects the real per-vector storage (data + extras).
  meta_.set_meta(data_type_, original_dim_ + extra_meta_size_);
  meta_.set_extra_meta_size(extra_meta_size_);

  ailego::Params metric_params;
  metric_params.set("proxima.quantized_integer.metric.origin_metric_name",
                    meta.metric_name());
  metric_params.set("proxima.quantized_integer.metric.origin_metric_params",
                    meta.metric_params());
  meta_.set_metric("QuantizedInteger", 0, metric_params);

  // Cache the distance dispatch for the new Quantizer interface.
  dp_query_func_ =
      get_distance_func(origin_metric_, DataType::kInt8, QuantizeType::kDefault,
                        CpuArchType::kAuto);
  dp_query_batch_func_ =
      get_batch_distance_func(origin_metric_, DataType::kInt8,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  return 0;
}

int RecordInt8Quantizer::quantize(const void *record,
                                  const core::IndexQueryMeta & /*rmeta*/,
                                  std::string *out,
                                  core::IndexQueryMeta *ometa) const {
  out->resize(quantized_length(), 0);
  quantize_one(record, &(*out)[0]);

  *ometa = core::IndexQueryMeta();
  // Match meta_ dimension (data + extras) using 2-arg set_meta so that
  // element_size() simply equals the inflated-dim byte count.
  ometa->set_meta(core::IndexMeta::DataType::DT_INT8,
                  original_dim_ + extra_meta_size_);
  return 0;
}

void RecordInt8Quantizer::quantize_one(const void *record, void *output) const {
  const float *src = reinterpret_cast<const float *>(record);
  const float *quantize_input = src;
  float norm = 1.0f;
  std::vector<float> normalized;

  if (cosine_) {
    float sq = 0.0f;
    for (uint32_t i = 0; i < original_dim_; ++i) {
      sq += src[i] * src[i];
    }
    norm = std::sqrt(sq);

    normalized.resize(original_dim_);
    if (norm > 0.0f) {
      for (uint32_t i = 0; i < original_dim_; ++i) {
        normalized[i] = src[i] / norm;
      }
    } else {
      std::memset(normalized.data(), 0, original_dim_ * sizeof(float));
    }
    quantize_input = normalized.data();
  }

  core::RecordQuantizer::quantize_record(quantize_input, original_dim_,
                                         core::IndexMeta::DataType::DT_INT8,
                                         false, output);

  if (cosine_) {
    char *base = reinterpret_cast<char *>(output);
    const int8_t *qvals = reinterpret_cast<const int8_t *>(output);
    float *extras = reinterpret_cast<float *>(base + original_dim_);
    float qa = extras[0];
    float qb = extras[1];
    float dequant_norm_sq = 0.0f;
    for (uint32_t i = 0; i < original_dim_; ++i) {
      float val = static_cast<float>(qvals[i]) * qa + qb;
      dequant_norm_sq += val * val;
    }
    float dequant_norm = std::sqrt(dequant_norm_sq);
    if (dequant_norm > 0.0f) {
      extras[0] = qa / dequant_norm;
      extras[1] = qb / dequant_norm;
      norm *= dequant_norm;
    }

    std::memcpy(base + original_dim_ + EXTRA_META_SIZE_INT8, &norm,
                sizeof(float));
  }
}

int RecordInt8Quantizer::dequantize(const void *in,
                                    const core::IndexQueryMeta & /*qmeta*/,
                                    std::string *out) const {
  out->resize(original_dim_ * sizeof(float));
  float *dst = reinterpret_cast<float *>(&(*out)[0]);

  core::RecordQuantizer::unquantize_record(
      in, original_dim_, core::IndexMeta::DataType::DT_INT8, dst);

  if (cosine_) {
    float norm = 0.0f;
    std::memcpy(
        &norm,
        static_cast<const char *>(in) + original_dim_ + EXTRA_META_SIZE_INT8,
        sizeof(float));
    for (uint32_t i = 0; i < original_dim_; ++i) {
      dst[i] *= norm;
    }
  }

  return 0;
}

DistanceImpl RecordInt8Quantizer::distance(
    const void *query, const core::IndexQueryMeta &qmeta) const {
  auto func = get_distance_func(origin_metric_, DataType::kInt8,
                                QuantizeType::kDefault, CpuArchType::kAuto);
  if (!func) {
    return DistanceImpl{};
  }
  auto batch_func =
      get_batch_distance_func(origin_metric_, DataType::kInt8,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  std::string quantized_query;
  if (qmeta.data_type() == IndexMeta::DataType::DT_INT8) {
    // Query is already quantized — copy it directly.
    quantized_query.assign(static_cast<const char *>(query),
                           qmeta.element_size());
  } else {
    // Query needs to be quantized (e.g. FP32 → INT8).
    quantized_query.resize(quantized_length(), '\0');
    quantize_one(query, &quantized_query[0]);
  }
  // Pass the raw (non-inflated) dimension to the distance implementation.
  return DistanceImpl(std::move(func), std::move(batch_func),
                      std::move(quantized_query), original_dim_);
}

float RecordInt8Quantizer::calc_distance_dp_query(const void *dp,
                                                  const void *query) const {
  float d = 0.0f;
  if (dp_query_func_) {
    dp_query_func_(dp, query, original_dim_, &d);
  }
  return d;
}

void RecordInt8Quantizer::calc_distance_dp_query_batch(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  if (dp_query_batch_func_) {
    dp_query_batch_func_(const_cast<const void **>(dp_list), query,
                         static_cast<size_t>(dp_num), original_dim_, dist_list);
    return;
  }
  for (int i = 0; i < dp_num; ++i) {
    dist_list[i] = calc_distance_dp_query(dp_list[i], query);
  }
}

float RecordInt8Quantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  return calc_distance_dp_query(dp, buf.data());
}

void RecordInt8Quantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  calc_distance_dp_query_batch(dp_list, dp_num, buf.data(), dist_list);
}

float RecordInt8Quantizer::calc_distance_dp_dp(const void *dp1,
                                               const void *dp2) const {
  return calc_distance_dp_query(dp1, dp2);
}

INDEX_FACTORY_REGISTER_QUANTIZER(RecordInt8Quantizer);

}  // namespace turbo
}  // namespace zvec