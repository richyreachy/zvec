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

#include "quantizer/record_int4_quantizer/record_int4_quantizer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>
#include "core/quantizer/record_quantizer.h"

namespace zvec {
namespace turbo {

int RecordInt4Quantizer::init(const core::IndexMeta &meta,
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
  data_type_ = core::IndexMeta::DataType::DT_INT4;
  meta_.set_meta(data_type_, meta_.dimension());

  extra_meta_size_ = EXTRA_META_SIZE_INT4;
  if (meta.metric_name() == "Cosine") {
    cosine_ = true;
    extra_meta_size_ += EXTRA_META_SIZE_COSINE;
  } else {
    if (meta.metric_name() == "SquaredEuclidean" ||
        meta.metric_name() == "Euclidean") {
      euclidean_ = true;
    }
  }

  origin_metric_ = metric_from_name(meta.metric_name());

  meta_.set_extra_meta_size(extra_meta_size_);

  ailego::Params metric_params;
  metric_params.set("proxima.quantized_integer.metric.origin_metric_name",
                    meta.metric_name());
  metric_params.set("proxima.quantized_integer.metric.origin_metric_params",
                    meta.metric_params());
  meta_.set_metric("QuantizedInteger", 0, metric_params);

  // Cache the distance dispatch for the new Quantizer interface.
  dp_query_func_ =
      get_distance_func(origin_metric_, DataType::kInt4, QuantizeType::kDefault,
                        CpuArchType::kAuto);
  dp_query_batch_func_ =
      get_batch_distance_func(origin_metric_, DataType::kInt4,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  return 0;
}

int RecordInt4Quantizer::quantize(const void *record,
                                  const core::IndexQueryMeta & /*rmeta*/,
                                  std::string *out,
                                  core::IndexQueryMeta *ometa) const {
  out->resize(quantized_length(), 0);
  quantize_one(record, &(*out)[0]);

  *ometa = core::IndexQueryMeta();
  ometa->set_meta(core::IndexMeta::DataType::DT_INT4, meta_.dimension(),
                  static_cast<uint32_t>(type_), extra_meta_size_);
  return 0;
}

void RecordInt4Quantizer::quantize_one(const void *record, void *output) const {
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

  // INT4 packed size: original_dim_/2 bytes for data, plus extras
  size_t packed_size = original_dim_ / 2;

  core::RecordQuantizer::quantize_record(quantize_input, original_dim_,
                                         core::IndexMeta::DataType::DT_INT4,
                                         euclidean_, output);

  if (cosine_) {
    char *base = reinterpret_cast<char *>(output);
    // Read back the quantized extras
    const uint8_t *packed = reinterpret_cast<const uint8_t *>(output);
    float *extras = reinterpret_cast<float *>(base + packed_size);
    float qa = extras[0];
    float qb = extras[1];

    // Compute dequantized norm of the quantized-then-normalized vector
    float dequant_norm_sq = 0.0f;
    for (uint32_t i = 0; i < original_dim_ / 2; ++i) {
      int8_t lo = (static_cast<int8_t>(packed[i] << 4) >> 4);
      int8_t hi = (static_cast<int8_t>(packed[i] & 0xf0) >> 4);
      float val_lo = static_cast<float>(lo) * qa + qb;
      float val_hi = static_cast<float>(hi) * qa + qb;
      dequant_norm_sq += val_lo * val_lo + val_hi * val_hi;
    }
    float dequant_norm = std::sqrt(dequant_norm_sq);
    if (dequant_norm > 0.0f) {
      extras[0] = qa / dequant_norm;
      extras[1] = qb / dequant_norm;
      norm *= dequant_norm;
    }

    std::memcpy(base + packed_size + EXTRA_META_SIZE_INT4, &norm,
                sizeof(float));
  }
}

int RecordInt4Quantizer::dequantize(const void *in,
                                    const core::IndexQueryMeta & /*qmeta*/,
                                    std::string *out) const {
  out->resize(original_dim_ * sizeof(float));
  float *dst = reinterpret_cast<float *>(&(*out)[0]);

  core::RecordQuantizer::unquantize_record(
      in, original_dim_, core::IndexMeta::DataType::DT_INT4, dst);

  if (cosine_) {
    float norm = 0.0f;
    size_t packed_size = original_dim_ / 2;
    std::memcpy(
        &norm,
        static_cast<const char *>(in) + packed_size + EXTRA_META_SIZE_INT4,
        sizeof(float));
    for (uint32_t i = 0; i < original_dim_; ++i) {
      dst[i] *= norm;
    }
  }

  return 0;
}

DistanceImpl RecordInt4Quantizer::distance(
    const void *query, const core::IndexQueryMeta &qmeta) const {
  auto func = get_distance_func(origin_metric_, DataType::kInt4,
                                QuantizeType::kDefault, CpuArchType::kAuto);
  if (!func) {
    return DistanceImpl{};
  }
  auto batch_func =
      get_batch_distance_func(origin_metric_, DataType::kInt4,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  // The query is assumed to be already quantized — copy it directly.
  std::string quantized_query(static_cast<const char *>(query),
                              qmeta.element_size());
  return DistanceImpl(std::move(func), std::move(batch_func),
                      std::move(quantized_query), original_dim_);
}

float RecordInt4Quantizer::calc_distance_dp_query(const void *dp,
                                                  const void *query) const {
  float d = 0.0f;
  if (dp_query_func_) {
    dp_query_func_(dp, query, original_dim_, &d);
  }
  return d;
}

void RecordInt4Quantizer::calc_distance_dp_query_batch(
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

float RecordInt4Quantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  return calc_distance_dp_query(dp, buf.data());
}

void RecordInt4Quantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  calc_distance_dp_query_batch(dp_list, dp_num, buf.data(), dist_list);
}

float RecordInt4Quantizer::calc_distance_dp_dp(const void *dp1,
                                               const void *dp2) const {
  return calc_distance_dp_query(dp1, dp2);
}

INDEX_FACTORY_REGISTER_QUANTIZER(RecordInt4Quantizer);

}  // namespace turbo
}  // namespace zvec
