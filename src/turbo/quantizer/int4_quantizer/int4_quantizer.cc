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

#include "quantizer/int4_quantizer/int4_quantizer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>
#include "core/quantizer/record_quantizer.h"

namespace zvec {
namespace turbo {

int Int4Quantizer::init(const core::IndexMeta &meta,
                        const ailego::Params &params) {
  data_type_ = IndexMeta::DataType::DT_INT4;
  meta_ = meta;
  meta_.set_meta(data_type_, meta.dimension());
  original_dim_ = meta.dimension();

  if (params.get(INT4_QUANTIZER_BIAS, &bias_) &&
      params.get(INT4_QUANTIZER_SCALE, &scale_)) {
    quantizer_.set_bias(bias_);
    quantizer_.set_scale(scale_);
  }

  extra_meta_size_ = EXTRA_META_SIZE_INT4;

  auto metric_name = meta.metric_name();
  auto reciprocal = scale_ == 0.0 ? 1.0f : (1.0f / scale_);
  if (metric_name == "SquaredEuclidean") {
    scale_reciprocal_ = reciprocal * reciprocal;
  } else if (metric_name == "Euclidean") {
    scale_reciprocal_ = reciprocal;
  } else if (metric_name == "InnerProduct") {
    inner_product_ = true;
    scale_reciprocal_ = reciprocal;
  } else if (metric_name == "Cosine") {
    inner_product_ = true;
    cosine_ = true;
    scale_reciprocal_ = reciprocal;  // missing query part

    extra_meta_size_ += EXTRA_META_SIZE_COSINE;
    meta_.set_extra_meta_size(extra_meta_size_);
  } else {
    LOG_WARN("Unsupported normalize the score for %s", metric_name.c_str());
    scale_reciprocal_ = 1.0f;
  }

  meta_.set_extra_meta_size(extra_meta_size_);

  // Cache the distance dispatch for the new Quantizer interface.
  dp_query_func_ =
      get_distance_func(metric_from_name(metric_name), DataType::kInt4,
                        QuantizeType::kDefault, CpuArchType::kAuto);
  dp_query_batch_func_ =
      get_batch_distance_func(metric_from_name(metric_name), DataType::kInt4,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  LOG_DEBUG("Init integer reformer, bias %f, scale %f", bias_, scale_);
  return 0;
}

int Int4Quantizer::train(core::IndexHolder::Pointer holder) {
  if (holder->dimension() != meta_.dimension() ||
      holder->data_type() != IndexMeta::DataType::DT_FP32) {
    return IndexError_Mismatch;
  }

  ailego::ElapsedTime timer;

  //! step1: compute max/min value
  auto iter = holder->create_iterator();
  if (!iter) {
    LOG_ERROR("Failed to create iterator of holder");
    return IndexError_Runtime;
  }
  std::vector<float> features;
  float max = -std::numeric_limits<float>::max();
  float min = std::numeric_limits<float>::max();
  for (; iter->is_valid(); iter->next()) {
    const float *vec = reinterpret_cast<const float *>(iter->data());
    for (size_t i = 0; i < meta_.dimension(); ++i) {
      max = std::max(max, vec[i]);
      min = std::min(min, vec[i]);
      features.emplace_back(vec[i]);
    }
  }
  quantizer_.set_max(max);
  quantizer_.set_min(min);

  //! step2: feed quantizer with training data
  for (size_t i = 0; i < features.size(); i += meta_.dimension()) {
    quantizer_.feed(&features[i], meta_.dimension());
  }

  //! step3: feed quantizer with training data
  if (!quantizer_.train()) {
    LOG_ERROR("Quantizer train failed");
    return IndexError_Runtime;
  }

  bias_ = quantizer_.bias();
  scale_ = quantizer_.scale();

  LOG_DEBUG(
      "IntegerQuantizerConverter train done, costtime %zums, scale %f, bias "
      "%f",
      (size_t)timer.milli_seconds(), scale_, bias_);

  return 0;
}

int Int4Quantizer::quantize(const void *record, const IndexQueryMeta &qmeta,
                            std::string *out, IndexQueryMeta *ometa) const {
  IndexMeta::DataType ft = qmeta.data_type();

  if (ft != IndexMeta::DataType::DT_FP32 ||
      qmeta.unit_size() !=
          IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
    return IndexError_Unsupported;
  }

  *ometa = qmeta;
  ometa->set_meta(data_type_, qmeta.dimension(), static_cast<uint32_t>(type_),
                  extra_meta_size_);
  size_t packed_size =
      IndexMeta::ElementSizeof(ometa->data_type(), ometa->dimension());
  size_t total_size = packed_size;
  if (inner_product_) {
    total_size += EXTRA_META_SIZE_INT4;
    if (cosine_) {
      total_size += EXTRA_META_SIZE_COSINE;
    }
  }
  out->resize(total_size, 0);
  quantize_one(record, &(*out)[0]);

  return 0;
}

void Int4Quantizer::quantize_one(const void *input, void *output) const {
  const float *vec = reinterpret_cast<const float *>(input);
  auto ovec = reinterpret_cast<uint8_t *>(output);
  size_t dim = original_dim_;

  if (!inner_product_) {
    quantizer_.encode(vec, dim, ovec);
  } else {
    const float *quantize_input = vec;
    float norm = 1.0f;
    std::vector<float> normalized;

    if (cosine_) {
      // L2-normalize the input so the cosine distance function (which returns
      // -<a,b> for unit vectors) produces -cos_sim.
      float sq = 0.0f;
      for (size_t i = 0; i < dim; ++i) {
        sq += quantize_input[i] * quantize_input[i];
      }
      norm = std::sqrt(sq);
      normalized.resize(dim);
      if (norm > 0.0f) {
        for (size_t i = 0; i < dim; ++i) {
          normalized[i] = quantize_input[i] / norm;
        }
      } else {
        std::memset(normalized.data(), 0, dim * sizeof(float));
      }
      quantize_input = normalized.data();
    }

    float abs_max = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      float a = std::abs(quantize_input[i]);
      abs_max = std::max(a, abs_max);
    }
    if (abs_max == 0.0f) abs_max = 1.0f;
    float scale = 7.0f / abs_max;
    float sum = 0.0f;
    float squared_sum = 0.0f;
    int int_sum = 0;

    // Pack int4 values (2 per byte): low nibble = even index, high nibble = odd
    for (size_t i = 0; i < dim; i += 2) {
      float lo_f = std::round(quantize_input[i] * scale);
      float hi_f = std::round(quantize_input[i + 1] * scale);
      int8_t lo = static_cast<int8_t>(lo_f);
      int8_t hi = static_cast<int8_t>(hi_f);
      ovec[i / 2] =
          (static_cast<uint8_t>(hi) << 4) | (static_cast<uint8_t>(lo) & 0xF);
      sum += lo_f + hi_f;
      squared_sum += lo_f * lo_f + hi_f * hi_f;
      int_sum += lo + hi;
    }

    size_t packed_bytes = dim / 2;
    float qa = abs_max / 7.0f;  // dequant scale
    float qb = 0.0f;            // dequant bias

    if (cosine_) {
      // Adjust qa/qb so the dequantized vector has unit norm.
      float dequant_norm_sq = 0.0f;
      for (size_t i = 0; i < packed_bytes; ++i) {
        int8_t lo = (static_cast<int8_t>(ovec[i] << 4) >> 4);
        int8_t hi = (static_cast<int8_t>(ovec[i] & 0xf0) >> 4);
        float val_lo = static_cast<float>(lo) * qa + qb;
        float val_hi = static_cast<float>(hi) * qa + qb;
        dequant_norm_sq += val_lo * val_lo + val_hi * val_hi;
      }
      float dequant_norm = std::sqrt(dequant_norm_sq);
      if (dequant_norm > 0.0f) {
        qa /= dequant_norm;
        qb /= dequant_norm;
        norm *= dequant_norm;
      }
    }

    // Write extras after packed int4 data
    float *extras = reinterpret_cast<float *>(ovec + packed_bytes);
    extras[0] = qa;           // dequant scale
    extras[1] = qb;           // dequant bias
    extras[2] = sum;          // sum of quantized values
    extras[3] = squared_sum;  // squared sum
    reinterpret_cast<int *>(extras)[4] = int_sum;

    if (cosine_) {
      std::memcpy(
          reinterpret_cast<char *>(ovec + packed_bytes) + EXTRA_META_SIZE_INT4,
          &norm, sizeof(float));
    }
  }
}

int Int4Quantizer::dequantize(const void *in, const IndexQueryMeta &qmeta,
                              std::string *out) const {
  if (!in || !out) {
    return IndexError_InvalidArgument;
  }

  // Use original_dim_ to avoid reading the inflated dimension (data + extras)
  size_t dim = original_dim_;
  const uint8_t *ivec = reinterpret_cast<const uint8_t *>(in);
  out->resize(dim * sizeof(float));
  float *ovec = reinterpret_cast<float *>(&(*out)[0]);

  if (!inner_product_) {
    quantizer_.decode(ivec, dim, ovec);
  } else {
    // Read dequant metadata (qa, qb) stored after the packed int4 data
    size_t packed_bytes = dim / 2;
    const float *extras = reinterpret_cast<const float *>(ivec + packed_bytes);
    float qa = extras[0];
    float qb = extras[1];
    for (size_t i = 0; i < packed_bytes; ++i) {
      int8_t lo = (static_cast<int8_t>(ivec[i] << 4) >> 4);
      int8_t hi = (static_cast<int8_t>(ivec[i] & 0xf0) >> 4);
      ovec[i * 2] = static_cast<float>(lo) * qa + qb;
      ovec[i * 2 + 1] = static_cast<float>(hi) * qa + qb;
    }
    if (cosine_) {
      // Denormalize using the stored original norm
      float norm = 0.0f;
      std::memcpy(&norm,
                  reinterpret_cast<const char *>(in) + packed_bytes +
                      EXTRA_META_SIZE_INT4,
                  sizeof(float));
      for (size_t i = 0; i < dim; ++i) {
        ovec[i] *= norm;
      }
    }
  }

  return 0;
}

int Int4Quantizer::serialize(std::string *out) const {
  if (!out) {
    return IndexError_InvalidArgument;
  }
  constexpr uint32_t kPayloadSize = sizeof(float) * 2;
  out->resize(sizeof(QuantizerSerHeader) + kPayloadSize);

  QuantizerSerHeader *header =
      reinterpret_cast<QuantizerSerHeader *>(&(*out)[0]);
  header->magic = kQuantizerMagic;
  header->version = kQuantizerSerVersion;
  header->quant_type = static_cast<uint16_t>(type_);
  header->dim = original_dim_;
  header->metric = static_cast<uint32_t>(metric_from_name(meta_.metric_name()));
  header->payload_size = kPayloadSize;
  header->reserved = 0;

  float *buf = reinterpret_cast<float *>(&(*out)[sizeof(QuantizerSerHeader)]);
  buf[0] = quantizer_.bias();
  buf[1] = quantizer_.scale();
  return 0;
}

int Int4Quantizer::deserialize(std::string &in) {
  return deserialize(in.data(), in.size());
}

int Int4Quantizer::deserialize(const void *data, size_t len) {
  if (!data || len < sizeof(QuantizerSerHeader)) {
    return IndexError_InvalidArgument;
  }
  const QuantizerSerHeader *header =
      reinterpret_cast<const QuantizerSerHeader *>(data);
  if (header->magic != kQuantizerMagic ||
      header->version != kQuantizerSerVersion ||
      header->payload_size < sizeof(float) * 2 ||
      len < sizeof(QuantizerSerHeader) + header->payload_size) {
    return IndexError_InvalidArgument;
  }
  if (header->dim != original_dim_ ||
      header->metric !=
          static_cast<uint32_t>(metric_from_name(meta_.metric_name()))) {
    return IndexError_InvalidArgument;
  }

  const float *buf = reinterpret_cast<const float *>(
      reinterpret_cast<const char *>(data) + sizeof(QuantizerSerHeader));
  bias_ = buf[0];
  scale_ = buf[1];
  quantizer_.set_bias(bias_);
  quantizer_.set_scale(scale_);
  return 0;
}

DistanceImpl Int4Quantizer::distance(const void *query,
                                     const IndexQueryMeta &qmeta) const {
  auto metric = metric_from_name(meta_.metric_name());
  auto func = get_distance_func(metric, DataType::kInt4, QuantizeType::kDefault,
                                CpuArchType::kAuto);
  if (!func) {
    return DistanceImpl{};
  }
  auto batch_func = get_batch_distance_func(
      metric, DataType::kInt4, QuantizeType::kDefault, CpuArchType::kAuto);

  // The query is assumed to be already quantized — copy it directly.
  std::string quantized_query(static_cast<const char *>(query),
                              qmeta.element_size());
  return DistanceImpl(std::move(func), std::move(batch_func),
                      std::move(quantized_query), original_dim_);
}

int Int4Quantizer::train(const void *data, size_t num, size_t stride) {
  ailego::ElapsedTime timer;

  std::vector<float> features;
  features.reserve(num * original_dim_);
  float max = -std::numeric_limits<float>::max();
  float min = std::numeric_limits<float>::max();
  const char *base = reinterpret_cast<const char *>(data);
  for (size_t n = 0; n < num; ++n) {
    const float *vec = reinterpret_cast<const float *>(base + n * stride);
    for (size_t i = 0; i < original_dim_; ++i) {
      max = std::max(max, vec[i]);
      min = std::min(min, vec[i]);
      features.emplace_back(vec[i]);
    }
  }
  quantizer_.set_max(max);
  quantizer_.set_min(min);

  for (size_t i = 0; i < features.size(); i += original_dim_) {
    quantizer_.feed(&features[i], original_dim_);
  }

  if (!quantizer_.train()) {
    LOG_ERROR("Quantizer train failed");
    return IndexError_Runtime;
  }

  bias_ = quantizer_.bias();
  scale_ = quantizer_.scale();

  LOG_DEBUG("Int4Quantizer train done, costtime %zums, scale %f, bias %f",
            (size_t)timer.milli_seconds(), scale_, bias_);
  return 0;
}

float Int4Quantizer::calc_distance_dp_query(const void *dp,
                                            const void *query) const {
  float d = 0.0f;
  if (dp_query_func_) {
    dp_query_func_(dp, query, original_dim_, &d);
  }
  return d;
}

void Int4Quantizer::calc_distance_dp_query_batch(const void *const *dp_list,
                                                 int dp_num, const void *query,
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

float Int4Quantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  return calc_distance_dp_query(dp, buf.data());
}

void Int4Quantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  calc_distance_dp_query_batch(dp_list, dp_num, buf.data(), dist_list);
}

float Int4Quantizer::calc_distance_dp_dp(const void *dp1,
                                         const void *dp2) const {
  return calc_distance_dp_query(dp1, dp2);
}

INDEX_FACTORY_REGISTER_QUANTIZER(Int4Quantizer);

}  // namespace turbo
}  // namespace zvec