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

#include "quantizer/int8_quantizer/int8_quantizer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>
#include "core/quantizer/record_quantizer.h"

namespace zvec {
namespace turbo {

int Int8Quantizer::init(const IndexMeta &meta, const ailego::Params &params) {
  data_type_ = IndexMeta::DataType::DT_INT8;
  meta_ = meta;
  original_dim_ = meta.dimension();

  if (params.get(INT8_QUANTIZER_BIAS, &bias_) &&
      params.get(INT8_QUANTIZER_SCALE, &scale_)) {
    quantizer_.set_bias(bias_);
    quantizer_.set_scale(scale_);
  }

  auto metric_name = meta.metric_name();
  auto reciprocal = scale_ == 0.0 ? 1.0f : (1.0f / scale_);

  extra_meta_size_ = 0;
  if (metric_name == "SquaredEuclidean") {
    // Per-vector quantization (RecordQuantizer layout) so the test does not
    // need to call train() first. Stored layout: [int8 data][20-byte tail =
    // 4 floats (qa/qb/qs/qs2) + 1 int (int8_sum)] which matches what the
    // turbo SE INT8 distance expects when the metric is wrapped in
    // QuantizedInteger.
    record_quantize_ = true;
    scale_reciprocal_ = reciprocal * reciprocal;
    extra_meta_size_ = EXTRA_META_SIZE_INT8;
  } else if (metric_name == "Euclidean") {
    scale_reciprocal_ = reciprocal;
  } else if (metric_name == "InnerProduct") {
    inner_product_ = true;
    scale_reciprocal_ = reciprocal;
    extra_meta_size_ = EXTRA_META_SIZE_INT8;
  } else if (metric_name == "Cosine") {
    inner_product_ = true;
    cosine_ = true;
    scale_reciprocal_ = reciprocal;
    extra_meta_size_ = EXTRA_META_SIZE_INT8 + EXTRA_META_SIZE_COSINE;
  } else {
    LOG_WARN("Unsupported normalize the score for %s", metric_name.c_str());
    scale_reciprocal_ = 1.0f;
  }

  // Inflate dimension by extra bytes (per-element unit=1 for INT8) so that
  // meta_.element_size() reflects the actual per-vector storage size and
  // HnswStreamer::check_params matches the ometa produced by quantize().
  meta_.set_meta(data_type_, original_dim_ + extra_meta_size_);
  meta_.set_extra_meta_size(extra_meta_size_);

  if (record_quantize_) {
    // Wrap the metric in QuantizedInteger so the streamer uses the turbo
    // metadata-aware INT8 distance (matches RecordInt8Quantizer's approach).
    ailego::Params metric_params;
    metric_params.set("proxima.quantized_integer.metric.origin_metric_name",
                      metric_name);
    metric_params.set("proxima.quantized_integer.metric.origin_metric_params",
                      meta.metric_params());
    origin_metric_ = metric_from_name(metric_name);
    meta_.set_metric("QuantizedInteger", 0, metric_params);
  }

  // Cache the distance dispatch for the new Quantizer interface. For the
  // record-quantize paths the wrapped meta_ metric is "QuantizedInteger";
  // we keep the original metric for the turbo dispatch.
  dist_metric_ =
      record_quantize_ ? origin_metric_ : metric_from_name(metric_name);
  dp_query_func_ =
      get_distance_func(dist_metric_, DataType::kInt8, QuantizeType::kDefault,
                        CpuArchType::kAuto);
  dp_query_batch_func_ =
      get_batch_distance_func(dist_metric_, DataType::kInt8,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  LOG_DEBUG("Init integer reformer, bias %f, scale %f", bias_, scale_);
  return 0;
}

int Int8Quantizer::train(core::IndexHolder::Pointer holder) {
  if (holder->dimension() != original_dim_ ||
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
  if (rotator_) {
    rotator_->train(nullptr, 0, 0);
  }
  std::vector<float> rotated;
  if (rotator_) {
    rotated.resize(original_dim_);
  }
  for (; iter->is_valid(); iter->next()) {
    const float *vec = reinterpret_cast<const float *>(iter->data());
    if (rotator_) {
      rotator_->apply(vec, rotated.data());
      vec = rotated.data();
    }
    for (size_t i = 0; i < original_dim_; ++i) {
      max = std::max(max, vec[i]);
      min = std::min(min, vec[i]);
      features.emplace_back(vec[i]);
    }
  }
  quantizer_.set_max(max);
  quantizer_.set_min(min);

  //! step2: feed quantizer with training data
  for (size_t i = 0; i < features.size(); i += original_dim_) {
    quantizer_.feed(&features[i], original_dim_);
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

int Int8Quantizer::train(const void *data, size_t num, size_t stride) {
  ailego::ElapsedTime timer;

  std::vector<float> features;
  features.reserve(num * original_dim_);
  float max = -std::numeric_limits<float>::max();
  float min = std::numeric_limits<float>::max();
  const char *base = reinterpret_cast<const char *>(data);
  if (rotator_) {
    rotator_->train(data, num, stride);
  }
  std::vector<float> rotated;
  if (rotator_) {
    rotated.resize(original_dim_);
  }
  for (size_t n = 0; n < num; ++n) {
    const float *vec = reinterpret_cast<const float *>(base + n * stride);
    if (rotator_) {
      rotator_->apply(vec, rotated.data());
      vec = rotated.data();
    }
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

  LOG_DEBUG("Int8Quantizer train done, costtime %zums, scale %f, bias %f",
            (size_t)timer.milli_seconds(), scale_, bias_);
  return 0;
}

int Int8Quantizer::quantize(const void *record, const IndexQueryMeta &qmeta,
                            std::string *out, IndexQueryMeta *ometa) const {
  IndexMeta::DataType ft = qmeta.data_type();

  if (ft != IndexMeta::DataType::DT_FP32 ||
      qmeta.unit_size() !=
          IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
    return IndexError_Unsupported;
  }

  *ometa = qmeta;
  // Inflate ometa dimension to match meta_ (data + extras). The HnswStreamer's
  // check_params validates qmeta.dimension() == meta_.dimension(), so the
  // output meta must use the same inflated dimension as quantizer->meta().
  ometa->set_meta(data_type_, qmeta.dimension() + extra_meta_size_);
  out->resize(ometa->element_size(), 0);
  quantize_one(record, &(*out)[0]);

  return 0;
}

void Int8Quantizer::quantize_one(const void *input, void *output) const {
  const float *vec = reinterpret_cast<const float *>(input);
  thread_local std::vector<float> rotated;
  if (rotator_) {
    rotated.resize(original_dim_);
    rotator_->apply(vec, rotated.data());
    vec = rotated.data();
  }
  auto ovec = reinterpret_cast<int8_t *>(output);
  size_t dim = original_dim_;

  if (record_quantize_) {
    // Per-vector quantization with RecordQuantizer layout (matches turbo SE
    // INT8 distance metadata format: [int8 data][qa][qb][qs][qs2][int8_sum]).
    core::RecordQuantizer::quantize_record(
        vec, dim, core::IndexMeta::DataType::DT_INT8, false, ovec);
  } else if (!inner_product_) {
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
    float scale = 127.0f / abs_max;
    float sum = 0.0f;
    float squared_sum = 0.0f;
    int int8_sum = 0;
    for (size_t i = 0; i < dim; ++i) {
      int8_t v = static_cast<int8_t>(std::round(quantize_input[i] * scale));
      ovec[i] = v;
      sum += static_cast<float>(v);
      squared_sum += static_cast<float>(v) * static_cast<float>(v);
      int8_sum += v;
    }

    float qa = abs_max / 127.0f;  // dequant scale
    float qb = 0.0f;              // dequant bias

    if (cosine_) {
      // Adjust qa/qb so the dequantized vector has unit norm, eliminating
      // quantization rounding error from the cosine computation.
      float dequant_norm_sq = 0.0f;
      for (size_t i = 0; i < dim; ++i) {
        float val = static_cast<float>(ovec[i]) * qa + qb;
        dequant_norm_sq += val * val;
      }
      float dequant_norm = std::sqrt(dequant_norm_sq);
      if (dequant_norm > 0.0f) {
        qa /= dequant_norm;
        qb /= dequant_norm;
        norm *= dequant_norm;
      }
    }

    // Write extras after int8 data
    float *extras = reinterpret_cast<float *>(ovec + dim);
    extras[0] = qa;           // dequant scale
    extras[1] = qb;           // dequant bias
    extras[2] = sum;          // sum of quantized values
    extras[3] = squared_sum;  // squared sum
    reinterpret_cast<int32_t *>(extras + 4)[0] = int8_sum;

    if (cosine_) {
      // Store the original norm for dequantization
      std::memcpy(reinterpret_cast<char *>(ovec + dim) + EXTRA_META_SIZE_INT8,
                  &norm, sizeof(float));
    }
  }
}

int Int8Quantizer::dequantize(const void *in, const IndexQueryMeta & /*qmeta*/,
                              std::string *out) const {
  if (!in || !out) {
    return IndexError_InvalidArgument;
  }

  // Always decode the original (pre-quantization) dimension; the IndexQueryMeta
  // passed in may have its dimension inflated by extras.
  size_t dim = original_dim_;
  const int8_t *ivec = reinterpret_cast<const int8_t *>(in);
  out->resize(dim * sizeof(float));
  float *ovec = reinterpret_cast<float *>(&(*out)[0]);

  if (record_quantize_) {
    // Decode using the per-vector tail metadata produced by
    // RecordQuantizer::quantize_record.
    core::RecordQuantizer::unquantize_record(
        in, dim, core::IndexMeta::DataType::DT_INT8, ovec);
  } else if (!inner_product_) {
    quantizer_.decode(ivec, dim, ovec);
  } else {
    // Read dequant metadata (qa, qb) stored after the int8 data
    const float *extras = reinterpret_cast<const float *>(ivec + dim);
    float qa = extras[0];
    float qb = extras[1];
    for (size_t i = 0; i < dim; ++i) {
      ovec[i] = static_cast<float>(ivec[i]) * qa + qb;
    }
    if (cosine_) {
      // Denormalize using the stored original norm
      float norm = 0.0f;
      std::memcpy(
          &norm,
          reinterpret_cast<const char *>(in) + dim + EXTRA_META_SIZE_INT8,
          sizeof(float));
      for (size_t i = 0; i < dim; ++i) {
        ovec[i] *= norm;
      }
    }
  }

  if (rotator_) {
    // Undo the rotation so the result returns to the original space.
    std::vector<float> tmp(dim);
    rotator_->apply_inverse(ovec, tmp.data());
    std::memcpy(ovec, tmp.data(), dim * sizeof(float));
  }

  return 0;
}

int Int8Quantizer::serialize(std::string *out) const {
  if (!out) {
    return IndexError_InvalidArgument;
  }
  constexpr uint32_t kScalarSize = sizeof(float) * 2;
  std::string rotator_blob;
  if (rotator_) {
    int rc = rotator_->serialize(&rotator_blob);
    if (rc != 0) {
      return rc;
    }
  }
  const uint32_t payload_size =
      kScalarSize + static_cast<uint32_t>(rotator_blob.size());
  out->resize(sizeof(QuantizerSerHeader) + payload_size);

  QuantizerSerHeader *header =
      reinterpret_cast<QuantizerSerHeader *>(&(*out)[0]);
  header->magic = kQuantizerMagic;
  header->version = kQuantizerSerVersion;
  header->quant_type = static_cast<uint16_t>(type_);
  header->dim = original_dim_;
  header->metric = static_cast<uint32_t>(dist_metric_);
  header->payload_size = payload_size;
  header->reserved = 0;

  float *buf = reinterpret_cast<float *>(&(*out)[sizeof(QuantizerSerHeader)]);
  buf[0] = quantizer_.bias();
  buf[1] = quantizer_.scale();
  if (!rotator_blob.empty()) {
    std::memcpy(&(*out)[sizeof(QuantizerSerHeader) + kScalarSize],
                rotator_blob.data(), rotator_blob.size());
  }
  return 0;
}

int Int8Quantizer::deserialize(std::string &in) {
  return deserialize(in.data(), in.size());
}

int Int8Quantizer::deserialize(const void *data, size_t len) {
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
      header->metric != static_cast<uint32_t>(dist_metric_)) {
    return IndexError_InvalidArgument;
  }

  const float *buf = reinterpret_cast<const float *>(
      reinterpret_cast<const char *>(data) + sizeof(QuantizerSerHeader));
  bias_ = buf[0];
  scale_ = buf[1];
  quantizer_.set_bias(bias_);
  quantizer_.set_scale(scale_);

  rotator_.reset();
  if (header->payload_size > sizeof(float) * 2) {
    const char *rot_ptr = reinterpret_cast<const char *>(data) +
                          sizeof(QuantizerSerHeader) + sizeof(float) * 2;
    const size_t rot_len = header->payload_size - sizeof(float) * 2;
    rotator_ = CreateRotatorFromBlob(rot_ptr, rot_len);
    if (!rotator_ || rotator_->in_dim() != static_cast<int>(original_dim_) ||
        rotator_->out_dim() != static_cast<int>(original_dim_)) {
      rotator_.reset();
      return IndexError_InvalidArgument;
    }
  }
  return 0;
}

int Int8Quantizer::set_rotator(Rotator::Pointer r) {
  if (!r || r->in_dim() != static_cast<int>(original_dim_) ||
      r->out_dim() != static_cast<int>(original_dim_)) {
    return IndexError_InvalidArgument;
  }
  rotator_ = std::move(r);
  return 0;
}

DistanceImpl Int8Quantizer::distance(const void *query,
                                     const IndexQueryMeta &qmeta) const {
  auto func = get_distance_func(dist_metric_, DataType::kInt8,
                                QuantizeType::kDefault, CpuArchType::kAuto);
  if (!func) {
    return DistanceImpl{};
  }
  auto batch_func =
      get_batch_distance_func(dist_metric_, DataType::kInt8,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  // The query is assumed to be already quantized — copy it directly.
  std::string quantized_query(static_cast<const char *>(query),
                              qmeta.element_size());
  // Pass the raw (non-inflated) dimension to the distance implementation.
  return DistanceImpl(std::move(func), std::move(batch_func),
                      std::move(quantized_query), original_dim_);
}

float Int8Quantizer::calc_distance_dp_query(const void *dp,
                                            const void *query) const {
  float d = 0.0f;
  if (dp_query_func_) {
    dp_query_func_(dp, query, original_dim_, &d);
  }
  return d;
}

void Int8Quantizer::calc_distance_dp_query_batch(const void *const *dp_list,
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

float Int8Quantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  return calc_distance_dp_query(dp, buf.data());
}

void Int8Quantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  calc_distance_dp_query_batch(dp_list, dp_num, buf.data(), dist_list);
}

float Int8Quantizer::calc_distance_dp_dp(const void *dp1,
                                         const void *dp2) const {
  return calc_distance_dp_query(dp1, dp2);
}

INDEX_FACTORY_REGISTER_QUANTIZER(Int8Quantizer);

}  // namespace turbo
}  // namespace zvec