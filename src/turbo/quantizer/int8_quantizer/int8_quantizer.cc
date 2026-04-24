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
  meta_.set_meta(data_type_, meta.dimension());
  original_dim_ = meta.dimension();

  if (params.get(INT8_QUANTIZER_BIAS, &bias_) &&
      params.get(INT8_QUANTIZER_SCALE, &scale_)) {
    quantizer_.set_bias(bias_);
    quantizer_.set_scale(scale_);
  }

  auto metric_name = meta.metric_name();
  auto reciprocal = scale_ == 0.0 ? 1.0f : (1.0f / scale_);

  extra_meta_size_ = EXTRA_META_SIZE_INT8;
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
    scale_reciprocal_ = reciprocal;
    extra_meta_size_ += EXTRA_META_SIZE_COSINE;
  } else {
    LOG_WARN("Unsupported normalize the score for %s", metric_name.c_str());
    scale_reciprocal_ = 1.0f;
  }

  meta_.set_extra_meta_size(extra_meta_size_);

  LOG_DEBUG("Init integer reformer, bias %f, scale %f", bias_, scale_);
  return 0;
}

int Int8Quantizer::train(core::IndexHolder::Pointer holder) {
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

int Int8Quantizer::quantize(const void *record, const IndexQueryMeta &qmeta,
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
  size_t base_size =
      IndexMeta::ElementSizeof(ometa->data_type(), ometa->dimension());
  if (inner_product_) {
    base_size += EXTRA_META_SIZE_INT8;
    if (cosine_) {
      base_size += EXTRA_META_SIZE_COSINE;
    }
  }
  out->resize(base_size, 0);
  const float *vec = reinterpret_cast<const float *>(record);
  auto ovec = reinterpret_cast<int8_t *>(&(*out)[0]);

  if (!inner_product_) {
    quantizer_.encode(vec, qmeta.dimension(), ovec);
  } else {
    size_t dim = qmeta.dimension();
    const float *quantize_input = vec;

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

    // Write extras after int8 data
    float *extras = reinterpret_cast<float *>(ovec + dim);
    extras[0] = abs_max / 127.0f;  // qa: dequant scale
    extras[1] = 0.0f;              // qb: dequant bias
    extras[2] = sum;               // qs: sum of quantized values
    extras[3] = squared_sum;       // squared sum
    reinterpret_cast<int32_t *>(extras + 4)[0] = int8_sum;
  }

  return 0;
}

int Int8Quantizer::dequantize(const void *in, const IndexQueryMeta &qmeta,
                              std::string *out) const {
  if (!in || !out) {
    return IndexError_InvalidArgument;
  }

  size_t dim = qmeta.dimension();
  const int8_t *ivec = reinterpret_cast<const int8_t *>(in);
  out->resize(dim * sizeof(float));
  float *ovec = reinterpret_cast<float *>(&(*out)[0]);

  if (!inner_product_) {
    quantizer_.decode(ivec, dim, ovec);
  } else {
    for (size_t i = 0; i < dim; ++i) {
      ovec[i] = static_cast<float>(ivec[i]);
    }
  }

  return 0;
}

int Int8Quantizer::serialize(std::string *out) const {
  if (!out) {
    return IndexError_InvalidArgument;
  }
  out->resize(sizeof(float) * 2);
  float *buf = reinterpret_cast<float *>(&(*out)[0]);
  buf[0] = quantizer_.bias();
  buf[1] = quantizer_.scale();
  return 0;
}

int Int8Quantizer::deserialize(std::string &in) {
  if (in.size() < sizeof(float) * 2) {
    return IndexError_InvalidArgument;
  }
  const float *buf = reinterpret_cast<const float *>(in.data());
  bias_ = buf[0];
  scale_ = buf[1];
  quantizer_.set_bias(bias_);
  quantizer_.set_scale(scale_);
  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(Int8Quantizer);

}  // namespace turbo
}  // namespace zvec