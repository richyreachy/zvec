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
  meta_.set_meta(data_type_, meta_.dimension());

  extra_meta_size_ = EXTRA_META_SIZE_INT8;
  if (meta.metric_name() == "Cosine") {
    cosine_ = true;
    extra_meta_size_ += EXTRA_META_SIZE_COSINE;
  }

  meta_.set_extra_meta_size(extra_meta_size_);

  ailego::Params metric_params;
  metric_params.set("proxima.quantized_integer.metric.origin_metric_name",
                    meta.metric_name());
  metric_params.set("proxima.quantized_integer.metric.origin_metric_params",
                    meta.metric_params());
  meta_.set_metric("QuantizedInteger", 0, metric_params);

  return 0;
}

int RecordInt8Quantizer::quantize(const void *record,
                                  const core::IndexQueryMeta & /*rmeta*/,
                                  std::string *out,
                                  core::IndexQueryMeta *ometa) const {
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

  size_t packed_size = original_dim_;
  size_t total_size = packed_size + EXTRA_META_SIZE_INT8;
  if (cosine_) {
    total_size += EXTRA_META_SIZE_COSINE;
  }
  out->resize(total_size, 0);
  core::RecordQuantizer::quantize_record(quantize_input, original_dim_,
                                         core::IndexMeta::DataType::DT_INT8,
                                         false, &(*out)[0]);

  if (cosine_) {
    const int8_t *qvals = reinterpret_cast<const int8_t *>(out->data());
    float *extras = reinterpret_cast<float *>(&(*out)[original_dim_]);
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

    std::memcpy(&(*out)[original_dim_ + EXTRA_META_SIZE_INT8], &norm,
                sizeof(float));
  }

  *ometa = core::IndexQueryMeta();
  ometa->set_meta(core::IndexMeta::DataType::DT_INT8, meta_.dimension(),
                  static_cast<uint32_t>(type_), extra_meta_size_);
  return 0;
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

INDEX_FACTORY_REGISTER_QUANTIZER(RecordInt8Quantizer);

}  // namespace turbo
}  // namespace zvec