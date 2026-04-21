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
  if (!params.get(INT8_QUANTIZER_BIAS, &bias_) ||
      !params.get(INT8_QUANTIZER_SCALE, &scale_)) {
    LOG_ERROR("Init IntegerReformer failed, required params bias and scale");
    return IndexError_InvalidArgument;
  }

  quantizer_.set_bias(bias_);
  quantizer_.set_scale(scale_);

  auto metric_name = meta.metric_name();
  auto reciprocal = scale_ == 0.0 ? 1.0f : (1.0f / scale_);
  if (metric_name == "SquaredEuclidean") {
    scale_reciprocal_ = reciprocal * reciprocal;
  } else if (metric_name == "Euclidean") {
    scale_reciprocal_ = reciprocal;
  } else if (metric_name == "InnerProduct" ||
             metric_name == "MipsSquaredEuclidean") {
    inner_product_ = true;
    scale_reciprocal_ = reciprocal;  // missing query part
  } else {
    LOG_WARN("Unsupported normalize the score for %s", metric_name.c_str());
    scale_reciprocal_ = 1.0f;
  }
  LOG_DEBUG("Init integer reformer, bias %f, scale %f", bias_, scale_);
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
  ometa->set_meta(data_type_, qmeta.dimension());
  out->resize(IndexMeta::ElementSizeof(ometa->data_type(), ometa->dimension()));
  const float *vec = reinterpret_cast<const float *>(record);
  auto ovec = reinterpret_cast<int8_t *>(&(*out)[0]);

  if (!inner_product_) {
    quantizer_.encode(vec, qmeta.dimension(), ovec);
  } else {
    size_t dim = qmeta.dimension();
    float abs_max = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      float abs = std::abs(vec[i]);
      abs_max = std::max(abs, abs_max);
    }
    float scale = 127.0f / abs_max;
    for (size_t i = 0; i < dim; ++i) {
      ovec[i] = static_cast<int8_t>(std::round(vec[i] * scale));
    }
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

INDEX_FACTORY_REGISTER_QUANTIZER(Int8Quantizer);

}  // namespace turbo
}  // namespace zvec