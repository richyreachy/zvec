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

#include <cmath>
#include <cstring>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>
#include "core/quantizer/record_quantizer.h"
#include "quantizer/int8_quantizer/int8_quantier.h"

namespace zvec {
namespace turbo {

int Int8Quantizer::init(const core::IndexMeta &meta,
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

  // Include extra dimensions in the dimension field so that element_size()
  // and the QuantizedInteger distance function both work correctly.
  // For SquaredEuclidean / InnerProduct:  original_dim = dim - 20
  meta_.set_meta(data_type_, original_dim_ + EXTRA_META_SIZE_INT8);

  ailego::Params metric_params;
  metric_params.set("proxima.quantized_integer.metric.origin_metric_name",
                    meta.metric_name());
  metric_params.set("proxima.quantized_integer.metric.origin_metric_params",
                    meta.metric_params());
  meta_.set_metric("QuantizedInteger", 0, metric_params);

  return 0;
}

int Int8Quantizer::quantize(const void *record,
                            const core::IndexQueryMeta & /*rmeta*/,
                            std::string *out,
                            core::IndexQueryMeta *ometa) const {
  const float *src = reinterpret_cast<const float *>(record);

  out->resize(meta_.element_size(), 0);
  core::RecordQuantizer::quantize_record(src, original_dim_,
                                         core::IndexMeta::DataType::DT_INT8,
                                         false, &(*out)[0]);

  *ometa = core::IndexQueryMeta(core::IndexMeta::DataType::DT_INT8,
                                meta_.dimension());
  return 0;
}

int Int8Quantizer::dequantize(const void *in,
                              const core::IndexQueryMeta & /*qmeta*/,
                              std::string *out) const {
  out->resize(original_dim_ * sizeof(float));
  float *dst = reinterpret_cast<float *>(&(*out)[0]);

  core::RecordQuantizer::unquantize_record(
      in, original_dim_, core::IndexMeta::DataType::DT_INT8, dst);

  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(Int8Quantizer);

}  // namespace turbo
}  // namespace zvec