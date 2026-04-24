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

#include "quantizer/fp16_quantizer/fp16_quantizer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>
#include "core/quantizer/record_quantizer.h"

namespace zvec {
namespace turbo {

int Fp16Quantizer::init(const IndexMeta &meta,
                        const ailego::Params & /*params*/) {
  meta_ = meta;

  meta_.set_meta(IndexMeta::DataType::DT_FP16, meta.dimension());

  auto metric_name = meta.metric_name();
  if (metric_name == "Cosine") {
    extra_meta_size_ = EXTRA_META_SIZE_COSINE;
    meta_.set_extra_meta_size(extra_meta_size_);
  }

  return 0;
}

int Fp16Quantizer::quantize(const void *query, const IndexQueryMeta &qmeta,
                            std::string *out, IndexQueryMeta *ometa) const {
  if (qmeta.unit_size() != sizeof(float)) {
    return IndexError_Unsupported;
  }
  out->resize(qmeta.dimension() * sizeof(ailego::Float16));
  ailego::FloatHelper::ToFP16(reinterpret_cast<const float *>(query),
                              qmeta.dimension(),
                              reinterpret_cast<uint16_t *>(&(*out)[0]));
  *ometa = qmeta;
  ometa->set_meta(IndexMeta::DataType::DT_FP16, qmeta.dimension(),
                  static_cast<uint32_t>(type_), extra_meta_size_);

  return 0;
}

int Fp16Quantizer::dequantize(const void *in, const IndexQueryMeta &qmeta,
                              std::string *out) const {
  if (qmeta.data_type() == IndexMeta::DataType::DT_FP16) {
    size_t dimension = qmeta.dimension();

    out->resize(dimension * sizeof(float));
    float *out_buf = reinterpret_cast<float *>(out->data());

    const uint16_t *in_buf = reinterpret_cast<const uint16_t *>(in);
    for (size_t i = 0; i < dimension; ++i) {
      out_buf[i] = ailego::FloatHelper::ToFP32(in_buf[i]);
    }
  }

  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(Fp16Quantizer);

}  // namespace turbo
}  // namespace zvec