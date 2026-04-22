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

#include "quantizer/fp32_quantizer/fp32_quantizer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>
#include "core/quantizer/record_quantizer.h"

namespace zvec {
namespace turbo {

int Fp32Quantizer::init(const IndexMeta &meta,
                        const ailego::Params & /*params*/) {
  meta_ = meta;

  meta_.set_meta(IndexMeta::DataType::DT_FP32, meta.dimension());

  auto metric_name = meta.metric_name();
  if (metric_name == "Cosine") {
    meta_.set_extra_meta_size(EXTRA_META_SIZE_COSINE);
  }

  return 0;
}

int Fp32Quantizer::quantize(const void *query, const IndexQueryMeta &qmeta,
                            std::string *out, IndexQueryMeta *ometa) const {
  if (qmeta.unit_size() != sizeof(float)) {
    return IndexError_Unsupported;
  }

  size_t byte_size = qmeta.dimension() * sizeof(float);
  out->resize(byte_size);
  std::memcpy(&(*out)[0], query, byte_size);

  *ometa = qmeta;
  ometa->set_meta(IndexMeta::DataType::DT_FP32, qmeta.dimension());

  return 0;
}

int Fp32Quantizer::dequantize(const void *in, const IndexQueryMeta &qmeta,
                              std::string *out) const {
  size_t byte_size = qmeta.dimension() * sizeof(float);
  out->resize(byte_size);
  std::memcpy(out->data(), in, byte_size);
  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(Fp32Quantizer);

}  // namespace turbo
}  // namespace zvec
