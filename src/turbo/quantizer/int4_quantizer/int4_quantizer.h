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

#pragma once

#include <zvec/ailego/algorithm/integer_quantizer.h>
#include <zvec/core/framework/index_converter.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/core/framework/index_stats.h>
#include "quantizer/quantizer.h"

namespace zvec {
namespace turbo {

using namespace zvec::core;

class Int4Quantizer : public Quantizer {
 public:
  Int4Quantizer() {
    type_ = QuantizeType::kRecordInt4;
  }

  virtual ~Int4Quantizer() {}

 public:
  QuantizeType type() const override {
    return type_;
  }

  int init(const IndexMeta &meta, const ailego::Params &params) override;

  const IndexMeta &meta(void) const override {
    return meta_;
  }

  int quantize(const void *query, const IndexQueryMeta &qmeta, std::string *out,
               IndexQueryMeta *ometa) const override;

  int dequantize(const void *in, const IndexQueryMeta &qmeta,
                 std::string *out) const override;

 private:
  static constexpr uint32_t EXTRA_META_SIZE = 20;
  const std::string INT4_QUANTIZER_BIAS = "int4_quantizer.bias";
  const std::string INT4_QUANTIZER_SCALE = "int4_quantizer.scale";

  float bias_{0.0f};
  float scale_{1.0f};
  float scale_reiprocal_{1.0f};

  ailego::EntropyInt8Quantizer quantizer_;
  IndexMeta meta_{};
  uint32_t original_dim_{0};
  IndexMeta::DataType data_type_{};
};


}  // namespace turbo
}  // namespace zvec
