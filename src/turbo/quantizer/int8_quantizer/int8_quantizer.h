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

#include <ailego/algorithm/integer_quantizer.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/core/framework/index_stats.h>
#include "quantizer/quantizer.h"

namespace zvec {
namespace turbo {

using namespace zvec::core;

class Int8Quantizer : public Quantizer {
 public:
  Int8Quantizer() {
    type_ = QuantizeType::kRecordInt8;
  }

  virtual ~Int8Quantizer() {}

 public:
  QuantizeType type() const override {
    return type_;
  }

  int init(const core::IndexMeta &meta, const ailego::Params &params) override;

  int train(core::IndexHolder::Pointer holder) const override;

  const core::IndexMeta &meta(void) const override {
    return meta_;
  }

  int quantize(const void *query, const core::IndexQueryMeta &qmeta,
               std::string *out, core::IndexQueryMeta *ometa) const override;

  int dequantize(const void *in, const core::IndexQueryMeta &qmeta,
                 std::string *out) const override;

 private:
  static constexpr uint32_t EXTRA_META_SIZE_INT8 = 20;
  const std::string INT8_QUANTIZER_BIAS = "int8_quantizer.bias";
  const std::string INT8_QUANTIZER_SCALE = "int8_quantizer.scale";

  float bias_{0.0f};
  float scale_{1.0f};
  float scale_reciprocal_{1.0f};
  bool inner_product_{false};

  mutable ailego::EntropyInt8Quantizer quantizer_;
  IndexMeta meta_{};
  uint32_t original_dim_{0};
  IndexMeta::DataType data_type_{};
};


}  // namespace turbo
}  // namespace zvec
