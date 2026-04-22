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

#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/core/framework/index_stats.h>
#include "quantizer/quantizer.h"

namespace zvec {
namespace turbo {

using namespace zvec::core;

class Fp32Quantizer : public Quantizer {
 public:
  Fp32Quantizer() {
    type_ = QuantizeType::kRecordInt8;
  }

  virtual ~Fp32Quantizer() {}

 public:
  QuantizeType type() const override {
    return type_;
  }

  int init(const core::IndexMeta &meta, const ailego::Params &params) override;

  int train(core::IndexHolder::Pointer /*holder*/) override {
    return 0;
  }

  const core::IndexMeta &meta(void) const override {
    return meta_;
  }

  int quantize(const void *query, const core::IndexQueryMeta &qmeta,
               std::string *out, core::IndexQueryMeta *ometa) const override;

  int dequantize(const void *in, const core::IndexQueryMeta &qmeta,
                 std::string *out) const override;

 private:
  static constexpr uint32_t EXTRA_META_SIZE_COSINE = 4;

  IndexMeta meta_{};
  uint32_t original_dim_{0};
  IndexMeta::DataType data_type_{};
};


}  // namespace turbo
}  // namespace zvec
