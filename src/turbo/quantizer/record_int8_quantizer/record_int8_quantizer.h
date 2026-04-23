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

using namespace zvec::core;

namespace zvec {
namespace turbo {

class RecordInt8Quantizer : public Quantizer {
 public:
  RecordInt8Quantizer() {
    type_ = QuantizeType::kRecordInt8;
  }

  virtual ~RecordInt8Quantizer() {}

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
  static constexpr uint32_t EXTRA_META_SIZE_INT8 = 20;
  static constexpr uint32_t EXTRA_META_SIZE_COSINE = 4;

  bool cosine_{false};
  uint32_t extra_meta_size_{0};

  uint32_t original_dim_{0};
  IndexHolder::Pointer holder_{};
  IndexMeta meta_{};
  IndexMeta::DataType data_type_{};
};


}  // namespace turbo
}  // namespace zvec
