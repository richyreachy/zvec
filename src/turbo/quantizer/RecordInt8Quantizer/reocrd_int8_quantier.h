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

#include <zvec/core/framework/index_meta.h>
#include <zvec/turbo/quantizer/quantizer.h>

#pragma once

namespace zvec {
namespace turbo {

class RecordInt8Quantizer : public Quantizer {
 public:
  RecordInt8Quantizer() : type_{QuantizeType::kRecordInt8} {}

  virtual ~RecordInt8Quantizer() {}

 public:
  QuantizeType type() const override {
    return type_;
  }

  const IndexMeta &meta(void) const override {
    return meta_;
  }

 private:
  IndexMeta meta_{};
  IndexHolder::Pointer holder_{};
  std::shared_ptr<Quantizer> quantizer_{};
  Stats stats_{};
  IndexMeta::DataType data_type_{};
};


}  // namespace turbo
}  // namespace zvec
