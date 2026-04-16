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
#include <zvec/turbo/turbo.h>

#pragma once

namespace zvec {
namespace turbo {

class Quantizer {
 public:
  Quantizer() {};
  virtual ~Quantizer() {};

 private:
  QuantizeType type_{QuantizeType::kDefault};
};

}  // namespace turbo
}  // namespace zvec
