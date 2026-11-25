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

#include <zvec/core/framework/index_meta.h>

namespace zvec::core {

bool QuantizedGraphSupported();

inline bool QuantizedGraphSupportsMeta(const IndexMeta &m) {
  return m.meta_type() == IndexMeta::MT_DENSE &&
         m.data_type() == IndexMeta::DT_FP32 && m.dimension() &&
         m.dimension() <= 4096 &&
         m.element_size() == size_t(m.dimension()) * sizeof(float) &&
         (m.metric_name() == "Cosine" || m.metric_name() == "SquaredEuclidean");
}

}  // namespace zvec::core
