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

#include "scalar/fp32/cosine.h"
#include <cmath>

namespace zvec::turbo::scalar {

void cosine_fp32_distance(const void *a, const void *b, size_t dim,
                          float *distance) {
  const float *fa = static_cast<const float *>(a);
  const float *fb = static_cast<const float *>(b);
  float dot = 0.0f;
  float na = 0.0f;
  float nb = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    dot += fa[i] * fb[i];
    na += fa[i] * fa[i];
    nb += fb[i] * fb[i];
  }
  float denom = std::sqrt(na) * std::sqrt(nb);
  if (denom < 1e-12f) {
    *distance = 1.0f;
  } else {
    *distance = 1.0f - dot / denom;
  }
}

void cosine_fp32_batch_distance(const void *const *vectors, const void *query,
                                size_t n, size_t dim, float *distances) {
  for (size_t i = 0; i < n; ++i) {
    cosine_fp32_distance(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::scalar