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

#include "avx/float32/cosine.h"
#include "avx/float32/common.h"
#include "avx/float32/inner_product.h"

#if defined(__AVX__)
#include <immintrin.h>
#endif

namespace zvec::turbo::avx {

void cosine_fp16_distance(const void *a, const void *b, size_t dim,
                          float *distance) {
#if defined(__AVX__)
  constexpr size_t extra_dim = 2;
  size_t d = dim - extra_dim;

  float ip;
  inner_product_fp16_avx(m, q, d, &ip);

  *out = 1 - ip;
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif  // __AVX__
}

void cosine_fp16_batch_distance(const void *const *vectors, const void *query,
                                size_t n, size_t dim, float *distances) {
#if defined(__AVX__)

#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif  //__AVX__
}

}  // namespace zvec::turbo::avx