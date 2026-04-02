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

#include "avx512_fp16/half_float/cosine.h"
#include "avx512_fp16/half_float/inner_product.h"
#include "avx512_fp16/half_float/inner_product_common.h"

#if defined(__AVX512FP16__)
#include <immintrin.h>
#endif

namespace zvec::turbo::avx512_fp16 {

void cosine_fp16_distance(const void *a, const void *b, size_t dim,
                          float *distance) {
#if defined(__AVX512FP16__)
  constexpr size_t extra_dim = 2;
  size_t original_dim = dim - extra_dim;

  float ip;
  inner_product_fp16_distance(a, b, original_dim, &ip);

  *distance = 1 - ip;
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif  // __AVX__
}

void cosine_fp16_batch_distance(const void *const *vectors, const void *query,
                                size_t n, size_t dim, float *distances) {
#if defined(__AVX512FP16__)

#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif  //__AVX__
}

}  // namespace zvec::turbo::avx512_fp16