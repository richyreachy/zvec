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

#include "avx/float32/inner_product.h"
#include "avx/float32/common.h"

#if defined(__AVX__)
#include <immintrin.h>
#endif

namespace zvec::turbo::avx {

// Compute squared Euclidean distance between a single quantized FP16
// vector pair.
void inner_product_fp16_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
#if defined(__AVX__)
  ACCUM_FP16_1X1_AVX(lhs, rhs, size, distance, 0ull, )
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif  // __AVX__
}

// Batch version of inner_product_fp16_distance.
void inner_product_fp16_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances) {
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
}

}  // namespace zvec::turbo::avx