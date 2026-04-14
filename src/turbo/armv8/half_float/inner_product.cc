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

#include <cstddef>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#include <zvec/ailego/utility/float_helper.h>
#include "armv8/half_float/inner_product.h"
#include "armv8/half_float/inner_product_common.h"

using namespace zvec::turbo::armv8::internal;
#endif

namespace zvec::turbo::armv8 {

// Compute squared Euclidean distance between a single quantized FP16
// vector pair.
void inner_product_fp16_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
#if defined(__ARM_NEON)
  const zvec::ailego::Float16 *lhs =
      reinterpret_cast<const zvec::ailego::Float16 *>(a);
  const zvec::ailego::Float16 *rhs =
      reinterpret_cast<const zvec::ailego::Float16 *>(b);

  ACCUM_FP16_1X1_NEON(lhs, rhs, dim, distance, 0ull, )

#endif
}

// Batch version of inner_product_fp16_distance.
void inner_product_fp16_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances) {
#if defined(__ARM_NEON)
  inner_product_fp16_batch_armv8(vectors, query, n, dim, distances);
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif  //__ARM_NEON
}

}  // namespace zvec::turbo::armv8
