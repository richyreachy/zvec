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
#include "armv8/float32/squared_euclidean.h"
#include "armv8/float32/squared_euclidean_common.h"

using namespace zvec::turbo::armv8::internal;
#endif

namespace zvec::turbo::armv8 {

void squared_euclidean_fp32_distance(const void *a, const void *b, size_t dim,
                                     float *distance) {
#if defined(__ARM_NEON)
  squared_euclidean_fp32_armv8(a, b, dim, distance);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif  // __ARM_NEON
}

void squared_euclidean_fp32_batch_distance(const void *const *vectors,
                                           const void *query, size_t n,
                                           size_t dim, float *distances) {
#if defined(__ARM_NEON)
  squared_euclidean_fp32_batch_armv8(vectors, query, n, dim, distances);
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif  //
}

}  // namespace zvec::turbo::armv8