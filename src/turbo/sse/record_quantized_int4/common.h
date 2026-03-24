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

// Shared AVX512-VNNI inner product kernels for record_quantized_int8 distance
// implementations (cosine, l2, mips_l2, etc.).
//
// All functions are marked always_inline so that when this header is included
// from a per-file-march .cc translation unit, the compiler can fully inline
// and optimize them under the correct -march flag without any cross-TU call
// overhead.

#pragma once

#if defined(__SSE4_1__)
#include <immintrin.h>
#include <array>
#include <cstdint>

namespace zvec::turbo::sse::internal {

static __attribute__((always_inline)) void ip_int4_sse(const void *a,
                                                       const void *b,
                                                       size_t size,
                                                       float *distance) {}

static __attribute__((always_inline)) void ip_int4_batch_sse(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances) {}

}  // namespace zvec::turbo::sse::internal

#endif  // defined(__SSE4_1__)
