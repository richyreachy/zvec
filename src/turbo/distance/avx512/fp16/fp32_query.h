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

#include <cstddef>

namespace zvec::turbo::avx512 {

// FP16 stored vector, original FP32 query, and FP32 accumulation. Cosine
// computes 1 - dot, so normalization belongs to the caller, as for FP32.
void cosine_fp16_fp32_distance_avx512(const void *vector, const float *query,
                                      size_t dim, float *distance);
void squared_euclidean_fp16_fp32_distance_avx512(const void *vector,
                                                 const float *query, size_t dim,
                                                 float *distance);

}  // namespace zvec::turbo::avx512
