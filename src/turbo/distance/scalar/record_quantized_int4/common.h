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

#include <cstdint>
#include <zvec/ailego/internal/platform.h>

namespace zvec::turbo::scalar::internal {

/*! Four-bits Integer Multiplication Table
 */
static const AILEGO_ALIGNED(64) int8_t Int4MulTable[256] = {
    0, 0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0, 1,  2,   3,   4,   5,   6,   7,   -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,
    0, 2,  4,   6,   8,   10,  12,  14,  -16, -14, -12, -10, -8,  -6,  -4,  -2,
    0, 3,  6,   9,   12,  15,  18,  21,  -24, -21, -18, -15, -12, -9,  -6,  -3,
    0, 4,  8,   12,  16,  20,  24,  28,  -32, -28, -24, -20, -16, -12, -8,  -4,
    0, 5,  10,  15,  20,  25,  30,  35,  -40, -35, -30, -25, -20, -15, -10, -5,
    0, 6,  12,  18,  24,  30,  36,  42,  -48, -42, -36, -30, -24, -18, -12, -6,
    0, 7,  14,  21,  28,  35,  42,  49,  -56, -49, -42, -35, -28, -21, -14, -7,
    0, -8, -16, -24, -32, -40, -48, -56, 64,  56,  48,  40,  32,  24,  16,  8,
    0, -7, -14, -21, -28, -35, -42, -49, 56,  49,  42,  35,  28,  21,  14,  7,
    0, -6, -12, -18, -24, -30, -36, -42, 48,  42,  36,  30,  24,  18,  12,  6,
    0, -5, -10, -15, -20, -25, -30, -35, 40,  35,  30,  25,  20,  15,  10,  5,
    0, -4, -8,  -12, -16, -20, -24, -28, 32,  28,  24,  20,  16,  12,  8,   4,
    0, -3, -6,  -9,  -12, -15, -18, -21, 24,  21,  18,  15,  12,  9,   6,   3,
    0, -2, -4,  -6,  -8,  -10, -12, -14, 16,  14,  12,  10,  8,   6,   4,   2,
    0, -1, -2,  -3,  -4,  -5,  -6,  -7,  8,   7,   6,   5,   4,   3,   2,   1,
};

static __attribute__((always_inline)) void inner_product_int4_scalar(
    const void *a, const void *b, size_t dim, float *distance) {
  const uint8_t *m = reinterpret_cast<const uint8_t *>(a);
  const uint8_t *q = reinterpret_cast<const uint8_t *>(b);

  float sum = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    uint8_t m_val = m[i];
    uint8_t q_val = q[i];
    sum += Int4MulTable[((m_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
           Int4MulTable[((m_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
  }

  *distance = sum;
}

}  // namespace zvec::turbo::scalar::internal