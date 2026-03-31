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
#include <zvec/ailego/internal/platform.h>

namespace zvec::turbo::sse::internal {

//! Four-bits Convert Table
static const AILEGO_ALIGNED(32) int8_t Int4ConvertTable[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};

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

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_INT4_GENERAL(m, q, sum)                               \
  sum += Int4MulTable[(((m) << 4) & 0xf0) | (((q) >> 0) & 0xf)] + \
         Int4MulTable[(((m) >> 0) & 0xf0) | (((q) >> 4) & 0xf)];

#define MASK_INT4_SSE _mm_set1_epi32(0x0f0f0f0f)
#define ONES_INT16_SSE _mm_set1_epi32(0x00010001)
#define INT4_LOOKUP_SSE _mm_load_si128((const __m128i *)Int4ConvertTable)

//! Compute the distance between matrix and query
#define FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)                       \
  {                                                                        \
    __m128i xmm_lhs_0 = _mm_shuffle_epi8(                                  \
        INT4_LOOKUP_SSE, _mm_and_si128((xmm_lhs), MASK_INT4_SSE));         \
    __m128i xmm_rhs_0 = _mm_shuffle_epi8(                                  \
        INT4_LOOKUP_SSE, _mm_and_si128((xmm_rhs), MASK_INT4_SSE));         \
    __m128i xmm_lhs_1 = _mm_shuffle_epi8(                                  \
        INT4_LOOKUP_SSE,                                                   \
        _mm_and_si128(_mm_srli_epi32((xmm_lhs), 4), MASK_INT4_SSE));       \
    __m128i xmm_rhs_1 = _mm_shuffle_epi8(                                  \
        INT4_LOOKUP_SSE,                                                   \
        _mm_and_si128(_mm_srli_epi32((xmm_rhs), 4), MASK_INT4_SSE));       \
    xmm_lhs_0 = _mm_sign_epi8(xmm_lhs_0, xmm_rhs_0);                       \
    xmm_lhs_1 = _mm_sign_epi8(xmm_lhs_1, xmm_rhs_1);                       \
    xmm_rhs_0 = _mm_abs_epi8(xmm_rhs_0);                                   \
    xmm_rhs_1 = _mm_abs_epi8(xmm_rhs_1);                                   \
    xmm_lhs_0 = _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs_0, xmm_lhs_0),    \
                               ONES_INT16_SSE);                            \
    xmm_lhs_1 = _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs_1, xmm_lhs_1),    \
                               ONES_INT16_SSE);                            \
    xmm_sum = _mm_add_epi32(_mm_add_epi32(xmm_lhs_0, xmm_lhs_1), xmm_sum); \
  }

static inline int32_t HorizontalAdd_INT32_V128(__m128i v) {
#ifdef __SSE3__
  __m128i x1 = _mm_hadd_epi32(v, v);
  __m128i x2 = _mm_hadd_epi32(x1, x1);
  return _mm_cvtsi128_si32(x2);
#else
  __m128i x1 = _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 0, 3, 2));
  __m128i x2 = _mm_add_epi32(v, x1);
  __m128i x3 = _mm_shuffle_epi32(x2, _MM_SHUFFLE(0, 0, 0, 1));
  __m128i x4 = _mm_add_epi32(x2, x3);
  return _mm_cvtsi128_si32(x4);
#endif
}

static __attribute__((always_inline)) void inner_product_int4_sse(
    const void *a, const void *b, size_t size, float *distance) {
  const uint8_t *lhs = reinterpret_cast<const uint8_t *>(a);
  const uint8_t *rhs = reinterpret_cast<const uint8_t *>(b);

  const uint8_t *last = lhs + size;
  const uint8_t *last_aligned = lhs + ((size >> 4) << 4);
  __m128i xmm_sum = _mm_setzero_si128();

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)(lhs));
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)(rhs));
      FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)(lhs));
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)(rhs));
      FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)
    }
  }
  float result = static_cast<float>(HorizontalAdd_INT32_V128(xmm_sum));

  switch (last - lhs) {
    case 15:
      FMA_INT4_GENERAL(lhs[14], rhs[14], result)
      /* FALLTHRU */
    case 14:
      FMA_INT4_GENERAL(lhs[13], rhs[13], result)
      /* FALLTHRU */
    case 13:
      FMA_INT4_GENERAL(lhs[12], rhs[12], result)
      /* FALLTHRU */
    case 12:
      FMA_INT4_GENERAL(lhs[11], rhs[11], result)
      /* FALLTHRU */
    case 11:
      FMA_INT4_GENERAL(lhs[10], rhs[10], result)
      /* FALLTHRU */
    case 10:
      FMA_INT4_GENERAL(lhs[9], rhs[9], result)
      /* FALLTHRU */
    case 9:
      FMA_INT4_GENERAL(lhs[8], rhs[8], result)
      /* FALLTHRU */
    case 8:
      FMA_INT4_GENERAL(lhs[7], rhs[7], result)
      /* FALLTHRU */
    case 7:
      FMA_INT4_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      FMA_INT4_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      FMA_INT4_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      FMA_INT4_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      FMA_INT4_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_INT4_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_INT4_GENERAL(lhs[0], rhs[0], result)
  }

  *distance = result;
}

}  // namespace zvec::turbo::sse::internal

#endif  // defined(__SSE4_1__)
