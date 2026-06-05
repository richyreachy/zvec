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

// This file is compiled with per-file -march=avx512vnni (set in CMakeLists.txt)
// so that all AVX512-VNNI intrinsics and the inlined inner product kernels from
// common.h are compiled with the correct target ISA.

#include "avx512_vnni/record_quantized_int8/cosine.h"
#include "avx512_vnni/record_quantized_int8/common.h"
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>
#endif
#include <cstring>
#include <vector>

// Tail layout for quantized INT8 cosine vectors:
//
//   [ original_dim bytes: int8_t elements ]
//   [ float scale_a       ]  (ma)
//   [ float bias_a        ]  (mb)
//   [ float sum_a         ]  (ms)
//   [ float square_sum_a  ]  (ms2)
//   [ int  int8_sum  ]  (sum of raw int8 elements, used when query is
//                        preprocessed to uint8 via +128 shift)
//
// The query tail has the same layout (qa, qb, qs, qs2) without int8_sum.

namespace zvec::turbo::avx512_vnni {

void cosine_int8_distance(const void *a, const void *b, size_t dim,
                          float *distance) {
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  // `dim` is the original_dim (the wrapper has already subtracted the 24-byte
  // metadata tail). The single-distance contract here is symmetric: both `a`
  // and `b` are raw int8 vectors (used both for query-vs-node and
  // node-vs-node distances). Query preprocessing for the batch path is
  // applied separately in `wrap_turbo_batch_distance`, not here.
  const int original_dim = dim;
  if (original_dim <= 0) {
    return;
  }

  internal::ip_int8_avx512_vnni(a, b, original_dim, distance);

  const float *a_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(a) + original_dim);
  const float *b_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(b) + original_dim);

  float ma = a_tail[0];
  float mb = a_tail[1];
  float ms = a_tail[2];

  float qa = b_tail[0];
  float qb = b_tail[1];
  float qs = b_tail[2];

  // Dequantize and compute cosine distance numerator (-<float_a, float_b>).
  // The metric layer's normalize() adds 1.0f to yield 1 - cos_sim.
  *distance = -(ma * qa * *distance + mb * qa * qs + qb * ma * ms +
                static_cast<float>(original_dim) * qb * mb);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void cosine_int8_batch_distance(const void *const *vectors, const void *query,
                                size_t n, size_t dim, float *distances) {
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  // `dim` is the original_dim (the wrapper has already subtracted the 24-byte
  // metadata tail). The query is passed in as RAW int8; we shift the data
  // bytes by +128 into a thread-local buffer so dpbusd (uint8 * int8) yields
  // the correct integer inner product. The corresponding `128 * sum(int8_a)`
  // bias is removed below using the precomputed `int8_sum` per stored vector.
  const int original_dim = dim;
  if (original_dim <= 0) {
    return;
  }

  // Shift the data portion of the query (+128) into a thread-local buffer so
  // dpbusd (uint8 * int8) yields the correct integer inner product. The query
  // metadata tail (3 floats: qa/qb/qs) is read directly from the caller's
  // query buffer below to avoid touching memory we don't need to.
  thread_local std::vector<uint8_t> query_buf;
  const size_t data_bytes = static_cast<size_t>(original_dim);
  if (query_buf.size() < data_bytes) {
    query_buf.resize(data_bytes);
  }
  std::memcpy(query_buf.data(), query, data_bytes);
  internal::shift_int8_to_uint8_avx512(query_buf.data(), original_dim);

  // Compute raw inner products for all vectors using dpbusd (uint8 * int8).
  internal::ip_int8_batch_avx512_vnni(vectors, query_buf.data(), n,
                                      original_dim, distances);

  const float *q_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(query) + original_dim);
  float qa = q_tail[0];
  float qb = q_tail[1];
  float qs = q_tail[2];

  for (size_t i = 0; i < n; ++i) {
    const float *m_tail = reinterpret_cast<const float *>(
        reinterpret_cast<const int8_t *>(vectors[i]) + original_dim);
    float ma = m_tail[0];
    float mb = m_tail[1];
    float ms = m_tail[2];
    // Correct for the +128 shift applied to the query during preprocessing:
    //   dpbusd computes sum(uint8_query[i] * int8_data[i])
    //         = sum((int8_query[i] + 128) * int8_data[i])
    //         = true_ip + 128 * sum(int8_data[i])
    // int8_sum is stored as the 5th int-sized field after the 4 floats.
    int int8_sum = reinterpret_cast<const int *>(m_tail)[4];
    float &result = distances[i];
    result -= 128.0f * static_cast<float>(int8_sum);

    // Dequantize and compute cosine distance numerator (-<float_a, float_b>).
    // The metric layer's normalize() adds 1.0f to yield 1 - cos_sim.
    result = -(ma * qa * result + mb * qa * qs + qb * ma * ms +
               static_cast<float>(original_dim) * qb * mb);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

void cosine_int8_query_preprocess(void *query, size_t dim) {
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  // The original vector occupies dim-24 bytes; only those bytes are shifted.
  const int original_dim = static_cast<int>(dim);
  if (original_dim <= 0) {
    return;
  }
  internal::shift_int8_to_uint8_avx512(query, original_dim);
#else
  (void)query;
  (void)dim;
#endif
}

}  // namespace zvec::turbo::avx512_vnni
