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

#include <ailego/internal/cpu_features.h>
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__ARM_NEON)
float InnerProductNEON(const float *lhs, const float *rhs, size_t size);
void InnerProductNEON_2X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_2X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_4X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_4X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_4X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_8X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_8X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_8X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_8X8(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_16X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_16X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_16X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_16X8(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_16X16(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_32X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_32X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_32X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_32X8(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_32X16(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductNEON_32X32(const float *lhs, const float *rhs, size_t size, float *out);

float MinusInnerProductNEON(const float *lhs, const float *rhs, size_t size);
void MinusInnerProductNEON_2X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_2X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_4X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_4X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_4X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_8X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_8X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_8X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_8X8(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_16X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_16X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_16X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_16X8(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_16X16(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_32X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_32X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_32X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_32X8(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_32X16(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductNEON_32X32(const float *lhs, const float *rhs, size_t size, float *out);
#endif

#if defined(__AVX512F__)
float InnerProductAVX512(const float *lhs, const float *rhs, size_t size);

void InnerProductAVX512_16X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX512_16X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX512_16X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX512_16X8(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX512_16X16(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX512_32X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX512_32X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX512_32X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX512_32X8(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX512_32X16(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX512_32X32(const float *lhs, const float *rhs, size_t size, float *out);

void MinusInnerProductAVX512_16X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX512_16X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX512_16X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX512_16X8(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX512_16X16(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX512_32X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX512_32X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX512_32X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX512_32X8(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX512_32X16(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX512_32X32(const float *lhs, const float *rhs, size_t size, float *out);
#endif

#if defined(__AVX__)
float InnerProductAVX(const float *lhs, const float *rhs, size_t size);
void InnerProductAVX_2X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_2X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_4X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_4X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_4X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_8X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_8X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_8X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_8X8(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_16X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_16X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_16X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_16X8(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_16X16(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_32X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_32X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_32X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_32X8(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_32X16(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductAVX_32X32(const float *lhs, const float *rhs, size_t size, float *out);

float MinusInnerProductAVX(const float *lhs, const float *rhs, size_t size);
void MinusInnerProductAVX_2X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_2X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_4X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_4X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_4X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_8X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_8X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_8X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_8X8(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_16X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_16X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_16X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_16X8(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_16X16(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_32X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_32X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_32X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_32X8(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_32X16(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductAVX_32X32(const float *lhs, const float *rhs, size_t size, float *out);
#endif

#if defined(__SSE__)
float InnerProductSSE(const float *lhs, const float *rhs, size_t size);
void InnerProductSSE_2X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_2X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_4X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_4X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_4X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_8X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_8X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_8X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_8X8(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_16X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_16X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_16X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_16X8(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_16X16(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_32X1(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_32X2(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_32X4(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_32X8(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_32X16(const float *lhs, const float *rhs, size_t size, float *out);
void InnerProductSSE_32X32(const float *lhs, const float *rhs, size_t size, float *out);

float MinusInnerProductSSE(const float *lhs, const float *rhs, size_t size);
void MinusInnerProductSSE_2X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_2X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_4X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_4X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_4X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_8X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_8X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_8X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_8X8(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_16X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_16X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_16X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_16X8(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_16X16(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_32X1(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_32X2(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_32X4(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_32X8(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_32X16(const float *lhs, const float *rhs, size_t size, float *out);
void MinusInnerProductSSE_32X32(const float *lhs, const float *rhs, size_t size, float *out);
#endif 

#if defined(__SSE__) || defined(__ARM_NEON)
//! Compute the distance between matrix and query (FP32, M=1, N=1)
void InnerProductMatrix<float, 1, 1>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    if (dim > 15) {
      *out = InnerProductAVX512(m, q, dim);
      return;
    }
  }
#endif  // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    if (dim > 7) {
      *out = InnerProductAVX(m, q, dim);
      return;
    }
  }
#endif  // __AVX__
  *out = InnerProductSSE(m, q, dim);
#endif  // __ARM_NEON
}

//! Compute the distance between matrix and query (FP32, M=2, N=1)
void InnerProductMatrix<float, 2, 1>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_2X1(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_2X1(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_2X1(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=2, N=2)
void InnerProductMatrix<float, 2, 2>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_2X2(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_2X2(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_2X2(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=4, N=1)
void InnerProductMatrix<float, 4, 1>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_4X1(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_4X1(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_4X1(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=4, N=2)
void InnerProductMatrix<float, 4, 2>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_4X2(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_4X2(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_4X2(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=4, N=4)
void InnerProductMatrix<float, 4, 4>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_4X4(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_4X4(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_4X4(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=1)
void InnerProductMatrix<float, 8, 1>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_8X1(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_8X1(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_8X1(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=2)
void InnerProductMatrix<float, 8, 2>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_8X2(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_8X2(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_8X2(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=4)
void InnerProductMatrix<float, 8, 4>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_8X4(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_8X4(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_8X4(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=8)
void InnerProductMatrix<float, 8, 8>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_8X8(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_8X8(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_8X8(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=1)
void InnerProductMatrix<float, 16, 1>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_16X1(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_16X1(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_16X1(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_16X1(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=2)
void InnerProductMatrix<float, 16, 2>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_16X2(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_16X2(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_16X2(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_16X2(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=4)
void InnerProductMatrix<float, 16, 4>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_16X4(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_16X4(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_16X4(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_16X4(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=8)
void InnerProductMatrix<float, 16, 8>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_16X8(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_16X8(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_16X8(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_16X8(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=16)
void InnerProductMatrix<float, 16, 16>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_16X1(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_16X16(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_16X16(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_16X16(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=1)
void InnerProductMatrix<float, 32, 1>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_32X1(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_32X1(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_32X1(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_32X1(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=2)
void InnerProductMatrix<float, 32, 2>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_32X2(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_32X2(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_32X2(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_32X2(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=4)
void InnerProductMatrix<float, 32, 4>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_32X4(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_32X4(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_32X4(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_32X4(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=8)
void InnerProductMatrix<float, 32, 8>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_32X8(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_32X8(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_32X8(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_32X8(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=16)
void InnerProductMatrix<float, 32, 16>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_32X16(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_32X16(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_32X16(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_32X16(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=32)
void InnerProductMatrix<float, 32, 32>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__ARM_NEON)
  InnerProductNEON_32X32(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_32X32(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    InnerProductAVX_32X32(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  InnerProductSSE_32X32(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=1, N=1)
void MinusInnerProductMatrix<float, 1, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = -InnerProductNEON(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    if (dim > 15) {
      *out = -InnerProductAVX512(m, q, dim);
      return;
    }
  }
#endif  // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    if (dim > 7) {
      *out = -InnerProductAVX(m, q, dim);
      return;
    }
  }
#endif  // __AVX__
  *out = -InnerProductSSE(m, q, dim);
#endif  // __ARM_NEON
}

//! Compute the distance between matrix and query (FP32, M=2, N=1)
void MinusInnerProductMatrix<float, 2, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_2X1(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_2X1(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_2X1(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=2, N=2)
void MinusInnerProductMatrix<float, 2, 2>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_2X2(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_2X2(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_2X2(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=4, N=1)
void MinusInnerProductMatrix<float, 4, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_4X1(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_4X1(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_4X1(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=4, N=2)
void MinusInnerProductMatrix<float, 4, 2>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_4X2(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_4X2(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_4X2(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=4, N=4)
void MinusInnerProductMatrix<float, 4, 4>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_4X4(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_4X4(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_4X4(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=1)
void MinusInnerProductMatrix<float, 8, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_8X1(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_8X1(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_8X1(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=2)
void MinusInnerProductMatrix<float, 8, 2>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_8X2(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_8X2(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_8X2(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=4)
void MinusInnerProductMatrix<float, 8, 4>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_8X4(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_8X4(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_8X4(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=8)
void MinusInnerProductMatrix<float, 8, 8>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_8X8(m, q, dim, out);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_8X8(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_8X8(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=1)
void MinusInnerProductMatrix<float, 16, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_16X1(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_16X1(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_16X1(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_16X1(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=2)
void MinusInnerProductMatrix<float, 16, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_16X2(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_16X2(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_16X2(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_16X2(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=4)
void MinusInnerProductMatrix<float, 16, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_16X4(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_16X4(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_16X4(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_16X4(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=8)
void MinusInnerProductMatrix<float, 16, 8>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_16X8(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_16X8(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_16X8(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_16X8(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=16)
void MinusInnerProductMatrix<float, 16, 16>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_16X16(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_16X16(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_16X16(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_16X16(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=1)
void MinusInnerProductMatrix<float, 32, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_32X1(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_32X1(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_32X1(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_32X1(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=2)
void MinusInnerProductMatrix<float, 32, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_32X2(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_32X2(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_32X2(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_32X2(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=4)
void MinusInnerProductMatrix<float, 32, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_32X4(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_32X4(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_32X4(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_32X4(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=8)
void MinusInnerProductMatrix<float, 32, 8>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_32X8(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_32X8(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_32X8(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_32X8(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=16)
void MinusInnerProductMatrix<float, 32, 16>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_32X16(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_32X16(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_32X16(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_32X16(m, q, dim, out);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=32)
void MinusInnerProductMatrix<float, 32, 32>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  MinusInnerProductNEON_32X32(m, q, dim, out);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_32X32(m, q, dim, out);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    MinusInnerProductAVX_32X32(m, q, dim, out);
    return;
  }
#endif  // __AVX__
  MinusInnerProductSSE_32X32(m, q, dim, out);
#endif
}

#endif
}  // namespace ailego
}  // namespace zvec
