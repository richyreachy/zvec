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
float InnerProductNEON_2X1(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_2X2(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_4X1(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_4X2(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_4X4(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_8X1(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_8X2(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_8X4(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_8X8(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_16X1(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_16X2(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_16X4(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_16X8(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_16X16(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_32X1(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_32X2(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_32X4(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_32X8(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_32X16(const float *lhs, const float *rhs, size_t size);
float InnerProductNEON_32X32(const float *lhs, const float *rhs, size_t size);

float MinusInnerProductNEON(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_2X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_2X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_4X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_4X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_4X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_8X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_8X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_8X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_8X8(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_16X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_16X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_16X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_16X8(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_16X16(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_32X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_32X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_32X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_32X8(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_32X16(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON_32X32(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__AVX512F__)
float InnerProductAVX512(const float *lhs, const float *rhs, size_t size);

float InnerProductAVX512_16X1(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX512_16X2(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX512_16X4(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX512_16X8(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX512_16X16(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX512_32X1(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX512_32X2(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX512_32X4(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX512_32X8(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX512_32X16(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX512_32X32(const float *lhs, const float *rhs, size_t size);

float MinusInnerProductAVX512_16X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX512_16X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX512_16X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX512_16X8(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX512_16X16(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX512_32X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX512_32X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX512_32X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX512_32X8(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX512_32X16(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX512_32X32(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__AVX__)
float InnerProductAVX(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_2X1(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_2X2(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_4X1(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_4X2(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_4X4(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_8X1(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_8X2(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_8X4(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_8X8(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_16X1(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_16X2(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_16X4(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_16X8(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_16X16(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_32X1(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_32X2(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_32X4(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_32X8(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_32X16(const float *lhs, const float *rhs, size_t size);
float InnerProductAVX_32X32(const float *lhs, const float *rhs, size_t size);

float MinusInnerProductAVX(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_2X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_2X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_4X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_4X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_4X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_8X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_8X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_8X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_8X8(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_16X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_16X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_16X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_16X8(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_16X16(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_32X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_32X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_32X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_32X8(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_32X16(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX_32X32(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__SSE__)
float InnerProductSSE(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_2X1(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_2X2(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_4X1(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_4X2(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_4X4(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_8X1(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_8X2(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_8X4(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_8X8(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_16X1(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_16X2(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_16X4(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_16X8(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_16X16(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_32X1(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_32X2(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_32X4(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_32X8(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_32X16(const float *lhs, const float *rhs, size_t size);
float InnerProductSSE_32X32(const float *lhs, const float *rhs, size_t size);

float MinusInnerProductSSE(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_2X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_2X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_4X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_4X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_4X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_8X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_8X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_8X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_8X8(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_16X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_16X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_16X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_16X8(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_16X16(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_32X1(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_32X2(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_32X4(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_32X8(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_32X16(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE_32X32(const float *lhs, const float *rhs, size_t size);
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
  *out = InnerProductNEON_2X1(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_2X1(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_2X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=2, N=2)
void InnerProductMatrix<float, 2, 2>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_2X2(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_2X2(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_2X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=4, N=1)
void InnerProductMatrix<float, 4, 1>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_4X1(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_4X1(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_4X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=4, N=2)
void InnerProductMatrix<float, 4, 2>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_4X2(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_4X2(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_4X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=4, N=4)
void InnerProductMatrix<float, 4, 4>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_4X4(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_4X4(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_4X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=1)
void InnerProductMatrix<float, 8, 1>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_8X1(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_8X1(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_8X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=2)
void InnerProductMatrix<float, 8, 2>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_8X2(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_8X2(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_8X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=4)
void InnerProductMatrix<float, 8, 4>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_8X4(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_8X4(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_8X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=8)
void InnerProductMatrix<float, 8, 8>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_8X8(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_8X8(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_8X8(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=1)
void InnerProductMatrix<float, 16, 1>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_16X1(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductAVX512_16X1(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_16X1(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_16X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=2)
void InnerProductMatrix<float, 16, 2>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_16X2(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductAVX512_16X2(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_16X2(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_16X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=4)
void InnerProductMatrix<float, 16, 4>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_16X4(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductAVX512_16X4(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_16X4(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_16X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=8)
void InnerProductMatrix<float, 16, 8>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_16X8(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductAVX512_16X8(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_16X8(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_16X8(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=16)
void InnerProductMatrix<float, 16, 16>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_16X1(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductAVX512_16X16(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_16X16(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_16X16(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=1)
void InnerProductMatrix<float, 32, 1>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_32X1(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductAVX512_32X1(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_32X1(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_32X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=2)
void InnerProductMatrix<float, 32, 2>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_32X2(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductAVX512_32X2(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_32X2(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_32X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=4)
void InnerProductMatrix<float, 32, 4>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_32X4(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductAVX512_32X4(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_32X4(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_32X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=8)
void InnerProductMatrix<float, 32, 8>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
                                               float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_32X8(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductAVX512_32X8(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_32X8(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_32X8(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=16)
void InnerProductMatrix<float, 32, 16>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_32X16(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductAVX512_32X16(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_32X16(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_32X16(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=32)
void InnerProductMatrix<float, 32, 32>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON_32X32(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductAVX512_32X32(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductAVX_32X32(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = InnerProductSSE_32X32(m, q, dim);
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
  *out = MinusInnerProductNEON_2X1(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_2X1(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_2X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=2, N=2)
void MinusInnerProductMatrix<float, 2, 2>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_2X2(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_2X2(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_2X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=4, N=1)
void MinusInnerProductMatrix<float, 4, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_4X1(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_4X1(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_4X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=4, N=2)
void MinusInnerProductMatrix<float, 4, 2>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_4X2(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_4X2(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_4X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=4, N=4)
void MinusInnerProductMatrix<float, 4, 4>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_4X4(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_4X4(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_4X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=1)
void MinusInnerProductMatrix<float, 8, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_8X1(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_8X1(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_8X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=2)
void MinusInnerProductMatrix<float, 8, 2>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_8X2(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_8X2(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_8X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=4)
void MinusInnerProductMatrix<float, 8, 4>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_8X4(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_8X4(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_8X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=8)
void MinusInnerProductMatrix<float, 8, 8>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_8X8(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_8X8(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_8X8(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=1)
void MinusInnerProductMatrix<float, 16, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_16X1(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductAVX512_16X1(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_16X1(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_16X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=2)
void MinusInnerProductMatrix<float, 16, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_16X2(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductAVX512_16X2(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_16X2(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_16X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=4)
void MinusInnerProductMatrix<float, 16, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_16X4(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductAVX512_16X4(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_16X4(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_16X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=8)
void MinusInnerProductMatrix<float, 16, 8>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_16X8(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductAVX512_16X8(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_16X8(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_16X8(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=16)
void MinusInnerProductMatrix<float, 16, 16>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_16X16(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductAVX512_16X16(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_16X16(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_16X16(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=1)
void MinusInnerProductMatrix<float, 32, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_32X1(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductAVX512_32X1(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_32X1(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_32X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=2)
void MinusInnerProductMatrix<float, 32, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_32X2(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductAVX512_32X2(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_32X2(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_32X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=4)
void MinusInnerProductMatrix<float, 32, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_32X4(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductAVX512_32X4(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_32X4(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_32X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=8)
void MinusInnerProductMatrix<float, 32, 8>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_32X8(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductAVX512_32X8(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_32X8(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_32X8(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=16)
void MinusInnerProductMatrix<float, 32, 16>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_32X16(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductAVX512_32X16(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_32X16(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_32X16(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=32)
void MinusInnerProductMatrix<float, 32, 32>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON_32X32(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductAVX512_32X32(m, q, dim);
    return;
  }
#endif // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductAVX_32X32(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE_32X32(m, q, dim);
#endif
}

#endif
}  // namespace ailego
}  // namespace zvec
