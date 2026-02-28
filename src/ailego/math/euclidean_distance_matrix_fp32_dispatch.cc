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
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__ARM_NEON)
float SquaredEuclideanDistanceNEON(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__AVX512F__)
float SquaredEuclideanDistanceAVX512(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX512_16X1(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX512_16X2(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX512_16X4(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX512_16X8(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX512_16X16(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX512_32X1(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX512_32X2(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX512_32X4(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX512_32X8(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX512_32X16(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX512_32X32(const float *lhs, const float *rhs, size_t size);

float EuclideanDistanceAVX512(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX512_16X1(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX512_16X2(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX512_16X4(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX512_16X8(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX512_16X16(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX512_32X1(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX512_32X2(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX512_32X4(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX512_32X8(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX512_32X16(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX512_32X32(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__AVX__)
float SquaredEuclideanDistanceAVX(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_2X1(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_2X2(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_4X1(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_4X2(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_4X4(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_8X1(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_8X2(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_8X4(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_8X8(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_16X1(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_16X2(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_16X4(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_16X8(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_16X16(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_32X1(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_32X2(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_32X4(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_32X8(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_32X16(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceAVX_32X32(const float *lhs, const float *rhs, size_t size);

float EuclideanDistanceAVX(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_2X1(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_2X2(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_4X1(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_4X2(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_4X4(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_8X1(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_8X2(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_8X4(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_8X8(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_16X1(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_16X2(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_16X4(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_16X8(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_16X16(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_32X1(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_32X2(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_32X4(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_32X8(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_32X16(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceAVX_32X32(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__SSE__)
float SquaredEuclideanDistanceSSE(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_2X1(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_2X2(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_4X1(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_4X2(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_4X4(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_8X1(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_8X2(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_8X4(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_8X8(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_16X1(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_16X2(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_16X4(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_16X8(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_16X16(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_32X1(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_32X2(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_32X4(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_32X8(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_32X16(const float *lhs, const float *rhs, size_t size);
float SquaredEuclideanDistanceSSE_32X32(const float *lhs, const float *rhs, size_t size);


float EuclideanDistanceSSE(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_2X1(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_2X2(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_4X1(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_4X2(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_4X4(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_8X1(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_8X2(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_8X4(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_8X8(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_16X1(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_16X2(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_16X4(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_16X8(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_16X16(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_32X1(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_32X2(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_32X4(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_32X8(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_32X16(const float *lhs, const float *rhs, size_t size);
float EuclideanDistanceSSE_32X32(const float *lhs, const float *rhs, size_t size);
#endif 

//-----------------------------------------------------------
//  SquaredEuclideanDistance
//-----------------------------------------------------------
#if defined(__SSE__) || defined(__ARM_NEON)
//! Compute the distance between matrix and query (FP32, M=1, N=1)
void SquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    if (dim > 15) {
      *out = SquaredEuclideanDistanceAVX512(m, q, dim);
      return;
    }
  }
#endif  // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    if (dim > 7) {
      *out = SquaredEuclideanDistanceAVX(m, q, dim);
      return;
    }
  }
#endif  // __AVX__
  *out = SquaredEuclideanDistanceSSE(m, q, dim);
#endif  // __ARM_NEON
}
#endif  // __SSE__ || __ARM_NEON 

//! Compute the distance between matrix and query (FP32, M=2, N=1)
void SquaredEuclideanDistanceMatrix<float, 2, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_2X1(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_2X1(m, q, dim);
    return;
  }
#endif // __AVX__
  *out = SquaredEuclideanDistanceSSE_2X1(m, q, dim);
#endif  
}

//! Compute the distance between matrix and query (FP32, M=2, N=2)
void SquaredEuclideanDistanceMatrix<float, 2, 2>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceNEON_2X2(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_2X2(m, q, dim);
    return;
  }
#endif // __AVX__
  *out = SquaredEuclideanDistanceSSE_2X2(m, q, dim);
#endif  
}

//! Compute the distance between matrix and query (FP32, M=4, N=1)
void SquaredEuclideanDistanceMatrix<float, 4, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_4X1(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_4X1(m, q, dim);
    return;
  }
#endif // __AVX__
  *out = SquaredEuclideanDistanceSSE_4X1(m, q, dim);
#endif  
}

//! Compute the distance between matrix and query (FP32, M=4, N=2)
void SquaredEuclideanDistanceMatrix<float, 4, 2>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_4X2(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_4X2(m, q, dim);
    return;
  }
#endif // __AVX__
  *out = SquaredEuclideanDistanceSSE_4X2(m, q, dim);
#endif  
}

//! Compute the distance between matrix and query (FP32, M=4, N=4)
void SquaredEuclideanDistanceMatrix<float, 4, 4>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_4X4(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_4X4(m, q, dim);
    return;
  }
#endif // __AVX__
  *out = SquaredEuclideanDistanceSSE_4X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=1)
void SquaredEuclideanDistanceMatrix<float, 8, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_8X1(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_8X1(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = SquaredEuclideanDistanceSSE_8X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=8, N=2)
void SquaredEuclideanDistanceMatrix<float, 8, 2>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_8X2(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_8X2(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = SquaredEuclideanDistanceSSE_8X2(m, q, dim);
#endif  
}

//! Compute the distance between matrix and query (FP32, M=8, N=4)
void SquaredEuclideanDistanceMatrix<float, 8, 4>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_8X4(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_8X4(m, q, dim);
    return;
  }
#endif  // __AVX__
  *out = SquaredEuclideanDistanceSSE_8X4(m, q, dim);
#endif 
}

//! Compute the distance between matrix and query (FP32, M=8, N=8)
void SquaredEuclideanDistanceMatrix<float, 8, 8>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_8X8(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_8X8(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceSSE_8X8(m, q, dim);
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=16, N=1)
void SquaredEuclideanDistanceMatrix<float, 16, 1>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_16X1(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceAVX512_16X1(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_16X1(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceSSE_16X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=2)
void SquaredEuclideanDistanceMatrix<float, 16, 2>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_16X2(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceAVX512_16X2(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_16X2(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceSSE_16X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=4)
void SquaredEuclideanDistanceMatrix<float, 16, 4>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_16X6(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceAVX512_16X4(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_16X4(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceSSE_16X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=8)
void SquaredEuclideanDistanceMatrix<float, 16, 8>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_16X8(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceAVX512_16X8(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_16X8(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceSSE_16X8(m, q, dim);
#endif 
}

//! Compute the distance between matrix and query (FP32, M=16, N=16)
void SquaredEuclideanDistanceMatrix<float, 16, 16>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_16X16(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceAVX512_16X16(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_16X16(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceSSE_16X16(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=1)
void SquaredEuclideanDistanceMatrix<float, 32, 1>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_32X1(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceAVX512_32X1(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_32X1(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceSSE_32X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=2)
void SquaredEuclideanDistanceMatrix<float, 32, 2>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_32X2(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceAVX512_32X2(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_32X2(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceSSE_32X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=4)
void SquaredEuclideanDistanceMatrix<float, 32, 4>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_32X4(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceAVX512_32X4(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_32X4(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceSSE_32X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=8)
void SquaredEuclideanDistanceMatrix<float, 32, 8>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_32X8(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceAVX512_32X8(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_32X8(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceSSE_32X8(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=16)
void SquaredEuclideanDistanceMatrix<float, 32, 16>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_32X16(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceAVX512_32X16(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_32X16(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceSSE_32X16(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=32)
void SquaredEuclideanDistanceMatrix<float, 32, 32>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON_32X32(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceAVX512_32X32(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceAVX_32X32(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceSSE_32X32(m, q, dim);
#endif
}

//-----------------------------------------------------------
//  EuclideanDistance
//-----------------------------------------------------------
#if defined(__SSE__) || (defined(__ARM_NEON) && defined(__aarch64__))
//! Compute the distance between matrix and query (FP32, M=1, N=1)
void EuclideanDistanceMatrix<float, 1, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = std::sqrt(SquaredEuclideanDistanceNEON(m, q, dim));
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    if (dim > 15) {
      *out = std::sqrt(SquaredEuclideanDistanceAVX512(m, q, dim));
      return;
    }
  }
#endif  // __AVX512F__

#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    if (dim > 7) {
      *out = std::sqrt(SquaredEuclideanDistanceAVX(m, q, dim));
      return;
    }
  }
#endif  // __AVX__
  *out = std::sqrt(SquaredEuclideanDistanceSSE(m, q, dim));
#endif  // __ARM_NEON
}
#endif  // __SSE__ || __ARM_NEON && __aarch64__


//! Compute the distance between matrix and query (FP32, M=2, N=1)
void EuclideanDistanceMatrix<float, 2, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_2X1(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_2X1(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_2X1(m, q, dim);
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=2, N=2)
void EuclideanDistanceMatrix<float, 2, 2>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_2X2(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_2X2(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_2X2(m, q, dim);
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=4, N=1)
void EuclideanDistanceMatrix<float, 4, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_4X1(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_4X1(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_4X1(m, q, dim);
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=4, N=2)
void EuclideanDistanceMatrix<float, 4, 2>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_4X2(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_4X2(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_4X2(m, q, dim);
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=4, N=4)
void EuclideanDistanceMatrix<float, 4, 4>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_4X4(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_4X4(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_4X4(m, q, dim);
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=1)
void EuclideanDistanceMatrix<float, 8, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_8X1(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_8X1(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_8X1(m, q, dim);
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=2)
void EuclideanDistanceMatrix<float, 8, 2>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_8X2(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_8X2(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_8X2(m, q, dim);
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=4)
void EuclideanDistanceMatrix<float, 8, 4>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_8X4(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_8X4(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_8X4(m, q, dim);
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=8)
void EuclideanDistanceMatrix<float, 8, 8>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_8X8(m, q, dim);
#else
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_8X8(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_8X8(m, q, dim);
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=16, N=1)
void EuclideanDistanceMatrix<float, 16, 1>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_16X1(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = EuclideanDistanceAVX512_16X1(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_16X1(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_16X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=2)
void EuclideanDistanceMatrix<float, 16, 2>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_16X2(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = EuclideanDistanceAVX512_16X2(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_16X2(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_16X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=4)
void EuclideanDistanceMatrix<float, 16, 4>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_16X6(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = EuclideanDistanceAVX512_16X4(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_16X4(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_16X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=8)
void EuclideanDistanceMatrix<float, 16, 8>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_16X8(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = EuclideanDistanceAVX512_16X8(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_16X8(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_16X8(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=16)
void EuclideanDistanceMatrix<float, 16, 16>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_16X16(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = EuclideanDistanceAVX512_16X16(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_16X16(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_16X16(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=1)
void EuclideanDistanceMatrix<float, 32, 1>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_32X1(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = EuclideanDistanceAVX512_32X1(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_32X1(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_32X1(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=2)
void EuclideanDistanceMatrix<float, 32, 2>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_32X2(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = EuclideanDistanceAVX512_32X2(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_32X2(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_32X2(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=4)
void EuclideanDistanceMatrix<float, 32, 4>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_32X4(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = EuclideanDistanceAVX512_32X4(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_32X4(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_32X4(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=8)
void EuclideanDistanceMatrix<float, 32, 8>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_32X8(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = EuclideanDistanceAVX512_32X8(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_32X8(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_32X8(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=16)
void EuclideanDistanceMatrix<float, 32, 16>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_32X16(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = EuclideanDistanceAVX512_32X16(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_32X16(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_32X16(m, q, dim);
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=32)
void EuclideanDistanceMatrix<float, 32, 32>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  *out = EuclideanDistanceNEON_32X32(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = EuclideanDistanceAVX512_32X32(m, q, dim);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = EuclideanDistanceAVX_32X32(m, q, dim);
    return;
  }
#endif
  *out = EuclideanDistanceSSE_32X32(m, q, dim);
#endif
}

}  // namespace ailego
}  // namespace zvec