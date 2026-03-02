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

#if defined(__AVX2__)
float SquaredEuclideanDistanceAVX2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
void SquaredEuclideanDistanceAVX2_2X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_2X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_4X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_4X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_4X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_8X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_8X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_8X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_8X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_16X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_16X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_16X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_16X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_16X16(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_32X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_32X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_32X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_32X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_32X16(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX2_32X32(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);

float EuclideanDistanceAVX2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
void EuclideanDistanceAVX2_2X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_2X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_4X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_4X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_4X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_8X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_8X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_8X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_8X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_16X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_16X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_16X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_16X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_16X16(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_32X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_32X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_32X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_32X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_32X16(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceAVX2_32X32(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
#endif

#if defined(__SSE4_1__)
float SquaredEuclideanDistanceSSE(const uint8_t *lhs, const uint8_t *rhs, size_t size);
void SquaredEuclideanDistanceSSE_2X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_2X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_4X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_4X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_4X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_8X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_8X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_8X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_8X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_16X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_16X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_16X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_16X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_16X16(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_32X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_32X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_32X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_32X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_32X16(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void SquaredEuclideanDistanceSSE_32X32(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);


float EuclideanDistanceSSE(const uint8_t *lhs, const uint8_t *rhs, size_t size);
void EuclideanDistanceSSE_2X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_2X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_4X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_4X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_4X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_8X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_8X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_8X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_8X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_16X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_16X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_16X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_16X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_16X16(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_32X1(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_32X2(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_32X4(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_32X8(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_32X16(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
void EuclideanDistanceSSE_32X32(const uint8_t *lhs, const uint8_t *rhs, size_t size, float *out);
#endif 

#if defined(__SSE4_1__)
//! Compute the distance between matrix and query (INT4, M=1, N=1)
void SquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__AVX2__)
  if (dim > 63) {
    *out = SquaredEuclideanDistanceAVX2(m, q, dim >> 1);
    return;
  }
#endif  // __AVX2__
  *out = SquaredEuclideanDistanceSSE(m, q, dim >> 1);
}

//! Compute the distance between matrix and query (INT4, M=2, N=1)
void SquaredEuclideanDistanceMatrix<uint8_t, 2, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_2X1(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_2X1(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=2, N=2)
void SquaredEuclideanDistanceMatrix<uint8_t, 2, 2>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_2X2(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_2X2(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=4, N=1)
void SquaredEuclideanDistanceMatrix<uint8_t, 4, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_4X1(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_4X1(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=4, N=2)
void SquaredEuclideanDistanceMatrix<uint8_t, 4, 2>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_4X2(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_4X2(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=4, N=4)
void SquaredEuclideanDistanceMatrix<uint8_t, 4, 4>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_4X4(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_4X4(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=8, N=1)
void SquaredEuclideanDistanceMatrix<uint8_t, 8, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_8X1(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_8X1(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=8, N=2)
void SquaredEuclideanDistanceMatrix<uint8_t, 8, 2>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_8X2(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_8X2(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=8, N=4)
void SquaredEuclideanDistanceMatrix<uint8_t, 8, 4>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_2X1(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_2X1(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=8, N=8)
void SquaredEuclideanDistanceMatrix<uint8_t, 8, 8>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_8X8(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_8X8(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=1)
void SquaredEuclideanDistanceMatrix<uint8_t, 16, 1>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_16X1(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_16X1(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=2)
void SquaredEuclideanDistanceMatrix<uint8_t, 16, 2>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_16X2(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_16X2(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=4)
void SquaredEuclideanDistanceMatrix<uint8_t, 16, 4>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_16X4(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_16X4(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=8)
void SquaredEuclideanDistanceMatrix<uint8_t, 16, 8>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_16X8(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_16X8(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=16)
void SquaredEuclideanDistanceMatrix<uint8_t, 16, 16>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_16X16(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_16X16(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=1)
void SquaredEuclideanDistanceMatrix<uint8_t, 32, 1>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_32X1(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_32X1(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=2)
void SquaredEuclideanDistanceMatrix<uint8_t, 32, 2>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_32X2(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_32X3(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=4)
void SquaredEuclideanDistanceMatrix<uint8_t, 32, 4>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_32X4(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_32X4(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=8)
void SquaredEuclideanDistanceMatrix<uint8_t, 32, 8>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_32X8(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_32X8(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=16)
void SquaredEuclideanDistanceMatrix<uint8_t, 32, 16>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_32X16(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_32X16(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=32)
void SquaredEuclideanDistanceMatrix<uint8_t, 32, 32>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    SquaredEuclideanDistanceAVX2_32X32(m, q, dim, out);
    return;
  }
#else
  SquaredEuclideanDistanceSSE_32X32(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=1, N=1)
void EuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (dim > 63) {
    *out = std::sqrt(SquaredEuclideanDistanceAVX2(m, q, dim >> 1));
    return;
  }
#endif  // __AVX2__
  *out = std::sqrt(SquaredEuclideanDistanceSSE(m, q, dim >> 1));
}

//! Compute the distance between matrix and query (INT8, M=2, N=1)
void EuclideanDistanceMatrix<uint8_t, 2, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_2X1(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_2X1(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=2, N=2)
void EuclideanDistanceMatrix<uint8_t, 2, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_2X2(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_2X2(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=4, N=1)
void EuclideanDistanceMatrix<uint8_t, 4, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_4X1(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_4X1(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=4, N=2)
void EuclideanDistanceMatrix<uint8_t, 4, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_4X2(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_4X2(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=4, N=4)
void EuclideanDistanceMatrix<uint8_t, 4, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_4X4(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_4X4(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=8, N=1)
void EuclideanDistanceMatrix<uint8_t, 8, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_8X1(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_8X1(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=8, N=2)
void EuclideanDistanceMatrix<uint8_t, 8, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_8X2(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_8X2(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=8, N=4)
void EuclideanDistanceMatrix<uint8_t, 8, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_8X4(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_8X4(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=8, N=8)
void EuclideanDistanceMatrix<uint8_t, 8, 8>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_8X8(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_8X8(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=16, N=1)
void EuclideanDistanceMatrix<uint8_t, 16, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_16X1(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_16X1(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=16, N=2)
void EuclideanDistanceMatrix<uint8_t, 16, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_16X2(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_16X2(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=16, N=4)
void EuclideanDistanceMatrix<uint8_t, 16, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_16X4(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_16X4(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=16, N=8)
void EuclideanDistanceMatrix<uint8_t, 16, 8>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_16X8(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_16X8(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=16, N=16)
void EuclideanDistanceMatrix<uint8_t, 16, 16>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_16X16(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_16X16(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=32, N=1)
void EuclideanDistanceMatrix<uint8_t, 32, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_32X1(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_32X1(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=32, N=2)
void EuclideanDistanceMatrix<uint8_t, 32, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_32X2(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_32X2(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=32, N=4)
void EuclideanDistanceMatrix<uint8_t, 32, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_32X4(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_32X4(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=32, N=8)
void EuclideanDistanceMatrix<uint8_t, 32, 8>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_32X8(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_32X8(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=32, N=16)
void EuclideanDistanceMatrix<uint8_t, 32, 16>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_32X16(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_32X16(m, q, dim, out);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT8, M=32, N=32)
void EuclideanDistanceMatrix<uint8_t, 32, 32>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    EuclideanDistanceAVX2_32X32(m, q, dim, out);
    return;
  }
#else
  EuclideanDistanceSSE_32X32(m, q, dim, out);
#endif  // __AVX2__
}
#endif  // __SSE4_1__

}  // namespace ailego
}  // namespace zvec