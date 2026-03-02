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
float SquaredEuclideanDistanceNEON(const Float16 *lhs,const Float16 *rhs, size_t size);
#endif

#if defined(__AVX512FP16__)
float SquaredEuclideanDistanceAVX512FP16(const Float16 *lhs,const Float16 *rhs, size_t size);
#endif

#if defined(__AVX512F__)
void SquaredEuclideanDistanceAVX512(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX512_16X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX512_16X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX512_16X4(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX512_16X8(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX512_16X16(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX512_32X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX512_32X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX512_32X4(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX512_32X8(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX512_32X16(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX512_32X32(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);

void EuclideanDistanceAVX512_1X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX512_16X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX512_16X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX512_16X4(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX512_16X8(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX512_16X16(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX512_32X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX512_32X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX512_32X4(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX512_32X8(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX512_32X16(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX512_32X32(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
#endif

#if defined(__AVX__)
void SquaredEuclideanDistanceAVX(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void SquaredEuclideanDistanceAVX_2X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_2X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_4X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_4X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_4X4(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_8X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_8X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_8X4(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_8X8(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_16X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_16X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_16X4(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_16X8(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_16X16(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_32X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_32X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_32X4(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_32X8(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_32X16(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void SquaredEuclideanDistanceAVX_32X32(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);

void EuclideanDistanceAVX_1X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_2X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_2X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_4X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_4X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_4X4(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_8X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_8X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_8X4(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_8X8(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_16X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_16X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_16X4(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_16X8(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_16X16(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_32X1(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_32X2(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_32X4(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_32X8(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_32X16(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
void EuclideanDistanceAVX_32X32(const Float16 *lhs, const Float16 *rhs, size_t size, float* out);
#endif

#if defined(__AVX__)
//! Compute the distance between matrix and query (FP16, M=1, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON(m, q, dim);  
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = SquaredEuclideanDistanceAVX512FP16(m, q, dim);
    return;
  }
#endif
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    SquaredEuclideanDistanceAVX512(m, q, dim, out);
    //ACCUM_FP16_1X1_AVX512(m, q, dim, out, 0ull, )
    return;
  }
#endif
  SquaredEuclideanDistanceAVX(m, q, dim, out);
  //ACCUM_FP16_1X1_AVX(m, q, dim, out, 0ull, )
#endif  //__ARM_NEON
}

//! Compute the distance between matrix and query (FP16, M=2, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 2, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  SquaredEuclideanDistanceAVX_2X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=2, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 2, 2>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  SquaredEuclideanDistanceAVX_2X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=4, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 4, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  SquaredEuclideanDistanceAVX_4X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=4, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 4, 2>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  SquaredEuclideanDistanceAVX_4X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=4, N=4)
void SquaredEuclideanDistanceMatrix<Float16, 4, 4>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  SquaredEuclideanDistanceAVX_4X4(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 8, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  SquaredEuclideanDistanceAVX_8X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 8, 2>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  SquaredEuclideanDistanceAVX_8X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=4)
void SquaredEuclideanDistanceMatrix<Float16, 8, 4>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  SquaredEuclideanDistanceAVX_8X4(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=8)
void SquaredEuclideanDistanceMatrix<Float16, 8, 8>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  SquaredEuclideanDistanceAVX_8X8(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=16, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 16, 1>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    SquaredEuclideanDistanceAVX512_16X1(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  SquaredEuclideanDistanceAVX_16X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=16, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 16, 2>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    SquaredEuclideanDistanceAVX512_16X2(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  SquaredEuclideanDistanceAVX_16X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=16, N=4)
void SquaredEuclideanDistanceMatrix<Float16, 16, 4>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    SquaredEuclideanDistanceAVX512_16X4(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  SquaredEuclideanDistanceAVX_16X4(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=16, N=8)
void SquaredEuclideanDistanceMatrix<Float16, 16, 8>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    SquaredEuclideanDistanceAVX512_16X8(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  SquaredEuclideanDistanceAVX_16X8(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=16, N=16)
void SquaredEuclideanDistanceMatrix<Float16, 16, 16>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    SquaredEuclideanDistanceAVX512_16X16(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  SquaredEuclideanDistanceAVX_16X16(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=32, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 32, 1>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    SquaredEuclideanDistanceAVX512_32X1(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  SquaredEuclideanDistanceAVX_32X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=32, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 32, 2>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    SquaredEuclideanDistanceAVX512_32X2(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  SquaredEuclideanDistanceAVX_32X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=32, N=4)
void SquaredEuclideanDistanceMatrix<Float16, 32, 4>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    SquaredEuclideanDistanceAVX512_32X4(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  SquaredEuclideanDistanceAVX_32X4(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=32, N=8)
void SquaredEuclideanDistanceMatrix<Float16, 32, 8>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    SquaredEuclideanDistanceAVX512_32X8(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  SquaredEuclideanDistanceAVX_32X8(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=32, N=16)
void SquaredEuclideanDistanceMatrix<Float16, 32, 16>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    SquaredEuclideanDistanceAVX512_32X16(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  SquaredEuclideanDistanceAVX_32X16(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=32, N=32)
void SquaredEuclideanDistanceMatrix<Float16, 32, 32>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    SquaredEuclideanDistanceAVX512_32X32(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  SquaredEuclideanDistanceAVX_32X32(m, q, dim, out);
}


//! Compute the distance between matrix and query (FP16, M=1, N=1)
void EuclideanDistanceMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNeon(m, q, dim);
  //ACCUM_FP16_1X1_NEON(m, q, dim, out, 0ull, std::sqrt)
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = std::sqrt(SquaredEuclideanDistanceAVX512FP16(m, q, dim));
    return;
  }
#endif
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    EuclideanDistanceAVX512_1X1(m, q, dim, out);
    return;
  }
#endif
  EuclideanDistanceAVX_1X1(m, q, dim, out);
#endif  //__ARM_NEON
}

//! Compute the distance between matrix and query (FP16, M=2, N=1)
void EuclideanDistanceMatrix<Float16, 2, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  EuclideanDistanceAVX_2X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=2, N=2)
void EuclideanDistanceMatrix<Float16, 2, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  EuclideanDistanceAVX_2X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=4, N=1)
void EuclideanDistanceMatrix<Float16, 4, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  EuclideanDistanceAVX_4X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=4, N=2)
void EuclideanDistanceMatrix<Float16, 4, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  EuclideanDistanceAVX_4X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=4, N=4)
void EuclideanDistanceMatrix<Float16, 4, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  EuclideanDistanceAVX_4X4(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=1)
void EuclideanDistanceMatrix<Float16, 8, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  EuclideanDistanceAVX_8X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=2)
void EuclideanDistanceMatrix<Float16, 8, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  EuclideanDistanceAVX_8X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=4)
void EuclideanDistanceMatrix<Float16, 8, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  EuclideanDistanceAVX_8X4(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=8)
void EuclideanDistanceMatrix<Float16, 8, 8>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  EuclideanDistanceAVX_8X8(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=16, N=1)
void EuclideanDistanceMatrix<Float16, 16, 1>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    EuclideanDistanceAVX512_16X1(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  EuclideanDistanceAVX_16X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=16, N=2)
void EuclideanDistanceMatrix<Float16, 16, 2>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    EuclideanDistanceAVX512_16X2(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  EuclideanDistanceAVX_16X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=16, N=4)
void EuclideanDistanceMatrix<Float16, 16, 4>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    EuclideanDistanceAVX512_16X4(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  EuclideanDistanceAVX_16X4(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=16, N=8)
void EuclideanDistanceMatrix<Float16, 16, 8>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    EuclideanDistanceAVX512_16X8(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  EuclideanDistanceAVX_16X8(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=16, N=16)
void EuclideanDistanceMatrix<Float16, 16, 16>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    EuclideanDistanceAVX512_16X16(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  EuclideanDistanceAVX_16X16(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=32, N=1)
void EuclideanDistanceMatrix<Float16, 32, 1>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    EuclideanDistanceAVX512_32X1(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  EuclideanDistanceAVX_32X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=32, N=2)
void EuclideanDistanceMatrix<Float16, 32, 2>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    EuclideanDistanceAVX512_32X2(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  EuclideanDistanceAVX_32X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=32, N=4)
void EuclideanDistanceMatrix<Float16, 32, 4>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    EuclideanDistanceAVX512_32X4(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  EuclideanDistanceAVX_32X4(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=32, N=8)
void EuclideanDistanceMatrix<Float16, 32, 8>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    EuclideanDistanceAVX512_32X8(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  EuclideanDistanceAVX_32X8(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=32, N=16)
void EuclideanDistanceMatrix<Float16, 32, 16>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    EuclideanDistanceAVX512_32X16(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  EuclideanDistanceAVX_32X16(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=32, N=32)
void EuclideanDistanceMatrix<Float16, 32, 32>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    EuclideanDistanceAVX512_32X32(m, q, dim, out);
    return;
  }
#endif  // __AVX512F__

  EuclideanDistanceAVX_32X32(m, q, dim, out);
}

#endif

}  // namespace ailego
}  // namespace zvec