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

static float SquaredEuclideanDistanceAVX512FP16(const Float16 *lhs,const Float16 *rhs, size_t size);

static float SquaredEuclideanDistanceAVX512(const Float16 *lhs, const Float16 *rhs, size_t size);

static float SquaredEuclideanDistanceAVX(const Float16 *lhs, const Float16 *rhs, size_t size);

//! Compute the distance between matrix and query (FP16, M=1, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP16_1X1_NEON(m, q, dim, out, 0ull, )
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = SquaredEuclideanDistanceAVX512FP16(m, q, dim);
    return;
  }
#endif
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceAVX512(m, q, dim);
    //ACCUM_FP16_1X1_AVX512(m, q, dim, out, 0ull, )
    return;
  }
#endif
  *out = SquaredEuclideanDistanceAVX(m, q, dim);
  //ACCUM_FP16_1X1_AVX(m, q, dim, out, 0ull, )
#endif  //__ARM_NEON
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
    *out = std::sqrt(SquaredEuclideanDistanceAVX512(m, q, dim));
    // ACCUM_FP16_1X1_AVX512(m, q, dim, out, 0ull, std::sqrt)
    return;
  }
#endif
  *out = std::sqrt(SquaredEuclideanDistanceAVX(m, q, dim));
  //ACCUM_FP16_1X1_AVX(m, q, dim, out, 0ull, std::sqrt)
#endif  //__ARM_NEON
}

#if 0
//! Compute the distance between matrix and query (FP16, M=2, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 2, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_2X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=2, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 2, 2>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_2X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=4, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 4, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_4X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=4, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 4, 2>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_4X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=4, N=4)
void SquaredEuclideanDistanceMatrix<Float16, 4, 4>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_4X4_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=8, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 8, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_8X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=8, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 8, 2>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_8X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=8, N=4)
void SquaredEuclideanDistanceMatrix<Float16, 8, 4>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_8X4_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=8, N=8)
void SquaredEuclideanDistanceMatrix<Float16, 8, 8>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_8X8_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=16, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 16, 1>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X1_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__

  ACCUM_FP16_16X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=16, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 16, 2>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X2_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__

  ACCUM_FP16_16X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=16, N=4)
void SquaredEuclideanDistanceMatrix<Float16, 16, 4>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X4_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X4_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=16, N=8)
void SquaredEuclideanDistanceMatrix<Float16, 16, 8>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X8_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X8_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=16, N=16)
void SquaredEuclideanDistanceMatrix<Float16, 16, 16>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X16_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X16_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=32, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 32, 1>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X1_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=32, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 32, 2>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X2_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=32, N=4)
void SquaredEuclideanDistanceMatrix<Float16, 32, 4>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X4_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X4_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=32, N=8)
void SquaredEuclideanDistanceMatrix<Float16, 32, 8>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X8_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X8_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=32, N=16)
void SquaredEuclideanDistanceMatrix<Float16, 32, 16>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X16_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X16_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=32, N=32)
void SquaredEuclideanDistanceMatrix<Float16, 32, 32>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X32_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X32_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=2, N=1)
void EuclideanDistanceMatrix<Float16, 2, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_2X1_AVX(m, q, dim, out, _mm_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=2, N=2)
void EuclideanDistanceMatrix<Float16, 2, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_2X2_AVX(m, q, dim, out, _mm_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=4, N=1)
void EuclideanDistanceMatrix<Float16, 4, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_4X1_AVX(m, q, dim, out, _mm_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=4, N=2)
void EuclideanDistanceMatrix<Float16, 4, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_4X2_AVX(m, q, dim, out, _mm_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=4, N=4)
void EuclideanDistanceMatrix<Float16, 4, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_4X4_AVX(m, q, dim, out, _mm_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=8, N=1)
void EuclideanDistanceMatrix<Float16, 8, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_8X1_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=8, N=2)
void EuclideanDistanceMatrix<Float16, 8, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_8X2_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=8, N=4)
void EuclideanDistanceMatrix<Float16, 8, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_8X4_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=8, N=8)
void EuclideanDistanceMatrix<Float16, 8, 8>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_8X8_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=16, N=1)
void EuclideanDistanceMatrix<Float16, 16, 1>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X1_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X1_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=16, N=2)
void EuclideanDistanceMatrix<Float16, 16, 2>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X2_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X2_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=16, N=4)
void EuclideanDistanceMatrix<Float16, 16, 4>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X4_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X4_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=16, N=8)
void EuclideanDistanceMatrix<Float16, 16, 8>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X8_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X8_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=16, N=16)
void EuclideanDistanceMatrix<Float16, 16, 16>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X16_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X16_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=32, N=1)
void EuclideanDistanceMatrix<Float16, 32, 1>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X1_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X1_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=32, N=2)
void EuclideanDistanceMatrix<Float16, 32, 2>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X2_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X2_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=32, N=4)
void EuclideanDistanceMatrix<Float16, 32, 4>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X4_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X4_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=32, N=8)
void EuclideanDistanceMatrix<Float16, 32, 8>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X8_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X8_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=32, N=16)
void EuclideanDistanceMatrix<Float16, 32, 16>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X16_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X16_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=32, N=32)
void EuclideanDistanceMatrix<Float16, 32, 32>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X32_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X32_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

#endif 
}  // namespace ailego
}  // namespace zvec