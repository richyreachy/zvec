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
float InnerProductNEON(const Float16 *lhs,const Float16 *rhs, size_t size);
float MinusInnerProductNEON(const Float16 *lhs,const Float16 *rhs, size_t size);
#endif

#if defined(__AVX__)
float InnerProductAVX(const Float16 *lhs, const Float16 *rhs, size_t size);
void InnerProductAVX_2X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_2X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_4X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_4X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_4X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_8X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_8X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_8X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_8X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_16X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_16X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_16X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_16X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_16X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_32X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_32X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_32X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_32X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_32X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX_32X32(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);

float MinusInnerProductAVX(const Float16 *lhs, const Float16 *rhs, size_t size);
void MinusInnerProductAVX_2X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_2X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_4X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_4X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_4X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_8X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_8X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_8X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_8X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_16X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_16X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_16X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_16X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_16X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_32X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_32X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_32X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_32X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_32X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX_32X32(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);

float InnerProductSparseInSegmentAVX(uint32_t m_sparse_count, const uint16_t *m_sparse_index,
                                     const Float16 *m_sparse_value, uint32_t q_sparse_count,
                                     const uint16_t *q_sparse_index, const Float16 *q_sparse_value);
#endif

#if defined(__AVX512F__)
float InnerProductAVX512(const Float16 *lhs, const Float16 *rhs, size_t size);
void InnerProductAVX512_16X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX512_16X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX512_16X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX512_16X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX512_16X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX512_32X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX512_32X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX512_32X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX512_32X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX512_32X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void InnerProductAVX512_32X32(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);

float MinusInnerProductAVX512(const Float16 *lhs, const Float16 *rhs, size_t size);
void MinusInnerProductAVX512_16X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX512_16X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX512_16X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX512_16X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX512_16X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX512_32X1(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX512_32X2(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX512_32X4(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX512_32X8(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX512_32X16(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
void MinusInnerProductAVX512_32X32(const Float16 *lhs, const Float16 *rhs, size_t size, float *out);
#endif

#if defined(__AVX512FP16__)
float InnerProductAVX512FP16(const Float16 *lhs,const Float16 *rhs, size_t size);
float InnerProductSparseInSegmentAVX512FP16(uint32_t m_sparse_count, const uint16_t *m_sparse_index,
                                     const Float16 *m_sparse_value, uint32_t q_sparse_count,
                                     const uint16_t *q_sparse_index, const Float16 *q_sparse_value);
#endif

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))
//! Compute the distance between matrix and query (FP16, M=1, N=1)
void InnerProductMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON(m, q, dim);
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = InnerProductAVX512FP16(m, q, dim);
    return;
  }
#endif  //__AVX512FP16__
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductAVX512(m, q, dim);
    return;
  }
#endif  //__AVX512F__
  *out = InnerProductAVX(m, q, dim);
#endif  //__ARM_NEON
}

//! Compute the distance between matrix and query (FP16, M=1, N=1)
void MinusInnerProductMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON(m, q, dim);
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = -InnerProductAVX512FP16(m, q, dim);
    return;
  }
#endif  //__AVX512FP16__
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductAVX512(m, q, dim);
    return;
  }
#endif  //__AVX512F__

  *out = MinusInnerProductAVX(m, q, dim);

#endif  //__ARM_NEON
}

#if !defined(__ARM_NEON)
//! Compute the distance between matrix and query (FP16, M=2, N=1)
void InnerProductMatrix<Float16, 2, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  InnerProductAVX_2X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=2, N=2)
void InnerProductMatrix<Float16, 2, 2>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  InnerProductAVX_2X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=4, N=1)
void InnerProductMatrix<Float16, 4, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  InnerProductAVX_4X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=4, N=2)
void InnerProductMatrix<Float16, 4, 2>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  InnerProductAVX_4X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=4, N=4)
void InnerProductMatrix<Float16, 4, 4>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  InnerProductAVX_4X4(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=1)
void InnerProductMatrix<Float16, 8, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  InnerProductAVX_8X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=2)
void InnerProductMatrix<Float16, 8, 2>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  InnerProductAVX_8X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=4)
void InnerProductMatrix<Float16, 8, 4>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  InnerProductAVX_8X4(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=8)
void InnerProductMatrix<Float16, 8, 8>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  InnerProductAVX_8X8(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=16, N=1)
void InnerProductMatrix<Float16, 16, 1>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_16X1(m, q, dim, out);
    return;
  }
#else
  InnerProductAVX_16X1(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=2)
void InnerProductMatrix<Float16, 16, 2>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_16X2(m, q, dim, out);
    return;
  }
#else
  InnerProductAVX_16X2(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=4)
void InnerProductMatrix<Float16, 16, 4>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_16X4(m, q, dim, out);
    return;
  }
#else
  InnerProductAVX_16X4(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=8)
void InnerProductMatrix<Float16, 16, 8>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_16X8(m, q, dim, out);
    return;
  }
#else
  InnerProductAVX_16X8(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=16)
void InnerProductMatrix<Float16, 16, 16>::Compute(const ValueType *m,
                                                  const ValueType *q,
                                                  size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_16X16(m, q, dim, out);
    return;
  }
#else
  InnerProductAVX_16X16(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=1)
void InnerProductMatrix<Float16, 32, 1>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_32X1(m, q, dim, out);
    return;
  }
#else
  InnerProductAVX_32X1(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=2)
void InnerProductMatrix<Float16, 32, 2>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_32X2(m, q, dim, out);
    return;
  }
#else
  InnerProductAVX_32X2(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=4)
void InnerProductMatrix<Float16, 32, 4>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_32X4(m, q, dim, out);
    return;
  }
#else
  InnerProductAVX_32X4(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=8)
void InnerProductMatrix<Float16, 32, 8>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_32X8(m, q, dim, out);
    return;
  }
#else
  InnerProductAVX_32X8(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=16)
void InnerProductMatrix<Float16, 32, 16>::Compute(const ValueType *m,
                                                  const ValueType *q,
                                                  size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_32X16(m, q, dim, out);
    return;
  }
#else
  InnerProductAVX_32X16(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=32)
void InnerProductMatrix<Float16, 32, 32>::Compute(const ValueType *m,
                                                  const ValueType *q,
                                                  size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    InnerProductAVX512_32X32(m, q, dim, out);
    return;
  }
#else
  InnerProductAVX_32X32(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=2, N=1)
void MinusInnerProductMatrix<Float16, 2, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  MinusInnerProductAVX_2X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=2, N=2)
void MinusInnerProductMatrix<Float16, 2, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  MinusInnerProductAVX_2X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=4, N=1)
void MinusInnerProductMatrix<Float16, 4, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  MinusInnerProductAVX_4X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=4, N=2)
void MinusInnerProductMatrix<Float16, 4, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  MinusInnerProductAVX_4X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=4, N=4)
void MinusInnerProductMatrix<Float16, 4, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  MinusInnerProductAVX_4X4(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=1)
void MinusInnerProductMatrix<Float16, 8, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  MinusInnerProductAVX_8X1(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=2)
void MinusInnerProductMatrix<Float16, 8, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  MinusInnerProductAVX_8X2(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=4)
void MinusInnerProductMatrix<Float16, 8, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  MinusInnerProductAVX_8X4(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=8, N=8)
void MinusInnerProductMatrix<Float16, 8, 8>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  MinusInnerProductAVX_8X8(m, q, dim, out);
}

//! Compute the distance between matrix and query (FP16, M=16, N=1)
void MinusInnerProductMatrix<Float16, 16, 1>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_16X1(m, q, dim, out);
    return;
  }
#else
  MinusInnerProductAVX_16X1(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=2)
void MinusInnerProductMatrix<Float16, 16, 2>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_16X2(m, q, dim, out);
    return;
  }
#else
  MinusInnerProductAVX_16X2(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=4)
void MinusInnerProductMatrix<Float16, 16, 4>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_16X4(m, q, dim, out);
    return;
  }
#else
  MinusInnerProductAVX_16X4(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=8)
void MinusInnerProductMatrix<Float16, 16, 8>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_16X8(m, q, dim, out);
    return;
  }
#else
  MinusInnerProductAVX_16X8(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=16)
void MinusInnerProductMatrix<Float16, 16, 16>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_16X16(m, q, dim, out);
    return;
  }
#else
  MinusInnerProductAVX_16X16(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=1)
void MinusInnerProductMatrix<Float16, 32, 1>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_32X1(m, q, dim, out);
    return;
  }
#else
  MinusInnerProductAVX_32X1(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=2)
void MinusInnerProductMatrix<Float16, 32, 2>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_32X2(m, q, dim, out);
    return;
  }
#else
  MinusInnerProductAVX_32X2(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=4)
void MinusInnerProductMatrix<Float16, 32, 4>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_32X4(m, q, dim, out);
    return;
  }
#else
  MinusInnerProductAVX_32X4(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=8)
void MinusInnerProductMatrix<Float16, 32, 8>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_32X8(m, q, dim, out);
    return;
  }
#else
  MinusInnerProductAVX_32X8(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=16)
void MinusInnerProductMatrix<Float16, 32, 16>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_32X16(m, q, dim, out);
    return;
  }
#else
  MinusInnerProductAVX_32X16(m, q, dim, out);
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=32)
void MinusInnerProductMatrix<Float16, 32, 32>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    MinusInnerProductAVX512_32X32(m, q, dim, out);
    return;
  }
#else
  MinusInnerProductAVX_32X32(m, q, dim, out);
#endif  // __AVX512F__
}
#endif  // !__ARM_NEON
#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

//sparse
float InnerProductSparseInSegment(uint32_t m_sparse_count,
                                  const uint16_t *m_sparse_index,
                                  const Float16 *m_sparse_value,
                                  uint32_t q_sparse_count,
                                  const uint16_t *q_sparse_index,
                                  const Float16 *q_sparse_value) {
  float sum = 0.0f;

  size_t m_i = 0;
  size_t q_i = 0;
  while (m_i < m_sparse_count && q_i < q_sparse_count) {
    if (m_sparse_index[m_i] == q_sparse_index[q_i]) {
      sum += m_sparse_value[m_i] * q_sparse_value[q_i];

      ++m_i;
      ++q_i;
    } else if (m_sparse_index[m_i] < q_sparse_index[q_i]) {
      ++m_i;
    } else {
      ++q_i;
    }
  }

  return sum;
}

template <>
float MinusInnerProductSparseMatrix<Float16>::
    ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                       const uint16_t *m_sparse_index,
                                       const ValueType *m_sparse_value,
                                       uint32_t q_sparse_count,
                                       const uint16_t *q_sparse_index,
                                       const ValueType *q_sparse_value) {
#if defined(__AVX__)
  return InnerProductSparseInSegmentAVX(m_sparse_count, m_sparse_index,
                                        m_sparse_value, q_sparse_count,
                                        q_sparse_index, q_sparse_value);
#elif defined(__AVX512FP16__)
  return InnerProductSparseInSegmentAVX512FP16(m_sparse_count, m_sparse_index,
                                               m_sparse_value, q_sparse_count,
                                               q_sparse_index, q_sparse_value);
#else
  return InnerProductSparseInSegment(m_sparse_count, m_sparse_index,
                                     m_sparse_value, q_sparse_count,
                                     q_sparse_index, q_sparse_value);
#endif
}

}  // namespace ailego
}  // namespace zvec