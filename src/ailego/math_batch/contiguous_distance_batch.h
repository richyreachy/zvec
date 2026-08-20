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

#include <array>
#include <cmath>
#include <ailego/math/inner_product_matrix.h>
#include "euclidean_distance_batch.h"
#include "inner_product_distance_batch.h"

namespace zvec::ailego::DistanceBatch {

//! Generic one-to-many sweep over a contiguous packed block of vectors
//! (stride between vectors == dim), for types without a dedicated contiguous
//! kernel. Each vector goes through the BatchSize=1 pointer-batch impl, which
//! does its own runtime ISA dispatch. Note that types whose BatchSize=1 impl
//! expects a preprocessed query (see GetQueryPreprocessFunc) keep the same
//! contract here.
template <template <typename, size_t> class BatchImpl, typename ValueType>
static inline void compute_contiguous_fallback(const ValueType *block,
                                               const ValueType *query,
                                               size_t num, size_t dim,
                                               float *results) {
  const ValueType *vec = block;
  for (size_t i = 0; i < num; ++i, vec += dim) {
    std::array<const ValueType *, 1> prefetch_ptrs{i + 1 < num ? vec + dim
                                                               : nullptr};
    BatchImpl<ValueType, 1>::compute_one_to_many(query, &vec, prefetch_ptrs,
                                                 dim, &results[i]);
  }
}

//! Contiguous inner product; the primary template covers any value type, the
//! specializations resolve to dedicated kernels in the dispatch unit.
template <typename T>
struct InnerProductContiguousBatchImpl {
  using ValueType = typename std::remove_cv<T>::type;
  static void compute(const ValueType *block, const ValueType *query,
                      size_t num, size_t dim, float *results) {
    compute_contiguous_fallback<InnerProductDistanceBatchImpl>(
        block, query, num, dim, results);
  }
};

template <>
struct InnerProductContiguousBatchImpl<float> {
  using ValueType = float;
  static void compute(const float *block, const float *query, size_t num,
                      size_t dim, float *results);
};

//! Contiguous squared euclidean distance; same structure as inner product.
template <typename T>
struct SquaredEuclideanContiguousBatchImpl {
  using ValueType = typename std::remove_cv<T>::type;
  static void compute(const ValueType *block, const ValueType *query,
                      size_t num, size_t dim, float *results) {
    compute_contiguous_fallback<SquaredEuclideanDistanceBatchImpl>(
        block, query, num, dim, results);
  }
};

template <>
struct SquaredEuclideanContiguousBatchImpl<float> {
  using ValueType = float;
  static void compute(const float *block, const float *query, size_t num,
                      size_t dim, float *results);
};

//! Result post-transforms. They run in the caller's (baseline-ISA)
//! translation unit; see MinusInnerProductDistanceBatchImpl for why they
//! must not live in the dispatch unit.
struct IdentityPostprocess {
  static inline void Apply(float *, size_t) {}
};

struct NegatePostprocess {
  static inline void Apply(float *results, size_t num) {
    for (size_t i = 0; i < num; ++i) {
      results[i] = -results[i];
    }
  }
};

struct SqrtPostprocess {
  static inline void Apply(float *results, size_t num) {
    for (size_t i = 0; i < num; ++i) {
      results[i] = std::sqrt(results[i]);
    }
  }
};

//! One-to-many distances over a contiguous packed block of vectors (stride
//! between vectors == dim). Faster than the pointer-batch path when features
//! are stored contiguously, since the linear sweep is covered by hardware
//! prefetch.
template <template <typename> class Impl, typename T,
          typename Postprocess = IdentityPostprocess>
struct ContiguousBatch {
  using ValueType = typename std::remove_cv<T>::type;
  static void Compute(const ValueType *block, const ValueType *query,
                      size_t num, size_t dim, float *results) {
    Impl<ValueType>::compute(block, query, num, dim, results);
    Postprocess::Apply(results, num);
  }
};

template <typename T>
using InnerProductContiguousBatch =
    ContiguousBatch<InnerProductContiguousBatchImpl, T>;

template <typename T>
using MinusInnerProductContiguousBatch =
    ContiguousBatch<InnerProductContiguousBatchImpl, T, NegatePostprocess>;

template <typename T>
using SquaredEuclideanContiguousBatch =
    ContiguousBatch<SquaredEuclideanContiguousBatchImpl, T>;

template <typename T>
using EuclideanContiguousBatch =
    ContiguousBatch<SquaredEuclideanContiguousBatchImpl, T, SqrtPostprocess>;

}  // namespace zvec::ailego::DistanceBatch
