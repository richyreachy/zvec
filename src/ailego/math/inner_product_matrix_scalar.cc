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

#include <cmath>
#include <string>
#include <vector>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>
#include "distance_utility.h"

namespace zvec {
namespace ailego {

/*! Inner Product Matrix
 */
template <typename T, size_t M, size_t N, typename = void>
struct InnerProductMatrix;

/*! Inner Product Matrix (M=1, N=1)
 */
template <typename T>
struct InnerProductMatrix<
    T, 1, 1, typename std::enable_if<IsSignedArithmetic<T>::value>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && out);

    float sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      sum += static_cast<float>(m[i] * q[i]);
    }
    *out = sum;
  }
};

#if !defined(__SSE4_1__)
/*! Inner Product Matrix (INT4, M=1, N=1)
 */
template <>
struct InnerProductMatrix<uint8_t, 1, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && !(dim & 1) && out);

    float sum = 0.0;
    for (size_t i = 0; i < (dim >> 1); ++i) {
      uint8_t m_val = m[i];
      uint8_t q_val = q[i];
      sum += Int4MulTable[((m_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
             Int4MulTable[((m_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
    }
    *out = sum;
  }
};
#endif  // !__SSE4_1__

template <typename T, size_t M, size_t N, typename = void>
struct MinusInnerProductMatrix;

/*! Minus Inner Product Matrix (M=1, N=1)
 */
template <typename T>
struct MinusInnerProductMatrix<
    T, 1, 1, typename std::enable_if<IsSignedArithmetic<T>::value>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && out);

    float sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      sum += static_cast<float>(m[i] * q[i]);
    }
    *out = -sum;
  }
};

/*! Minus Inner Product Matrix (INT4, M=1, N=1)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 1, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && !(dim & 1) && out);

    float sum = 0.0;
    for (size_t i = 0; i < (dim >> 1); ++i) {
      uint8_t m_val = m[i];
      uint8_t q_val = q[i];
      sum -= Int4MulTable[((m_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
             Int4MulTable[((m_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
    }
    *out = sum;
  }
};

/*! Inner Product Matrix (FP32, M=1, N=1)
 */
template <>
struct InnerProductMatrix<float, 1, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=1, N=1)
 */
template <>
struct MinusInnerProductMatrix<float, 1, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=1, N=1)
 */
template <>
struct InnerProductMatrix<Float16, 1, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=1, N=1)
 */
template <>
struct MinusInnerProductMatrix<Float16, 1, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=1, N=1)
 */
template <>
struct InnerProductMatrix<int8_t, 1, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=1, N=1)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 1, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};


/*! Inner Product Matrix (INT4, M=1, N=1)
 */
template <>
struct InnerProductMatrix<uint8_t, 1, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=1, N=1)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 1, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};


// sparse
template <typename T>
struct MinusInnerProductSparseMatrix {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  static constexpr uint32_t SEGMENT_ID_BITS = 16;
  static constexpr uint32_t SEGMENT_ID_MASK = 0xFFFF;

  struct SparseSegmentInfo {
   public:
    uint32_t seg_id_{-1U};
    uint32_t vec_cnt_{0};

   public:
    SparseSegmentInfo() : seg_id_{-1U}, vec_cnt_{0} {}

    SparseSegmentInfo(uint32_t seg_id, uint32_t vec_cnt)
        : seg_id_{seg_id}, vec_cnt_{vec_cnt} {}
  };

  static inline void transform_sparse_format(uint32_t sparse_count,
                                             const uint32_t *sparse_index,
                                             const void *sparse_value,
                                             std::string &buffer);

  static inline float ComputeInnerProductSparseInSegment(
      uint32_t m_sparse_count, const uint16_t *m_sparse_index,
      const ValueType *m_sparse_value, uint32_t q_sparse_count,
      const uint16_t *q_sparse_index, const ValueType *q_sparse_value);

  //! Compute the distance between matrix and query
  static inline void Compute(const void *m_sparse_data_in,
                             const void *q_sparse_data_in, float *out) {
    ailego_assert(m_sparse_data_in && q_sparse_data_in && out);

    const uint8_t *m_sparse_data =
        reinterpret_cast<const uint8_t *>(m_sparse_data_in);
    const uint8_t *q_sparse_data =
        reinterpret_cast<const uint8_t *>(q_sparse_data_in);

    const uint32_t m_sparse_count =
        *reinterpret_cast<const uint32_t *>(m_sparse_data);
    const uint32_t q_sparse_count =
        *reinterpret_cast<const uint32_t *>(q_sparse_data);

    if (m_sparse_count == 0 || q_sparse_count == 0) {
      *out = 0;

      return;
    }

    const uint32_t m_seg_count =
        *reinterpret_cast<const uint32_t *>(m_sparse_data + sizeof(uint32_t));
    const uint32_t q_seg_count =
        *reinterpret_cast<const uint32_t *>(q_sparse_data + sizeof(uint32_t));

    const uint32_t *m_seg_id = reinterpret_cast<const uint32_t *>(
        m_sparse_data + 2 * sizeof(uint32_t));
    const uint32_t *q_seg_id = reinterpret_cast<const uint32_t *>(
        q_sparse_data + 2 * sizeof(uint32_t));

    const uint32_t *m_seg_vec_cnt = reinterpret_cast<const uint32_t *>(
        m_sparse_data + 2 * sizeof(uint32_t) + m_seg_count * sizeof(uint32_t));
    const uint32_t *q_seg_vec_cnt = reinterpret_cast<const uint32_t *>(
        q_sparse_data + 2 * sizeof(uint32_t) + q_seg_count * sizeof(uint32_t));

    const uint16_t *m_sparse_index = reinterpret_cast<const uint16_t *>(
        m_sparse_data + 2 * sizeof(uint32_t) +
        m_seg_count * 2 * sizeof(uint32_t));
    const uint16_t *q_sparse_index = reinterpret_cast<const uint16_t *>(
        q_sparse_data + 2 * sizeof(uint32_t) +
        q_seg_count * 2 * sizeof(uint32_t));

    const ValueType *m_sparse_value = reinterpret_cast<const ValueType *>(
        m_sparse_data + 2 * sizeof(uint32_t) +
        m_seg_count * 2 * sizeof(uint32_t) + m_sparse_count * sizeof(uint16_t));
    const ValueType *q_sparse_value = reinterpret_cast<const ValueType *>(
        q_sparse_data + 2 * sizeof(uint32_t) +
        q_seg_count * 2 * sizeof(uint32_t) + q_sparse_count * sizeof(uint16_t));

    float sum = 0.0f;

    size_t m_s = 0;
    size_t q_s = 0;

    size_t m_count = 0;
    size_t q_count = 0;

    while (m_s < m_seg_count && q_s < q_seg_count) {
      if (m_seg_id[m_s] == q_seg_id[q_s]) {
        sum += ComputeInnerProductSparseInSegment(
            m_seg_vec_cnt[m_s], m_sparse_index + m_count,
            m_sparse_value + m_count, q_seg_vec_cnt[q_s],
            q_sparse_index + q_count, q_sparse_value + q_count);

        m_count += m_seg_vec_cnt[m_s];
        q_count += q_seg_vec_cnt[q_s];

        ++m_s;
        ++q_s;
      } else if (m_seg_id[m_s] < q_seg_id[q_s]) {
        m_count += m_seg_vec_cnt[m_s];

        ++m_s;
      } else {
        q_count += q_seg_vec_cnt[q_s];

        ++q_s;
      }
    }

    *out = -sum;
  }
};

template <typename T>
float MinusInnerProductSparseMatrix<T>::ComputeInnerProductSparseInSegment(
    uint32_t m_sparse_count, const uint16_t *m_sparse_index,
    const ValueType *m_sparse_value, uint32_t q_sparse_count,
    const uint16_t *q_sparse_index, const ValueType *q_sparse_value) {
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

template <typename T>
void MinusInnerProductSparseMatrix<T>::transform_sparse_format(
    uint32_t sparse_count, const uint32_t *sparse_index,
    const void *sparse_value, std::string &buffer) {
  uint32_t unit_size = sizeof(T);

  uint32_t seg_count = 0;
  if (sparse_count == 0) {
    buffer.reserve(sizeof(uint32_t) + sizeof(uint32_t));

    buffer.append(reinterpret_cast<const char *>(&sparse_count),
                  sizeof(uint32_t));

    buffer.append(reinterpret_cast<const char *>(&seg_count), sizeof(uint32_t));

    return;
  }

  std::vector<SparseSegmentInfo> seg_infos;

  uint32_t cur_seg_id = -1U;
  uint32_t cur_vec_cnt = 0;

  for (size_t i = 0; i < sparse_count; ++i) {
    uint32_t seg_id = sparse_index[i] >> SEGMENT_ID_BITS;
    if (cur_seg_id == -1U) {
      cur_seg_id = seg_id;
      cur_vec_cnt++;
    } else {
      if (seg_id == cur_seg_id) {
        cur_vec_cnt++;
      } else if (seg_id > cur_seg_id) {
        seg_infos.emplace_back(cur_seg_id, cur_vec_cnt);

        cur_seg_id = seg_id;
        cur_vec_cnt = 1;
      } else {
        // std::abort();
      }
    }
  }

  if (cur_vec_cnt > 0) {
    seg_infos.emplace_back(cur_seg_id, cur_vec_cnt);
  }

  uint32_t buffer_len = 2 * sizeof(uint32_t) +
                        seg_infos.size() * 2 * sizeof(uint32_t) +
                        sparse_count * (sizeof(uint16_t) + sizeof(T));

  buffer.reserve(buffer_len);

  buffer.append(reinterpret_cast<const char *>(&sparse_count),
                sizeof(uint32_t));

  seg_count = seg_infos.size();
  buffer.append(reinterpret_cast<const char *>(&seg_count), sizeof(uint32_t));

  for (size_t i = 0; i < seg_count; ++i) {
    uint32_t seg_id = seg_infos[i].seg_id_;
    buffer.append(reinterpret_cast<const char *>(&seg_id), sizeof(uint32_t));
  }

  for (size_t i = 0; i < seg_count; ++i) {
    uint32_t vec_cnt = seg_infos[i].vec_cnt_;
    buffer.append(reinterpret_cast<const char *>(&vec_cnt), sizeof(uint32_t));
  }

  for (size_t i = 0; i < sparse_count; ++i) {
    uint16_t temp_dim = sparse_index[i] & SEGMENT_ID_MASK;
    buffer.append(reinterpret_cast<const char *>(&temp_dim), sizeof(uint16_t));
  }

  const char *sparse_value_ptr = reinterpret_cast<const char *>(sparse_value);
  for (size_t i = 0; i < sparse_count; ++i) {
    buffer.append(sparse_value_ptr, unit_size);
    sparse_value_ptr += unit_size;
  }
}

#if defined(__SSE4_1__)
template <>
float MinusInnerProductSparseMatrix<float>::ComputeInnerProductSparseInSegment(
    uint32_t m_sparse_count, const uint16_t *m_sparse_index,
    const ValueType *m_sparse_value, uint32_t q_sparse_count,
    const uint16_t *q_sparse_index, const ValueType *q_sparse_value);

template <>
float MinusInnerProductSparseMatrix<Float16>::
    ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                       const uint16_t *m_sparse_index,
                                       const ValueType *m_sparse_value,
                                       uint32_t q_sparse_count,
                                       const uint16_t *q_sparse_index,
                                       const ValueType *q_sparse_value);
#endif

#if defined(__AVX512FP16__)
template <>
float MinusInnerProductSparseMatrix<Float16>::
    ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                       const uint16_t *m_sparse_index,
                                       const ValueType *m_sparse_value,
                                       uint32_t q_sparse_count,
                                       const uint16_t *q_sparse_index,
                                       const ValueType *q_sparse_value);
#endif

}  // namespace ailego
}  // namespace zvec
