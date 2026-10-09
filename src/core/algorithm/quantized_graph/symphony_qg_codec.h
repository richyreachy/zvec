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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>
#include "symphony_qg_kernels.h"
#if RABITQ_SUPPORTED
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/quantization/rabitq.hpp>
#endif

namespace zvec::core {

// Randomized normalized Hadamard passes form an orthogonal rotation. A single
// pass matches upstream SymphonyQG's rotator; more passes improve the
// Gaussianity of rotated coordinates at proportional query-time cost.
// Fixed signs make derived codes reproducible after reopen without changing
// the source graph storage format. Zero padding preserves original L2
// distances. The 1/sqrt(padded) normalization is folded into the signs so the
// hot path is one element-wise multiply plus the transform.
class SymphonyQGRotation {
 public:
  static constexpr size_t kPasses = 1;
  explicit SymphonyQGRotation(size_t dimension) : dimension_(dimension) {
    padded_ = 64;
    while (padded_ < dimension) padded_ *= 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(padded_));
    std::mt19937 rng(0x535147U);
    signs_.resize(kPasses * padded_);
    for (auto &sign : signs_) sign = ((rng() & 1U) ? 1.0f : -1.0f) * scale;
  }
  size_t padded_dim() const {
    return padded_;
  }
  void rotate(const void *data, std::vector<float> &out) const {
    const auto *values = static_cast<const float *>(data);
    out.assign(padded_, 0.0f);
    std::copy(values, values + dimension_, out.begin());
    for (size_t pass = 0; pass < kPasses; ++pass) {
      const float *signs = signs_.data() + pass * padded_;
      if (!symqg_mul(signs, out.data(), padded_)) {
        for (size_t i = 0; i < padded_; ++i) out[i] *= signs[i];
      }
      if (!symqg_hadamard(out.data(), padded_)) {
        for (size_t stride = 1; stride < padded_; stride *= 2) {
          for (size_t offset = 0; offset < padded_; offset += 2 * stride) {
            for (size_t i = 0; i < stride; ++i) {
              float a = out[offset + i], b = out[offset + i + stride];
              out[offset + i] = a + b;
              out[offset + i + stride] = a - b;
            }
          }
        }
      }
    }
  }

 private:
  size_t dimension_, padded_;
  std::vector<float> signs_;
};

// Convert node-centered squared-L2 coefficients to 1 - dot(query, neighbor).
// With g_cos = 1 - dot(query, center), the additive correction is
// (f_add_l2 + ||center||^2 - ||neighbor||^2) / 2. Keeping both norms also
// handles zero vectors, for which simply halving L2 would be incorrect.
// The caller supplies normalized coordinates, without any norm metadata.
template <class AddFactors, class ScaleFactors>
inline void ConvertSymphonyQGCosineFactors(const float *center,
                                           const float *vectors, size_t count,
                                           size_t dimension, AddFactors f_add,
                                           ScaleFactors f_rescale) {
  const float center_norm =
      std::inner_product(center, center + dimension, center, 0.0f);
  for (size_t i = 0; i < count; ++i) {
    const float *vector = vectors + i * dimension;
    const float norm =
        std::inner_product(vector, vector + dimension, vector, 0.0f);
    f_add[i] = 0.5f * (f_add[i] + center_norm - norm);
    f_rescale[i] = 0.5f * f_rescale[i];
  }
}

#if RABITQ_SUPPORTED


// Query-side state for the SymphonyQG fastscan hot loop. Produces the same
// 8-bit LUT, delta and correction terms as rabitqlib's Lut/BatchQuery: the
// AVX512 kernel builds them in two fused passes over the rotated query (the
// 16 entries of each 4-dim codebook are the full subset-sum lattice of four
// query values), and the rabitqlib Lut path is the reference fallback.
class SymQuery {
 public:
  void reset(const float *rotated_query, size_t padded_dim) {
    g_add_ = 0;
    lut_storage_.resize(padded_dim << 2);
    if (symqg_lut_build(rotated_query, padded_dim, lut_storage_.data(), &delta_,
                        &sum_vl_, &k1xsumq_)) {
      lut_ = lut_storage_.data();
      return;
    }
    ref_lut_.reset(rotated_query, padded_dim);
    lut_ = ref_lut_.lut();
    delta_ = ref_lut_.delta();
    sum_vl_ = ref_lut_.sum_vl();
    const float sumq =
        std::accumulate(rotated_query, rotated_query + padded_dim, 0.0f);
    k1xsumq_ = sumq * -0.5f;
  }

  [[nodiscard]] const uint8_t *lut() const {
    return lut_;
  }
  [[nodiscard]] float delta() const {
    return delta_;
  }
  [[nodiscard]] float sum_vl_lut() const {
    return sum_vl_;
  }
  [[nodiscard]] float k1xsumq() const {
    return k1xsumq_;
  }
  [[nodiscard]] float g_add() const {
    return g_add_;
  }
  void set_g_add(float dist) {
    g_add_ = dist;
  }

 private:
  std::vector<uint8_t> lut_storage_;
  rabitqlib::Lut<float> ref_lut_;
  const uint8_t *lut_ = nullptr;
  float delta_ = 0.0f;
  float sum_vl_ = 0.0f;
  float k1xsumq_ = 0.0f;
  float g_add_ = 0.0f;
};

// RaBitQ 0.1 accumulates into uint16_t; 0.3.8 uses int32_t.
// At most 256 four-dimensional LUT entries (1024 dimensions) fit without
// overflow: 256 * 255 < 65536. Accumulate slices and widen between slices to
// support all zvec RaBitQ dimensions, including padding to 4096.
inline void ScanSymphonyQGBatch(const char *codes, const SymQuery &query,
                                size_t padded_dim, float *distances) {
  constexpr size_t batch_size = rabitqlib::fastscan::kBatchSize;
  rabitqlib::ConstQGBatchDataMap<float> batch(codes, padded_dim);
  std::array<uint32_t, batch_size> sums{};
  // Match the installed FastScan API while retaining bounded slices for 0.1.
  using Accumulator = std::conditional_t<
      std::is_invocable_v<decltype(&rabitqlib::fastscan::accumulate),
                          const uint8_t *, const uint8_t *, int32_t *, size_t>,
      int32_t, uint16_t>;
  std::array<Accumulator, batch_size> partial;
  for (size_t dim = 0; dim < padded_dim; dim += 1024) {
    rabitqlib::fastscan::accumulate(batch.bin_code() + dim * 4,
                                    query.lut() + dim * 4, partial.data(),
                                    std::min(size_t{1024}, padded_dim - dim));
    for (size_t i = 0; i < batch_size; ++i) sums[i] += partial[i];
  }
  for (size_t i = 0; i < batch_size; ++i) {
    distances[i] =
        batch.f_add()[i] + query.g_add() +
        batch.f_rescale()[i] *
            (query.delta() * sums[i] + query.sum_vl_lut() + query.k1xsumq());
  }
}


// Codec policy for QuantizedGraph: all RaBitQ and rotation details live here.
class SymphonyQGCodec {
 public:
  static constexpr size_t kBatchSize = rabitqlib::fastscan::kBatchSize;
  struct Query {
    std::vector<float> rotated;
    SymQuery lookup;
  };

  explicit SymphonyQGCodec(size_t dimension, bool cosine = false)
      : rotation_(dimension), cosine_(cosine) {}

  size_t encoded_dim() const {
    return rotation_.padded_dim();
  }
  size_t batch_bytes() const {
    return rabitqlib::QGBatchDataMap<float>::data_bytes(encoded_dim());
  }
  void transform(const void *vector, std::vector<float> &out) const {
    rotation_.rotate(vector, out);
  }
  void encode(const float *center, const float *vectors, size_t count,
              char *codes) const {
    rabitqlib::quant::quantize_qg_batch(vectors, center, count, encoded_dim(),
                                        codes, rabitqlib::METRIC_L2);
    if (cosine_) {
      rabitqlib::QGBatchDataMap<float> batch(codes, encoded_dim());
      ConvertSymphonyQGCosineFactors(center, vectors, count, encoded_dim(),
                                     batch.f_add(), batch.f_rescale());
    }
  }
  void prepare_query(const void *vector, Query &query) const {
    transform(vector, query.rotated);
    query.lookup.reset(query.rotated.data(), encoded_dim());
  }
  void scan(const char *codes, Query &query, float center_distance,
            float *distances) const {
    query.lookup.set_g_add(center_distance);
    ScanSymphonyQGBatch(codes, query.lookup, encoded_dim(), distances);
  }

 private:
  SymphonyQGRotation rotation_;
  bool cosine_;
};
#endif
}  // namespace zvec::core
