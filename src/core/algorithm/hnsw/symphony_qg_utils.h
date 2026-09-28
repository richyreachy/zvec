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
#include <cstdint>
#include <numeric>
#include <vector>
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/quantization/data_layout.hpp>
#include "symphony_qg_beam.h"
#include "symphony_qg_kernels.h"

namespace zvec::core {

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

// The upstream QG estimator accumulates a whole vector into uint16_t.
// At most 256 four-dimensional LUT entries (1024 dimensions) fit without
// overflow: 256 * 255 < 65536. Accumulate slices and widen between slices to
// support all zvec RaBitQ dimensions, including padding to 4096.
inline void ScanSymphonyQGBatch(const char *codes, const SymQuery &query,
                                size_t padded_dim, float *distances) {
  constexpr size_t batch_size = rabitqlib::fastscan::kBatchSize;
  rabitqlib::ConstQGBatchDataMap<float> batch(codes, padded_dim);
  std::array<uint32_t, batch_size> sums{};
  std::array<uint16_t, batch_size> partial;
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

}  // namespace zvec::core
