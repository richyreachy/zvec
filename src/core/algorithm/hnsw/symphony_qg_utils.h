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
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/quantization/data_layout.hpp>
#include "symphony_qg_beam.h"

namespace zvec::core {

// The upstream QG estimator accumulates a whole vector into uint16_t.
// At most 256 four-dimensional LUT entries (1024 dimensions) fit without
// overflow: 256 * 255 < 65536. Accumulate slices and widen between slices to
// support all zvec RaBitQ dimensions, including padding to 4096.
inline void ScanSymphonyQGBatch(const char *codes,
                                const rabitqlib::BatchQuery<float> &query,
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
