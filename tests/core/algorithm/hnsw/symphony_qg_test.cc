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

#include <gtest/gtest.h>
#if RABITQ_SUPPORTED
#include <rabitqlib/quantization/rabitq.hpp>
#include <rabitqlib/utils/cpu_features.hpp>
#include "symphony_qg_utils.h"

namespace zvec::core {

TEST(SymphonyQGTest, FastScanHandlesPartialBlocksDuplicatesAndLargeDimensions) {
  if (!rabitqlib::cpu::has_avx2() && !rabitqlib::cpu::has_avx512_core()) {
    GTEST_SKIP() << "RaBitQ requires AVX2/FMA or AVX512";
  }
  for (size_t dim : {64U, 128U, 1024U, 2048U, 4096U}) {
    for (size_t count : {1U, 17U, 32U}) {
      SCOPED_TRACE(std::to_string(dim) + "/" + std::to_string(count));
      // Constant positive query deliberately produces large LUT sums at 4096D.
      std::vector<float> query(dim, 1.0f), center(dim, 0.5f);
      std::vector<float> data(count * dim);
      for (size_t i = 0; i < count; ++i) {
        std::fill(data.begin() + i * dim, data.begin() + (i + 1) * dim,
                  0.5f + i * 0.125f);
      }
      std::vector<char> codes(rabitqlib::QGBatchDataMap<float>::data_bytes(dim),
                              0);
      rabitqlib::quant::quantize_qg_batch(data.data(), center.data(), count,
                                          dim, codes.data(),
                                          rabitqlib::METRIC_L2);
      rabitqlib::BatchQuery<float> q(query.data(), dim);
      q.set_g_add(0.25f * dim);
      std::array<float, 32> result;
      ScanSymphonyQGBatch(codes.data(), q, dim, result.data());
      for (size_t i = 0; i < count; ++i) {
        const float diff = 0.5f + i * 0.125f - 1.0f;
        EXPECT_NEAR(diff * diff * dim, result[i], 0.01f * dim);
        EXPECT_TRUE(std::isfinite(result[i]));
      }
    }
  }
}

}  // namespace zvec::core

#endif
