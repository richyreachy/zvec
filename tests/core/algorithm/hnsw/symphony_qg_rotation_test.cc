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
#include "symphony_qg_rotation.h"
#include <gtest/gtest.h>

namespace zvec::core {
TEST(SymphonyQGRotationTest, PreservesDistancesAndRebuildsIdentically) {
  for (size_t dim : {1U, 16U, 63U, 64U, 65U, 128U, 1025U, 4096U}) {
    SCOPED_TRACE(dim);
    SymphonyQGRotation rotation(dim), reopened(dim);
    std::vector<float> a(dim), b(dim), ra, rb, repeat;
    double original = 0;
    for (size_t i = 0; i < dim; ++i) {
      a[i] = std::sin(static_cast<float>(i));
      b[i] = std::cos(static_cast<float>(i) * 0.7f);
      original += (a[i] - b[i]) * (a[i] - b[i]);
    }
    rotation.rotate(a.data(), ra);
    rotation.rotate(b.data(), rb);
    reopened.rotate(a.data(), repeat);
    EXPECT_EQ(ra, repeat);
    double transformed = 0;
    for (size_t i = 0; i < ra.size(); ++i) {
      transformed += (ra[i] - rb[i]) * (ra[i] - rb[i]);
    }
    EXPECT_NEAR(original, transformed, 1e-5 * std::max(1.0, original));
  }
}
}  // namespace zvec::core
