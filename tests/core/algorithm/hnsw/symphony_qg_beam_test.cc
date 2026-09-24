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

#include "symphony_qg_beam.h"
#include <gtest/gtest.h>

namespace zvec::core {

TEST(SymphonyQGTest, BeamKeepsFullWidthIdsAndRevisitsBetterEstimates) {
  SymphonyQGBeam beam(2);
  beam.insert(0x80000001U, 5);
  beam.insert(2, 10);
  EXPECT_EQ(0x80000001U, beam.pop());
  beam.insert(3, 20);  // beyond the beam
  beam.insert(2, 1);   // a better estimate from a different center
  ASSERT_TRUE(beam.has_next());
  EXPECT_EQ(2U, beam.pop());
  EXPECT_FALSE(beam.has_next());
  beam.insert(4, 0);  // can reopen a previously exhausted beam
  EXPECT_EQ(4U, beam.pop());
  EXPECT_FALSE(beam.has_next());
}

TEST(SymphonyQGTest, BeamCapacityAndEqualEstimates) {
  SymphonyQGBeam beam(0);
  beam.insert(1, 5);
  beam.insert(2, 4);
  EXPECT_EQ(2U, beam.pop());
  EXPECT_FALSE(beam.has_next());
  beam.insert(3, 4);
  ASSERT_TRUE(beam.has_next());
  EXPECT_EQ(3U, beam.pop());
  EXPECT_FALSE(beam.has_next());
}

}  // namespace zvec::core
