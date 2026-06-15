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

#include "quantizer/rotator/rotator.h"
#include <cmath>
#include <random>
#include <vector>
#include <gtest/gtest.h>

using namespace zvec;
using namespace zvec::turbo;

namespace {

float Norm(const float *v, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) {
    s += v[i] * v[i];
  }
  return std::sqrt(s);
}

}  // namespace

TEST(Rotator, MatrixRoundTripAndNorm) {
  const int DIM = 16;
  auto rotator = CreateRotator(RotatorType::kMatrix, DIM);
  ASSERT_TRUE(rotator);
  rotator->train(nullptr, 0, 0);

  EXPECT_EQ(RotatorType::kMatrix, rotator->type());
  EXPECT_EQ(DIM, rotator->in_dim());
  EXPECT_EQ(DIM, rotator->out_dim());

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> in(DIM);
  for (int i = 0; i < DIM; ++i) {
    in[i] = dist(gen);
  }

  std::vector<float> rotated(DIM);
  rotator->apply(in.data(), rotated.data());

  // Orthogonal rotation preserves the L2 norm.
  EXPECT_NEAR(Norm(in.data(), DIM), Norm(rotated.data(), DIM), 1e-3);

  // apply then apply_inverse recovers the input.
  std::vector<float> back(DIM);
  rotator->apply_inverse(rotated.data(), back.data());
  for (int i = 0; i < DIM; ++i) {
    EXPECT_NEAR(in[i], back[i], 1e-4);
  }
}

TEST(Rotator, MatrixSerializeRestore) {
  const int DIM = 16;
  auto rotator = CreateRotator(RotatorType::kMatrix, DIM);
  ASSERT_TRUE(rotator);
  rotator->train(nullptr, 0, 0);

  std::string blob;
  ASSERT_EQ(0, rotator->serialize(&blob));
  ASSERT_GE(blob.size(), sizeof(RotatorSerHeader));
  const auto *header = reinterpret_cast<const RotatorSerHeader *>(blob.data());
  EXPECT_EQ(kRotatorMagic, header->magic);
  EXPECT_EQ(kRotatorSerVersion, header->version);
  EXPECT_EQ(static_cast<uint16_t>(RotatorType::kMatrix), header->rotator_type);
  EXPECT_EQ(static_cast<uint32_t>(DIM), header->in_dim);
  EXPECT_EQ(static_cast<uint32_t>(DIM), header->out_dim);

  auto restored = CreateRotatorFromBlob(blob.data(), blob.size());
  ASSERT_TRUE(restored);
  EXPECT_EQ(RotatorType::kMatrix, restored->type());

  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> in(DIM);
  for (int i = 0; i < DIM; ++i) {
    in[i] = dist(gen);
  }
  std::vector<float> a(DIM);
  std::vector<float> b(DIM);
  rotator->apply(in.data(), a.data());
  restored->apply(in.data(), b.data());
  for (int i = 0; i < DIM; ++i) {
    EXPECT_NEAR(a[i], b[i], 1e-5);
  }
}

TEST(Rotator, FhtPowerOfTwoRoundTrip) {
  const int DIM = 16;  // already a power of two
  auto rotator = CreateRotator(RotatorType::kFht, DIM);
  ASSERT_TRUE(rotator);
  rotator->train(nullptr, 0, 0);

  EXPECT_EQ(RotatorType::kFht, rotator->type());
  EXPECT_EQ(DIM, rotator->in_dim());
  EXPECT_EQ(DIM, rotator->out_dim());

  std::mt19937 gen(99);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> in(DIM);
  for (int i = 0; i < DIM; ++i) {
    in[i] = dist(gen);
  }
  std::vector<float> rotated(rotator->out_dim());
  rotator->apply(in.data(), rotated.data());

  // Norm preserved for the power-of-two (no padding) case.
  EXPECT_NEAR(Norm(in.data(), DIM), Norm(rotated.data(), rotator->out_dim()),
              1e-3);

  std::vector<float> back(DIM);
  rotator->apply_inverse(rotated.data(), back.data());
  for (int i = 0; i < DIM; ++i) {
    EXPECT_NEAR(in[i], back[i], 1e-4);
  }
}

TEST(Rotator, FhtPaddedRoundTripAndSerialize) {
  const int DIM = 12;  // padded up to 16
  auto rotator = CreateRotator(RotatorType::kFht, DIM);
  ASSERT_TRUE(rotator);
  rotator->train(nullptr, 0, 0);

  EXPECT_EQ(DIM, rotator->in_dim());
  EXPECT_EQ(16, rotator->out_dim());

  std::mt19937 gen(2024);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> in(DIM);
  for (int i = 0; i < DIM; ++i) {
    in[i] = dist(gen);
  }
  std::vector<float> rotated(rotator->out_dim());
  rotator->apply(in.data(), rotated.data());

  std::vector<float> back(DIM);
  rotator->apply_inverse(rotated.data(), back.data());
  for (int i = 0; i < DIM; ++i) {
    EXPECT_NEAR(in[i], back[i], 1e-4);
  }

  // Serialize and restore through the factory; apply must match.
  std::string blob;
  ASSERT_EQ(0, rotator->serialize(&blob));
  const auto *header = reinterpret_cast<const RotatorSerHeader *>(blob.data());
  EXPECT_EQ(static_cast<uint16_t>(RotatorType::kFht), header->rotator_type);
  EXPECT_EQ(static_cast<uint32_t>(DIM), header->in_dim);
  EXPECT_EQ(16u, header->out_dim);

  auto restored = CreateRotatorFromBlob(blob.data(), blob.size());
  ASSERT_TRUE(restored);
  std::vector<float> rotated2(restored->out_dim());
  restored->apply(in.data(), rotated2.data());
  for (int i = 0; i < restored->out_dim(); ++i) {
    EXPECT_NEAR(rotated[i], rotated2[i], 1e-5);
  }
}
