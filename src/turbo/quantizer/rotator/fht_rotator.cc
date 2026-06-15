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

#include "quantizer/rotator/fht_rotator.h"
#include <cmath>
#include <cstring>
#include <random>
#include <zvec/core/framework/index_error.h>

namespace zvec {
namespace turbo {

using zvec::core::IndexError_InvalidArgument;

namespace {

//! Smallest power of two >= v (returns 0 for v <= 0, 1 for v == 1).
int NextPow2(int v) {
  if (v <= 0) {
    return 0;
  }
  int n = 1;
  while (n < v) {
    n <<= 1;
  }
  return n;
}

//! In-place, unnormalized fast Walsh-Hadamard transform on n (power-of-two)
//! elements.
void Fwht(float *buf, int n) {
  for (int len = 1; len < n; len <<= 1) {
    for (int i = 0; i < n; i += (len << 1)) {
      for (int j = i; j < i + len; ++j) {
        const float a = buf[j];
        const float b = buf[j + len];
        buf[j] = a + b;
        buf[j + len] = a - b;
      }
    }
  }
}

}  // namespace

FhtRotator::FhtRotator(int in_dim, uint64_t seed)
    : in_dim_(in_dim), out_dim_(NextPow2(in_dim)), seed_(seed) {
  rebuild();
}

void FhtRotator::rebuild() {
  signs_.assign(static_cast<size_t>(out_dim_ > 0 ? out_dim_ : 0), 1.0f);
  if (out_dim_ <= 0) {
    return;
  }
  std::mt19937_64 gen(seed_);
  for (int i = 0; i < out_dim_; ++i) {
    signs_[static_cast<size_t>(i)] = (gen() & 1ull) ? 1.0f : -1.0f;
  }
}

void FhtRotator::apply(const float *in, float *out) const {
  if (out_dim_ <= 0) {
    return;
  }
  // Zero-pad to out_dim_ and apply the Rademacher sign flips.
  for (int i = 0; i < out_dim_; ++i) {
    const float v = (i < in_dim_) ? in[i] : 0.0f;
    out[i] = v * signs_[static_cast<size_t>(i)];
  }
  Fwht(out, out_dim_);
  const float scale = 1.0f / std::sqrt(static_cast<float>(out_dim_));
  for (int i = 0; i < out_dim_; ++i) {
    out[i] *= scale;
  }
}

void FhtRotator::apply_inverse(const float *in, float *out) const {
  if (out_dim_ <= 0) {
    return;
  }
  std::vector<float> buf(static_cast<size_t>(out_dim_));
  std::memcpy(buf.data(), in, static_cast<size_t>(out_dim_) * sizeof(float));
  Fwht(buf.data(), out_dim_);
  const float scale = 1.0f / std::sqrt(static_cast<float>(out_dim_));
  for (int i = 0; i < out_dim_; ++i) {
    buf[static_cast<size_t>(i)] *= scale * signs_[static_cast<size_t>(i)];
  }
  // Recover only the original (un-padded) components.
  for (int i = 0; i < in_dim_; ++i) {
    out[i] = buf[static_cast<size_t>(i)];
  }
}

void FhtRotator::train(const void * /*data*/, size_t /*num*/,
                       size_t /*stride*/) {
  // The Hadamard rotation is fully determined by in_dim_ and seed_; there is
  // nothing to fit from data. Ensure signs are materialized.
  rebuild();
}

int FhtRotator::serialize(std::string *out) const {
  if (!out) {
    return IndexError_InvalidArgument;
  }
  const uint32_t payload_size = static_cast<uint32_t>(sizeof(uint64_t));
  out->resize(sizeof(RotatorSerHeader) + payload_size);

  RotatorSerHeader *header = reinterpret_cast<RotatorSerHeader *>(&(*out)[0]);
  header->magic = kRotatorMagic;
  header->version = kRotatorSerVersion;
  header->rotator_type = static_cast<uint16_t>(RotatorType::kFht);
  header->in_dim = static_cast<uint32_t>(in_dim_);
  header->out_dim = static_cast<uint32_t>(out_dim_);
  header->payload_size = payload_size;
  header->reserved = 0;

  std::memcpy(&(*out)[sizeof(RotatorSerHeader)], &seed_, sizeof(uint64_t));
  return 0;
}

int FhtRotator::deserialize(const void *data, size_t len) {
  if (!data || len < sizeof(RotatorSerHeader)) {
    return IndexError_InvalidArgument;
  }
  const RotatorSerHeader *header =
      reinterpret_cast<const RotatorSerHeader *>(data);
  if (header->magic != kRotatorMagic || header->version != kRotatorSerVersion ||
      header->rotator_type != static_cast<uint16_t>(RotatorType::kFht) ||
      header->payload_size != sizeof(uint64_t) ||
      len < sizeof(RotatorSerHeader) + sizeof(uint64_t)) {
    return IndexError_InvalidArgument;
  }

  in_dim_ = static_cast<int>(header->in_dim);
  out_dim_ = static_cast<int>(header->out_dim);
  std::memcpy(&seed_,
              reinterpret_cast<const char *>(data) + sizeof(RotatorSerHeader),
              sizeof(uint64_t));
  rebuild();
  return 0;
}

}  // namespace turbo
}  // namespace zvec
