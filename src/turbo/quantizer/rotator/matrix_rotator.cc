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

#include "quantizer/rotator/matrix_rotator.h"
#include <cmath>
#include <cstring>
#include <random>
#include <zvec/core/framework/index_error.h>

namespace zvec {
namespace turbo {

using zvec::core::IndexError_InvalidArgument;

void MatrixRotator::apply(const float *in, float *out) const {
  for (int i = 0; i < dim_; ++i) {
    const float *row = &mat_[static_cast<size_t>(i) * dim_];
    float acc = 0.0f;
    for (int j = 0; j < dim_; ++j) {
      acc += row[j] * in[j];
    }
    out[i] = acc;
  }
}

void MatrixRotator::apply_inverse(const float *in, float *out) const {
  // For an orthonormal matrix the inverse is the transpose: out = R^T * in.
  for (int j = 0; j < dim_; ++j) {
    out[j] = 0.0f;
  }
  for (int i = 0; i < dim_; ++i) {
    const float *row = &mat_[static_cast<size_t>(i) * dim_];
    const float v = in[i];
    for (int j = 0; j < dim_; ++j) {
      out[j] += row[j] * v;
    }
  }
}

void MatrixRotator::train(const void * /*data*/, size_t /*num*/,
                          size_t /*stride*/) {
  // Random-rotation baseline: build a Gaussian matrix and orthonormalize it
  // with the (modified) Gram-Schmidt process. The data is intentionally
  // ignored here; data-driven rotations (e.g. OPQ) can override this later.
  if (dim_ <= 0) {
    return;
  }

  std::mt19937 gen(0x9E3779B9u);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  mat_.assign(static_cast<size_t>(dim_) * dim_, 0.0f);
  for (auto &v : mat_) {
    v = dist(gen);
  }

  // Modified Gram-Schmidt over the rows.
  for (int i = 0; i < dim_; ++i) {
    float *row_i = &mat_[static_cast<size_t>(i) * dim_];
    for (int k = 0; k < i; ++k) {
      const float *row_k = &mat_[static_cast<size_t>(k) * dim_];
      float dot = 0.0f;
      for (int j = 0; j < dim_; ++j) {
        dot += row_i[j] * row_k[j];
      }
      for (int j = 0; j < dim_; ++j) {
        row_i[j] -= dot * row_k[j];
      }
    }
    float norm = 0.0f;
    for (int j = 0; j < dim_; ++j) {
      norm += row_i[j] * row_i[j];
    }
    norm = std::sqrt(norm);
    if (norm < 1e-12f) {
      norm = 1.0f;
    }
    const float inv = 1.0f / norm;
    for (int j = 0; j < dim_; ++j) {
      row_i[j] *= inv;
    }
  }
}

int MatrixRotator::serialize(std::string *out) const {
  if (!out) {
    return IndexError_InvalidArgument;
  }
  const uint32_t payload_size =
      static_cast<uint32_t>(mat_.size() * sizeof(float));
  out->resize(sizeof(RotatorSerHeader) + payload_size);

  RotatorSerHeader *header = reinterpret_cast<RotatorSerHeader *>(&(*out)[0]);
  header->magic = kRotatorMagic;
  header->version = kRotatorSerVersion;
  header->rotator_type = static_cast<uint16_t>(RotatorType::kMatrix);
  header->in_dim = static_cast<uint32_t>(dim_);
  header->out_dim = static_cast<uint32_t>(dim_);
  header->payload_size = payload_size;
  header->reserved = 0;

  if (payload_size > 0) {
    std::memcpy(&(*out)[sizeof(RotatorSerHeader)], mat_.data(), payload_size);
  }
  return 0;
}

int MatrixRotator::deserialize(const void *data, size_t len) {
  if (!data || len < sizeof(RotatorSerHeader)) {
    return IndexError_InvalidArgument;
  }
  const RotatorSerHeader *header =
      reinterpret_cast<const RotatorSerHeader *>(data);
  if (header->magic != kRotatorMagic || header->version != kRotatorSerVersion ||
      header->rotator_type != static_cast<uint16_t>(RotatorType::kMatrix) ||
      header->in_dim != header->out_dim) {
    return IndexError_InvalidArgument;
  }

  const uint32_t dim = header->in_dim;
  const size_t expect = static_cast<size_t>(dim) * dim * sizeof(float);
  if (header->payload_size != expect ||
      len < sizeof(RotatorSerHeader) + expect) {
    return IndexError_InvalidArgument;
  }

  dim_ = static_cast<int>(dim);
  mat_.resize(static_cast<size_t>(dim_) * dim_);
  if (!mat_.empty()) {
    std::memcpy(mat_.data(),
                reinterpret_cast<const char *>(data) + sizeof(RotatorSerHeader),
                expect);
  }
  return 0;
}

}  // namespace turbo
}  // namespace zvec
