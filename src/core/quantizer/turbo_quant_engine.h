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
//
// TurboQuant engine: shared quantization/dequantization logic used by
// the converter, reformer, and metric.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>
#include "rotator/rotator.h"
#include "turbo_quant_codebook.h"
#include "turbo_quant_params.h"

namespace zvec {
namespace core {

// Portable pi constant — MSVC does not define M_PI by default.
constexpr float kPi = 3.14159265358979323846f;

/*! Shared QJL matrix: d x d i.i.d. N(0,1) entries, generated from a seed.
 */
class TurboQuantQjlMatrix {
 public:
  static std::shared_ptr<TurboQuantQjlMatrix> get(size_t dimension,
                                                  uint64_t seed) {
    uint64_t key = (static_cast<uint64_t>(dimension) << 16) | (seed & 0xFFFF);
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto &cache = matrix_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto m = std::make_shared<TurboQuantQjlMatrix>(dimension, seed);
    cache[key] = m;
    return m;
  }

  explicit TurboQuantQjlMatrix(size_t dimension, uint64_t seed)
      : dim_(dimension), data_(dimension * dimension) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] = dist(rng);
    }
  }

  void mat_vec(const float *vec, float *out) const {
    const float *S = data_.data();
    for (size_t i = 0; i < dim_; ++i) {
      float sum = 0.0f;
      const float *row = S + i * dim_;
      for (size_t j = 0; j < dim_; ++j) {
        sum += row[j] * vec[j];
      }
      out[i] = sum;
    }
  }

  void matT_vec_sign(const int8_t *sign_vec, float *out) const {
    const float *S = data_.data();
    std::fill(out, out + dim_, 0.0f);
    for (size_t i = 0; i < dim_; ++i) {
      float s = static_cast<float>(sign_vec[i]);
      const float *row = S + i * dim_;
      for (size_t j = 0; j < dim_; ++j) {
        out[j] += row[j] * s;
      }
    }
  }

  size_t dimension() const {
    return dim_;
  }
  const float *data() const {
    return data_.data();
  }

 private:
  using CacheMap =
      std::unordered_map<uint64_t, std::shared_ptr<TurboQuantQjlMatrix>>;
  static CacheMap &matrix_cache() {
    static CacheMap inst;
    return inst;
  }
  static std::mutex &cache_mutex() {
    static std::mutex m;
    return m;
  }

  size_t dim_;
  std::vector<float> data_;
};

/*! TurboQuant engine: quantization/dequantization logic.
 *
 * Implements both MSE-optimal (Algorithm 1) and inner-product-optimal
 * (Algorithm 2) modes from the TurboQuant paper.
 */
class TurboQuantEngine {
 public:
  TurboQuantEngine(size_t dimension, int bits, bool prod_mode,
                   bool enable_rotate, uint64_t seed)
      : dim_(dimension),
        bits_(bits),
        prod_mode_(prod_mode),
        enable_rotate_(enable_rotate),
        seed_(seed) {
    mse_bits_ = prod_mode_ ? bits_ - 1 : bits_;
    if (mse_bits_ > 0) {
      codebook_ = TurboQuantCodebook::get(dim_, mse_bits_);
    }
    if (enable_rotate_) {
      Rotator::create(&rotator_, dim_);
    }
    if (prod_mode_) {
      qjl_matrix_ = TurboQuantQjlMatrix::get(dim_, seed_);
    }
    mse_packed_bytes_ = TurboQuantPacking::packed_bytes(dim_, mse_bits_);
    qjl_packed_bytes_ =
        prod_mode_ ? TurboQuantPacking::packed_bytes(dim_, 1) : 0;
    extra_bytes_ = sizeof(float);  // norm
    if (prod_mode_) {
      extra_bytes_ += sizeof(float);  // residual norm gamma
    }
    total_bytes_ = mse_packed_bytes_ + qjl_packed_bytes_ + extra_bytes_;
  }

  void set_rotator(std::shared_ptr<Rotator> rotator) {
    rotator_ = std::move(rotator);
  }

  //! Quantize a single FP32 vector into a byte buffer.
  size_t quantize(const float *vec, uint8_t *out) const {
    float norm = 0.0f;
    for (size_t i = 0; i < dim_; ++i) {
      norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    float inv_norm = (norm > 0.0f) ? 1.0f / norm : 0.0f;

    std::vector<float> normalized(dim_);
    for (size_t i = 0; i < dim_; ++i) {
      normalized[i] = vec[i] * inv_norm;
    }

    const float *rotated = normalized.data();
    std::vector<float> rotated_buf;
    if (enable_rotate_ && rotator_) {
      rotated_buf.resize(dim_);
      rotator_->rotate(normalized.data(), rotated_buf.data());
      rotated = rotated_buf.data();
    }

    std::vector<uint32_t> indices(dim_, 0);
    if (mse_bits_ > 0 && codebook_) {
      for (size_t i = 0; i < dim_; ++i) {
        indices[i] = codebook_->quantize(rotated[i]);
      }
    }

    size_t offset = 0;
    if (mse_bits_ > 0) {
      TurboQuantPacking::pack(indices.data(), dim_, mse_bits_, out + offset);
      offset += mse_packed_bytes_;
    }

    if (prod_mode_) {
      std::vector<float> dequant_rotated(dim_, 0.0f);
      if (mse_bits_ > 0 && codebook_) {
        for (size_t i = 0; i < dim_; ++i) {
          dequant_rotated[i] = codebook_->dequantize(indices[i]);
        }
      }

      std::vector<float> dequant_original(dim_, 0.0f);
      if (enable_rotate_ && rotator_) {
        rotator_->unrotate(dequant_rotated.data(), dequant_original.data());
      } else {
        dequant_original = dequant_rotated;
      }

      std::vector<float> residual(dim_);
      for (size_t i = 0; i < dim_; ++i) {
        residual[i] = normalized[i] - dequant_original[i];
      }

      std::vector<float> projected(dim_);
      qjl_matrix_->mat_vec(residual.data(), projected.data());
      std::vector<uint32_t> sign_indices(dim_);
      for (size_t i = 0; i < dim_; ++i) {
        sign_indices[i] = (projected[i] >= 0.0f) ? 1u : 0u;
      }
      TurboQuantPacking::pack(sign_indices.data(), dim_, 1, out + offset);
      offset += qjl_packed_bytes_;

      float gamma = 0.0f;
      for (size_t i = 0; i < dim_; ++i) {
        gamma += residual[i] * residual[i];
      }
      gamma = std::sqrt(gamma);
      std::memcpy(out + offset, &gamma, sizeof(float));
      offset += sizeof(float);
    }

    std::memcpy(out + offset, &norm, sizeof(float));
    offset += sizeof(float);

    return offset;
  }

  //! Dequantize a byte buffer back to FP32.
  void dequantize(const uint8_t *data, float *out) const {
    size_t offset = 0;

    std::vector<uint32_t> indices(dim_, 0);
    if (mse_bits_ > 0) {
      TurboQuantPacking::unpack(data + offset, dim_, mse_bits_, indices.data());
      offset += mse_packed_bytes_;
    }

    std::vector<float> rotated_dequant(dim_, 0.0f);
    if (mse_bits_ > 0 && codebook_) {
      for (size_t i = 0; i < dim_; ++i) {
        rotated_dequant[i] = codebook_->dequantize(indices[i]);
      }
    }

    std::vector<float> mse_reconstructed(dim_, 0.0f);
    if (enable_rotate_ && rotator_) {
      rotator_->unrotate(rotated_dequant.data(), mse_reconstructed.data());
    } else {
      mse_reconstructed = rotated_dequant;
    }

    std::vector<float> qjl_reconstructed;
    if (prod_mode_) {
      std::vector<uint32_t> sign_indices(dim_);
      TurboQuantPacking::unpack(data + offset, dim_, 1, sign_indices.data());
      offset += qjl_packed_bytes_;

      std::vector<int8_t> signs(dim_);
      for (size_t i = 0; i < dim_; ++i) {
        signs[i] = (sign_indices[i] > 0) ? 1 : -1;
      }

      float gamma;
      std::memcpy(&gamma, data + offset, sizeof(float));
      offset += sizeof(float);

      std::vector<float> st_qjl(dim_);
      qjl_matrix_->matT_vec_sign(signs.data(), st_qjl.data());
      float scale = std::sqrt(kPi / 2.0f) / static_cast<float>(dim_) * gamma;
      qjl_reconstructed.resize(dim_);
      for (size_t i = 0; i < dim_; ++i) {
        qjl_reconstructed[i] = scale * st_qjl[i];
      }
    }

    float norm;
    for (size_t i = 0; i < dim_; ++i) {
      out[i] = mse_reconstructed[i];
      if (prod_mode_ && !qjl_reconstructed.empty()) {
        out[i] += qjl_reconstructed[i];
      }
    }

    std::memcpy(&norm, data + offset, sizeof(float));
    for (size_t i = 0; i < dim_; ++i) {
      out[i] *= norm;
    }
  }

  //! Compute SDC inner product between two quantized vectors.
  //! Uses precomputed centroid IP table + approximate QJL IP.
  float sdc_inner_product(const uint8_t *a, const uint8_t *b) const {
    size_t offset = 0;
    float ip = 0.0f;

    // MSE part: sum of c_{idx_a[j]} * c_{idx_b[j]}
    if (mse_bits_ > 0 && codebook_) {
      std::vector<uint32_t> idx_a(dim_), idx_b(dim_);
      TurboQuantPacking::unpack(a + offset, dim_, mse_bits_, idx_a.data());
      TurboQuantPacking::unpack(b + offset, dim_, mse_bits_, idx_b.data());
      offset += mse_packed_bytes_;

      const auto &centroids = codebook_->centroids();
      for (size_t i = 0; i < dim_; ++i) {
        ip += centroids[idx_a[i]] * centroids[idx_b[i]];
      }
    } else {
      offset += mse_packed_bytes_;
    }

    // QJL part (prod mode only)
    if (prod_mode_) {
      std::vector<uint32_t> sign_a(dim_), sign_b(dim_);
      TurboQuantPacking::unpack(a + offset, dim_, 1, sign_a.data());
      TurboQuantPacking::unpack(b + offset, dim_, 1, sign_b.data());
      offset += qjl_packed_bytes_;

      float gamma_a, gamma_b;
      std::memcpy(&gamma_a, a + offset, sizeof(float));
      std::memcpy(&gamma_b, b + offset, sizeof(float));
      offset += sizeof(float);

      // Approximate S·S^T ≈ d·I for large d
      // IP_qjl = (π/(2d)) · γ_a · γ_b · qjl_a^T · (d·I) · qjl_b
      //        = (π/2) · γ_a · γ_b · sum(sign_a * sign_b)
      float sign_ip = 0.0f;
      for (size_t i = 0; i < dim_; ++i) {
        float sa = (sign_a[i] > 0) ? 1.0f : -1.0f;
        float sb = (sign_b[i] > 0) ? 1.0f : -1.0f;
        sign_ip += sa * sb;
      }
      ip +=
          kPi / (2.0f * static_cast<float>(dim_)) * gamma_a * gamma_b * sign_ip;
    }

    // Scale by norms
    float norm_a, norm_b;
    std::memcpy(&norm_a, a + offset, sizeof(float));
    std::memcpy(&norm_b, b + offset, sizeof(float));
    ip *= norm_a * norm_b;

    return ip;
  }

  //! Compute SDC squared L2 distance between two quantized vectors.
  float sdc_squared_l2(const uint8_t *a, const uint8_t *b) const {
    size_t offset = 0;
    float dist = 0.0f;

    if (mse_bits_ > 0 && codebook_) {
      std::vector<uint32_t> idx_a(dim_), idx_b(dim_);
      TurboQuantPacking::unpack(a + offset, dim_, mse_bits_, idx_a.data());
      TurboQuantPacking::unpack(b + offset, dim_, mse_bits_, idx_b.data());
      offset += mse_packed_bytes_;

      const auto &centroids = codebook_->centroids();
      for (size_t i = 0; i < dim_; ++i) {
        float diff = centroids[idx_a[i]] - centroids[idx_b[i]];
        dist += diff * diff;
      }
    } else {
      offset += mse_packed_bytes_;
    }

    // QJL part (prod mode): add QJL distance contribution
    if (prod_mode_) {
      std::vector<uint32_t> sign_a(dim_), sign_b(dim_);
      TurboQuantPacking::unpack(a + offset, dim_, 1, sign_a.data());
      TurboQuantPacking::unpack(b + offset, dim_, 1, sign_b.data());
      offset += qjl_packed_bytes_;

      float gamma_a, gamma_b;
      std::memcpy(&gamma_a, a + offset, sizeof(float));
      std::memcpy(&gamma_b, b + offset, sizeof(float));
      offset += sizeof(float);

      // ||x_qjl_a - x_qjl_b||^2 = ||x_qjl_a||^2 + ||x_qjl_b||^2 - 2*<x_qjl_a,
      // x_qjl_b>
      // ||x_qjl||^2 = (π/(2d)) * γ^2 * ||S^T * qjl||^2 ≈ (π/(2d)) * γ^2 * d =
      // (π/2) * γ^2
      float norm_a_sq = kPi / 2.0f * gamma_a * gamma_a;
      float norm_b_sq = kPi / 2.0f * gamma_b * gamma_b;
      float sign_ip = 0.0f;
      for (size_t i = 0; i < dim_; ++i) {
        float sa = (sign_a[i] > 0) ? 1.0f : -1.0f;
        float sb = (sign_b[i] > 0) ? 1.0f : -1.0f;
        sign_ip += sa * sb;
      }
      float ip_qjl =
          kPi / (2.0f * static_cast<float>(dim_)) * gamma_a * gamma_b * sign_ip;
      dist += norm_a_sq + norm_b_sq - 2.0f * ip_qjl;
    }

    // Scale by norms
    float norm_a, norm_b;
    std::memcpy(&norm_a, a + offset, sizeof(float));
    std::memcpy(&norm_b, b + offset, sizeof(float));
    dist *= norm_a * norm_b;

    return dist;
  }

  // Accessors
  size_t dimension() const {
    return dim_;
  }
  int bits() const {
    return bits_;
  }
  int mse_bits() const {
    return mse_bits_;
  }
  bool prod_mode() const {
    return prod_mode_;
  }
  bool enable_rotate() const {
    return enable_rotate_;
  }
  uint64_t seed() const {
    return seed_;
  }
  size_t total_bytes() const {
    return total_bytes_;
  }
  size_t mse_packed_bytes() const {
    return mse_packed_bytes_;
  }
  size_t qjl_packed_bytes() const {
    return qjl_packed_bytes_;
  }
  size_t extra_bytes() const {
    return extra_bytes_;
  }
  std::shared_ptr<Rotator> rotator() const {
    return rotator_;
  }
  std::shared_ptr<TurboQuantCodebook> codebook() const {
    return codebook_;
  }
  std::shared_ptr<TurboQuantQjlMatrix> qjl_matrix() const {
    return qjl_matrix_;
  }

 private:
  size_t dim_;
  int bits_;
  int mse_bits_;
  bool prod_mode_;
  bool enable_rotate_;
  uint64_t seed_;

  std::shared_ptr<Rotator> rotator_;
  std::shared_ptr<TurboQuantCodebook> codebook_;
  std::shared_ptr<TurboQuantQjlMatrix> qjl_matrix_;

  size_t mse_packed_bytes_{0};
  size_t qjl_packed_bytes_{0};
  size_t extra_bytes_{0};
  size_t total_bytes_{0};
};

}  // namespace core
}  // namespace zvec
