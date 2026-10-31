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
// Research prototype: block quantization with an optional shared low-rank
// anchor.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <turbo/distance/common/symqg_distance.h>
#include <turbo/preprocessor/symqg_rotator/symqg_rotator.h>
#include <zvec/ailego/utility/float_helper.h>

namespace zvec::turbo {
#if RABITQ_SUPPORTED
class SymQGBlockQuantizer {
 public:
  static constexpr size_t kBatchSize = rabitqlib::fastscan::kBatchSize;
  struct Query {
    std::vector<float> rotated;
    SymQGQuery lookup;
    float mean_distance = 0;
    std::vector<float> rank_query;
  };
  static constexpr size_t kMaxModelBytes = 1 << 20;
  static std::string read_model(const std::string &path) {
    if (path.empty()) return {};
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto bytes = file.tellg();
    if (!file || bytes <= 0 || uint64_t(bytes) > kMaxModelBytes)
      throw std::runtime_error("invalid block codebook file size");
    std::string data(size_t(bytes), '\0');
    file.seekg(0);
    if (!file.read(data.data(), data.size()))
      throw std::runtime_error("cannot read block codebook file");
    return data;
  }
  // An explicit byte string (including empty/native) never consults the
  // environment. Persisted snapshots must use exactly their embedded codec.
  explicit SymQGBlockQuantizer(size_t dimension, bool cosine = false,
                               const std::string *model_data = nullptr)
      : rotation_(dimension), cosine_(cosine) {
    if (model_data)
      model_data_ = *model_data;
    else if (const char *path = std::getenv("QG_BLOCK_MODEL"))
      model_data_ = read_model(path);
    if (model_data_.empty()) return;
    if (model_data_.size() > kMaxModelBytes)
      throw std::runtime_error("block codebook model too large");
    std::istringstream file(model_data_, std::ios::in | std::ios::binary);
    uint32_t header[5]{};
    file.read(reinterpret_cast<char *>(header), sizeof(header));
    file.read(reinterpret_cast<char *>(&lambda_), sizeof(lambda_));
    if (!file || header[0] != 0x51474243 || header[1] != 1 ||
        header[2] != dimension || (header[4] != 1 && header[4] != 3) ||
        (header[3] != 4 && header[3] != 6 && header[3] != 8) ||
        !std::isfinite(lambda_) || lambda_ < 0 || lambda_ > 1)
      throw std::runtime_error("invalid block codebook model");
    block_dim_ = header[3];
    groups_ = (encoded_dim() + block_dim_ - 1) / block_dim_;
    code_dim_ = ((groups_ * 4 + 63) / 64) * 64;
    std::vector<float> mean(dimension);
    words_.resize(block_dim_ * 16);
    file.read(reinterpret_cast<char *>(mean.data()), dimension * sizeof(float));
    file.read(reinterpret_cast<char *>(words_.data()),
              words_.size() * sizeof(float));
    if (header[4] == 3) {
      uint32_t rank = 0;
      file.read(reinterpret_cast<char *>(&rank), 4);
      if (!file || (rank != 8 && rank != 16 && rank != 32))
        throw std::runtime_error("low-rank model shape");
      rank_ = rank;
      rank_basis_.resize(rank_ * encoded_dim());
      file.read(reinterpret_cast<char *>(rank_basis_.data()),
                4 * rank_basis_.size());
      for (float value : rank_basis_)
        if (!std::isfinite(value)) throw std::runtime_error("low-rank matrix");
    }
    if (!file || file.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("truncated block codebook model");
    for (float v : mean)
      if (!std::isfinite(v)) throw std::runtime_error("nonfinite mean");
    columns_.resize(words_.size());
    double total = 0;
    for (size_t i = 0; i < 16; ++i) {
      double norm = 0;
      for (size_t d = 0; d < block_dim_; ++d) {
        const float v = words_[i * block_dim_ + d];
        if (!std::isfinite(v)) throw std::runtime_error("nonfinite codeword");
        columns_[d * 16 + i] = v;
        norm += double(v) * v;
      }
      word_norm2_[i] = norm;
      total += norm;
    }
    structured_ = block_dim_ == 6 || block_dim_ == 8;
    for (size_t j = 0; j < 16 && structured_; ++j)
      for (size_t d = 0; d < block_dim_; ++d) {
        const float expected =
            (__builtin_popcount(unsigned(d & (j & 7))) & 1 ? -1.f : 1.f) *
            (j < 8 ? 1.f : -1.f);
        if (words_[j * block_dim_ + d] != expected) structured_ = false;
      }
    bonus_scale_ = float(2 * std::sqrt(total / (16 * block_dim_)));
    rotation_.rotate(mean.data(), mean_);
  }
  size_t encoded_dim() const {
    return rotation_.padded_dim();
  }
  const std::string &model_data() const {
    return model_data_;
  }
  size_t quantized_dim() const {
    return block_dim_ ? code_dim_ : encoded_dim();
  }
  // One scale and FP16 coefficients per 32-neighbor batch. No full local
  // center.
  size_t anchor_bytes() const {
    return rank_ ? 16 + rank_ * 2 : 0;
  }
  size_t batch_bytes() const {
    return anchor_bytes() +
           rabitqlib::QGBatchDataMap<float>::data_bytes(quantized_dim());
  }
  void transform(const void *vector, std::vector<float> &out) const {
    rotation_.rotate(vector, out);
  }
  void encode(const float *center, const float *vectors, size_t count,
              char *codes) const {
    if (!block_dim_) {
      rabitqlib::quant::quantize_qg_batch(vectors, center, count, encoded_dim(),
                                          codes, rabitqlib::METRIC_L2);
      if (cosine_) {
        rabitqlib::QGBatchDataMap<float> batch(codes, quantized_dim());
        ConvertSymQGCosineFactors(center, vectors, count, encoded_dim(),
                                  batch.f_add(), batch.f_rescale());
      }
      return;
    }
    if (count > kBatchSize) throw std::invalid_argument("block count");
    if (!count) return;
    const size_t dim = encoded_dim(), bytes = code_dim_ / 8;
    std::vector<float> a(dim);
    double delta2 = 0;
    for (size_t d = 0; d < dim; ++d) {
      a[d] = (1 - lambda_) * center[d] + lambda_ * mean_[d];
      const double delta = double(center[d]) - mean_[d];
      delta2 += delta * delta;
    }
    double correction = lambda_ * (1 - lambda_) * delta2;
    if (rank_) {
      // Project the neighbor-mean displacement into the shared orthonormal
      // basis. Reconstruct from stored coefficients so encoding and scoring
      // agree.
      std::vector<float> residual(dim), z(rank_), decoded(rank_);
      for (size_t d = 0; d < dim; ++d) {
        double mean = 0;
        for (size_t i = 0; i < count; ++i) mean += vectors[i * dim + d];
        residual[d] = float(mean / count - a[d]);
      }
      rank_project(residual.data(), z.data());
      float maximum = 0;
      for (float value : z) maximum = std::max(maximum, std::abs(value));
      const float scale = std::max(1.f, maximum / 32752.f);
      for (float &value : z) value /= scale;
      std::vector<uint16_t> packed(rank_);
      ailego::FloatHelper::ToFP16(z.data(), rank_, packed.data());
      ailego::FloatHelper::ToFP32(packed.data(), rank_, decoded.data());
      const float factors[4] = {scale, 0, 0, 0};
      std::memcpy(codes, factors, 16);
      std::memcpy(codes + 16, packed.data(), rank_ * 2);
      for (float &value : decoded) value *= scale;
      double norm_difference = 0;
      for (size_t d = 0; d < dim; ++d) {
        double delta = 0;
        for (size_t j = 0; j < rank_; ++j)
          delta += double(rank_basis_[j * dim + d]) * decoded[j];
        const float previous = a[d];
        a[d] = float(double(previous) + delta);
        norm_difference += double(a[d]) * a[d] - double(previous) * previous;
      }
      // ||q-a||^2 = mixed_distance - 2*q^T*U*z - K + norm_difference.
      // Fold the query-independent terms into the existing additive factor.
      correction -= norm_difference;
      codes += anchor_bytes();
    }
    std::vector<uint8_t> compact(count * bytes, 0);
    std::vector<double> residual(dim);
    rabitqlib::QGBatchDataMap<float> batch(codes, code_dim_);
    for (size_t i = 0; i < count; ++i) {
      const float *v = vectors + i * dim;
      double norm2 = 0, a_dot_r = 0;
      for (size_t d = 0; d < dim; ++d) {
        residual[d] = double(v[d]) - a[d];
        norm2 += residual[d] * residual[d];
        a_dot_r += a[d] * residual[d];
      }
      const double alpha = std::sqrt(norm2 / dim);
      double r_dot_b = 0, a_dot_b = 0;
      for (size_t group = 0; group < groups_; ++group) {
        const size_t offset = group * block_dim_;
        const size_t active = std::min(block_dim_, dim - offset);
        double best = -std::numeric_limits<double>::infinity();
        size_t label = 0;
        for (size_t j = 0; j < 16; ++j) {
          double dot = 0;
          for (size_t d = 0; d < active; ++d)
            dot += residual[offset + d] * words_[j * block_dim_ + d];
          const double score = 2 * alpha * dot - alpha * alpha * word_norm2_[j];
          if (score > best) {
            best = score;
            label = j;
          }
        }
        compact[i * bytes + group / 2] |=
            uint8_t(label << ((group & 1) ? 0 : 4));
        for (size_t d = 0; d < active; ++d) {
          const float word = words_[label * block_dim_ + d];
          r_dot_b += residual[offset + d] * word;
          a_dot_b += a[offset + d] * word;
        }
      }
      if (norm2 != 0 && !(r_dot_b > 0))
        throw std::runtime_error("nonpositive radial denominator");
      // Radial calibration makes r orthogonal to the reconstruction error.
      const double beta = norm2 == 0 ? 0 : norm2 / r_dot_b;
      batch.f_add()[i] =
          float(cosine_ ? beta * a_dot_b - a_dot_r
                        : norm2 + 2 * beta * a_dot_b - correction);
      batch.f_rescale()[i] = float((cosine_ ? -1 : -2) * beta);
    }
    rabitqlib::fastscan::pack_codes(code_dim_, compact.data(), count,
                                    batch.bin_code());
  }
  void prepare_query(const void *vector, Query &query) const {
    rotation_.rotate(vector, query.rotated, true);
    if (rank_) {
      query.rank_query.resize(rank_);
      rank_project(query.rotated.data(), query.rank_query.data());
    }
    if (!block_dim_) {
      query.lookup.reset(query.rotated.data(), quantized_dim());
      return;
    }
    query.lookup.reset_block(query.rotated.data(), encoded_dim(),
                             columns_.data(), block_dim_, code_dim_,
                             structured_);
    if (symqg_block_mean_distance(query.rotated.data(), mean_.data(),
                                  encoded_dim(), cosine_, &query.mean_distance))
      return;
    double value = 0;
    for (size_t d = 0; d < encoded_dim(); ++d) {
      if (cosine_)
        value += double(query.rotated[d]) * mean_[d];
      else {
        const double z = double(query.rotated[d]) - mean_[d];
        value += z * z;
      }
    }
    query.mean_distance = cosine_ ? float(1 - value) : float(value);
  }
  void scan(const char *codes, Query &query, float center_distance,
            float *distances, float uncertainty_bonus = 0) const {
    float navigation = block_dim_ ? (1 - lambda_) * center_distance +
                                        lambda_ * query.mean_distance
                                  : center_distance;
    if (rank_) {
      float dot = 0, scale;
      std::memcpy(&scale, codes, 4);
      if (!symqg_rank_dot(query.rank_query.data(), codes + 16, rank_, &dot)) {
        double exact = 0;
        for (size_t j = 0; j < rank_; ++j) {
          uint16_t value;
          std::memcpy(&value, codes + 16 + 2 * j, 2);
          exact +=
              double(query.rank_query[j]) * ailego::FloatHelper::ToFP32(value);
        }
        dot = float(exact);
      }
      // U^T*q was computed once; this correction costs only rank_ operations.
      navigation -= (cosine_ ? 1.f : 2.f) * scale * dot;
      codes += anchor_bytes();
    }
    query.lookup.set_g_add(navigation);
    query.lookup.set_priority_bonus(
        cosine_ && uncertainty_bonus > 0
            ? (block_dim_ ? bonus_scale_ : 1) * uncertainty_bonus *
                  std::sqrt(std::max(0.f, 2.f * center_distance))
            : 0);
    ScanSymQGBatch(codes, query.lookup, quantized_dim(), distances);
  }

 private:
  void rank_project(const float *input, float *out) const {
    if (symqg_rank_project(rank_basis_.data(), input, rank_, encoded_dim(),
                           out))
      return;
    for (size_t j = 0; j < rank_; ++j) {
      double value = 0;
      for (size_t d = 0; d < encoded_dim(); ++d)
        value += double(rank_basis_[j * encoded_dim() + d]) * input[d];
      out[j] = float(value);
    }
  }


  std::string model_data_;
  size_t rank_ = 0;
  std::vector<float> rank_basis_;
  SymQGRotation rotation_;
  bool cosine_;
  bool structured_ = false;
  size_t block_dim_ = 0, groups_ = 0, code_dim_ = 0;
  float lambda_ = 0, bonus_scale_ = 1;
  std::vector<float> mean_, words_, columns_;
  std::array<double, 16> word_norm2_{};
};
#endif
}  // namespace zvec::turbo
