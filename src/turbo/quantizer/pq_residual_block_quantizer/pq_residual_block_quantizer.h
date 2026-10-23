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

#include <cassert>
#include <cmath>
#include <cstring>
#include <turbo/quantizer/pq_block_quantizer/pq_block_quantizer.h>

namespace zvec::turbo {

// Node-centered PQ4: encode (neighbor - center) with a shared codebook.
// The query is a single -dot(q, residual_centroid) LUT, reused for every node.
// L2 payload: [packed 32-lane PQ4 codes][32 FP32 additive corrections].
// Cosine payload: packed codes only, on caller-normalized coordinates.
class PQResidualBlockQuantizer {
 public:
  static constexpr size_t kBatchSize = PQBlockQuantizer::kBatchSize;
  using Query = PQBlockQuantizer::Query;

  explicit PQResidualBlockQuantizer(size_t dimension, bool cosine = false)
      : dimension_(dimension), cosine_(cosine), model_(dimension, true) {}
  size_t encoded_dim() const {
    return model_.encoded_dim();
  }
  size_t batch_bytes() const {
    return model_.batch_bytes() + (cosine_ ? 0 : kBatchSize * sizeof(float));
  }
  bool trained() const {
    return model_.trained();
  }
  const PqFastQuantizer *model() const {
    return model_.model();
  }
  void reset_training() {
    model_.reset_training();
  }
  int train(const core::IndexHolder::Pointer &holder, size_t threads) {
    // Always train an IP-search model: encoding still minimizes L2 error,
    // and residuals must never be cosine-normalized or globally centered.
    return model_.train(holder, threads);
  }
  void transform(const void *vector, std::vector<float> &out) const {
    model_.transform(vector, out);
  }
  int transform_residual(const void *center, const void *neighbor,
                         std::vector<float> &out) const {
    const auto *u = static_cast<const float *>(center);
    const auto *v = static_cast<const float *>(neighbor);
    out.assign(encoded_dim(), 0.0f);
    for (size_t d = 0; d < dimension_; ++d) {
      out[d] = v[d] - u[d];
      if (!std::isfinite(u[d]) || !std::isfinite(v[d]) ||
          !std::isfinite(out[d]))
        return core::IndexError_Mismatch;
    }
    return 0;
  }
  void encode(const float *center, const float *vectors, size_t count,
              char *codes) const {
    assert(count <= kBatchSize);
    const auto *quantizer = model();
    const size_t stride = quantizer->quantized_datapoint_vector_length();
    std::vector<char> plain(std::max(size_t{1}, count) * stride);
    std::vector<float> residual(encoded_dim()), reconstructed(encoded_dim());
    std::memset(codes, 0, batch_bytes());
    for (size_t i = 0; i < count; ++i) {
      for (size_t d = 0; d < encoded_dim(); ++d)
        residual[d] = vectors[i * encoded_dim() + d] - center[d];
      char *code = plain.data() + i * stride;
      quantizer->quantize_data(residual.data(), code);
      if (!cosine_) {
        const int ret =
            quantizer->reconstruct_plain_fp32(code, reconstructed.data());
        assert(ret == 0);
        (void)ret;
        // Double accumulation reduces cancellation in large translated data;
        // the stored correction and scan outputs remain FP32.
        double correction = 0;
        for (size_t d = 0; d < encoded_dim(); ++d) {
          const double r = reconstructed[d];
          correction += r * r + 2.0 * center[d] * r;
        }
        const float add = static_cast<float>(correction);
        std::memcpy(codes + model_.batch_bytes() + i * sizeof(float), &add,
                    sizeof(add));
      }
    }
    const int ret = quantizer->pack_codes(plain.data(), count, stride, codes);
    assert(ret == 0);
    (void)ret;
  }
  void prepare_query(const void *vector, Query &query) const {
    model_.prepare_query(vector, query);
  }
  void scan(const char *codes, Query &query, float center_distance,
            float *distances) const {
    model()->calc_distance_packed_block(codes, kBatchSize, query.data(),
                                        distances);
    for (size_t i = 0; i < kBatchSize; ++i) {
      if (cosine_) {
        distances[i] += center_distance;
      } else {
        float add;
        std::memcpy(&add, codes + model_.batch_bytes() + i * sizeof(float),
                    sizeof(add));
        distances[i] = static_cast<float>(double(center_distance) + add +
                                          2.0 * distances[i]);
      }
    }
  }

 private:
  size_t dimension_;
  bool cosine_;
  PQBlockQuantizer model_;
};
}  // namespace zvec::turbo
