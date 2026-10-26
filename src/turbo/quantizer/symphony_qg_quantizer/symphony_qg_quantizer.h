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

#include <turbo/distance/common/symphony_qg_distance.h>
#include <turbo/preprocessor/symphony_qg_rotator/symphony_qg_rotator.h>

namespace zvec::turbo {
#if RABITQ_SUPPORTED
// Node-centered RaBitQ blocks with the SymphonyQG rotation and FastScan layout.
class SymphonyQGBlockQuantizer {
 public:
  static constexpr size_t kBatchSize = rabitqlib::fastscan::kBatchSize;
  struct Query {
    std::vector<float> rotated;
    SymQuery lookup;
  };

  explicit SymphonyQGBlockQuantizer(size_t dimension, bool cosine = false)
      : rotation_(dimension), cosine_(cosine) {}

  size_t encoded_dim() const {
    return rotation_.padded_dim();
  }
  size_t quantized_dim() const {
#if defined(QG_PROJECT_DIM)
    return std::min(encoded_dim(), size_t{QG_PROJECT_DIM});
#else
    return encoded_dim();
#endif
  }
  size_t batch_bytes() const {
    return rabitqlib::QGBatchDataMap<float>::data_bytes(quantized_dim());
  }
  void transform(const void *vector, std::vector<float> &out) const {
    rotation_.rotate(vector, out);
  }
  void encode(const float *center, const float *vectors, size_t count,
              char *codes) const {
#if defined(QG_PROJECT_DIM)
    const size_t qdim = quantized_dim();
    std::vector<float> compact(count * qdim);
    for (size_t i = 0; i < count; ++i)
      std::copy_n(vectors + i * encoded_dim(), qdim, compact.data() + i * qdim);
    rabitqlib::quant::quantize_qg_batch(compact.data(), center, count, qdim,
                                        codes, rabitqlib::METRIC_L2);
    rabitqlib::QGBatchDataMap<float> projected(codes, qdim);
    for (size_t i = 0; i < count; ++i) {
      double full = 0, partial = 0;
      for (size_t d = 0; d < encoded_dim(); ++d) {
        const double x = double(vectors[i * encoded_dim() + d]) - center[d];
        full += x * x;
        if (d < qdim) partial += x * x;
      }
      if (partial > 0) {
        const float ratio = float(full / partial);
        projected.f_add()[i] *= ratio;
        projected.f_rescale()[i] *= ratio;
      } else {
        // A zero projected residual contains no directional information.
        projected.f_add()[i] = float(full);
        projected.f_rescale()[i] = 0;
      }
    }
#else
    rabitqlib::quant::quantize_qg_batch(vectors, center, count, encoded_dim(),
                                        codes, rabitqlib::METRIC_L2);
#endif
    if (cosine_) {
      rabitqlib::QGBatchDataMap<float> batch(codes, quantized_dim());
      ConvertSymphonyQGCosineFactors(center, vectors, count, encoded_dim(),
                                     batch.f_add(), batch.f_rescale());
    }
#if defined(QG_GEOM_BIAS)
    // Research score family: native calibrated estimate + gamma*||e||^2/2
    // for cosine (gamma*||e||^2 for L2). gamma=1 is the squared distance
    // to the reconstructed residual endpoint before query-LUT rounding.
    rabitqlib::QGBatchDataMap<float> adjusted(codes, encoded_dim());
    for (size_t i = 0; i < count; ++i) {
      double norm2 = 0, half_l1 = 0;
      for (size_t d = 0; d < encoded_dim(); ++d) {
        const double r = double(vectors[i * encoded_dim() + d]) - center[d];
        norm2 += r * r;
        half_l1 += 0.5 * std::abs(r);
      }
      const double error2 =
          half_l1 > 0 ? std::max(0.0, norm2 * norm2 * (0.25 * encoded_dim()) /
                                              (half_l1 * half_l1) -
                                          norm2)
                      : 0.0;
      adjusted.f_add()[i] +=
          float(QG_GEOM_BIAS * (cosine_ ? 0.5 : 1.0) * error2);
    }
#endif
  }
  void prepare_query(const void *vector, Query &query) const {
    transform(vector, query.rotated);
    query.lookup.reset(query.rotated.data(), quantized_dim());
  }
  void scan(const char *codes, Query &query, float center_distance,
            float *distances) const {
    query.lookup.set_g_add(center_distance);
    ScanSymphonyQGBatch(codes, query.lookup, quantized_dim(), distances);
  }

 private:
  SymphonyQGRotation rotation_;
  bool cosine_;
};
#endif
}  // namespace zvec::turbo
