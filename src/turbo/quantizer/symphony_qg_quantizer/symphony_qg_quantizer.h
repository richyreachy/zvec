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

#if defined(QG_LOWRANK)
#include <immintrin.h>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#if defined(QG_PROJECT_DIM) || defined(QG_RESIDUAL_TWO_STAGE) || \
    defined(QG_GEOM_BIAS)
#error Low-rank correction requires the native full-dimensional one-bit codec
#endif
#endif

#if defined(QG_RADIAL_CALIBRATION)
#include <immintrin.h>
#if defined(QG_LOWRANK) || defined(QG_PROJECT_DIM) || \
    defined(QG_RESIDUAL_TWO_STAGE) || defined(QG_GEOM_BIAS)
#error Radial calibration requires the native full-dimensional one-bit codec
#endif
#include "radial_calibration.h"
#endif

#if defined(QG_DIRECTION_BETA) || defined(QG_DIRECTION_MSE)
#if defined(QG_RADIAL_CALIBRATION) || defined(QG_LOWRANK) ||     \
    defined(QG_PROJECT_DIM) || defined(QG_RESIDUAL_TWO_STAGE) || \
    defined(QG_GEOM_BIAS)
#error Directional calibration requires the native full-dimensional one-bit codec
#endif
#if defined(QG_DIRECTION_BETA) && defined(QG_DIRECTION_MSE)
#error Choose one directional calibration objective
#endif
#if defined(QG_DIRECTION_BETA)
static_assert(QG_DIRECTION_BETA >= .5 && QG_DIRECTION_BETA <= 1.,
              "Directional beta must be in [0.5,1]");
#endif
#include "directional_calibration.h"
#endif

namespace zvec::turbo {
#if RABITQ_SUPPORTED
// Node-centered RaBitQ blocks with the SymphonyQG rotation and FastScan layout.
class SymphonyQGBlockQuantizer {
 public:
  static constexpr size_t kBatchSize = rabitqlib::fastscan::kBatchSize;
  struct Query {
    std::vector<float> rotated;
    SymQuery lookup;
#if defined(QG_LOWRANK)
    std::array<float, QG_LOWRANK> projection{};
#endif
  };

  explicit SymphonyQGBlockQuantizer(size_t dimension, bool cosine = false)
      : rotation_(dimension), cosine_(cosine) {
#if defined(QG_LOWRANK)
    const char *path = std::getenv("QG_LOWRANK_BASIS");
    if (!path) throw std::runtime_error("QG_LOWRANK_BASIS is required");
    std::ifstream input(path, std::ios::binary);
    std::vector<float> raw(dimension), rotated;
    basis_.resize(QG_LOWRANK * encoded_dim());
    for (size_t j = 0; j < QG_LOWRANK; ++j) {
      if (!input.read(reinterpret_cast<char *>(raw.data()),
                      dimension * sizeof(float)))
        throw std::runtime_error("Truncated low-rank basis");
      transform(raw.data(), rotated);
      std::copy(rotated.begin(), rotated.end(),
                basis_.begin() + j * encoded_dim());
    }
#endif
  }

#if defined(QG_FUSION)
  bool fusion_supported() const {
    return cosine_ && encoded_dim() == 768 && quantized_dim() == 768;
  }
#endif
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
    return rabitqlib::QGBatchDataMap<float>::data_bytes(quantized_dim())
#if defined(QG_RESIDUAL_TWO_STAGE)
               * 2
#endif
#if defined(QG_LOWRANK)
           + QG_LOWRANK * kBatchSize * sizeof(float)
#endif
        ;
  }
  void transform(const void *vector, std::vector<float> &out) const {
    rotation_.rotate(vector, out);
  }
  void encode(const float *center, const float *vectors, size_t count,
              char *codes) const {
#if defined(QG_RESIDUAL_TWO_STAGE)
#if defined(QG_PROJECT_DIM)
#error Two-stage prototype requires full transformed dimensions
#endif
    const size_t d = encoded_dim(),
                 bytes = rabitqlib::QGBatchDataMap<float>::data_bytes(d);
    std::vector<float> residual(count * d), error(count * d), zero(d, 0.f),
        a1(count), a2(count), scale(count), norm2(count), cent_dot(count);
    for (size_t i = 0; i < count; ++i) {
      double l1 = 0;
      for (size_t j = 0; j < d; ++j) {
        float r = vectors[i * d + j] - center[j];
        residual[i * d + j] = r;
        l1 += std::abs(double(r));
      }
      a1[i] = float(l1 / d);
      l1 = 0;
      for (size_t j = 0; j < d; ++j) {
        float r = residual[i * d + j];
        float e = r - (r > 0 ? a1[i] : -a1[i]);
        error[i * d + j] = e;
        l1 += std::abs(double(e));
      }
      a2[i] = float(l1 / d);
      double rr = 0, rh = 0;
      for (size_t j = 0; j < d; ++j) {
        double r = residual[i * d + j],
               h = (r > 0 ? a1[i] : -a1[i]) +
                   (error[i * d + j] > 0 ? a2[i] : -a2[i]);
        rr += r * r;
        rh += r * h;
      }
      scale[i] = rh > 0 ? float(rr / rh) : 0;
      norm2[i] = float(rr);
      double correction = 0;
      for (size_t j = 0; j < d; ++j) {
        double r = residual[i * d + j],
               h = scale[i] * double((r > 0 ? a1[i] : -a1[i]) +
                                     (error[i * d + j] > 0 ? a2[i] : -a2[i]));
        correction += center[j] * (cosine_ ? h - r : 2 * h);
      }
      cent_dot[i] = float(correction);
    }
    rabitqlib::quant::quantize_qg_batch(residual.data(), zero.data(), count, d,
                                        codes, rabitqlib::METRIC_L2);
    rabitqlib::quant::quantize_qg_batch(error.data(), zero.data(), count, d,
                                        codes + bytes, rabitqlib::METRIC_L2);
    rabitqlib::QGBatchDataMap<float> first(codes, d), second(codes + bytes, d);
    for (size_t i = 0; i < count; ++i) {
      first.f_add()[i] = cent_dot[i] + (cosine_ ? 0.f : norm2[i]);
      second.f_add()[i] = 0;
      first.f_rescale()[i] = (cosine_ ? -2.f : -4.f) * scale[i] * a1[i];
      second.f_rescale()[i] = (cosine_ ? -2.f : -4.f) * scale[i] * a2[i];
    }
    return;
#endif
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
#if defined(QG_DIRECTION_BETA) || defined(QG_DIRECTION_MSE)
    if (cosine_) {
      rabitqlib::QGBatchDataMap<float> batch(codes, encoded_dim());
#if defined(QG_DIRECTION_BETA)
      constexpr double beta = QG_DIRECTION_BETA;
      constexpr bool per_edge_mse = false;
#else
      constexpr double beta = 1.;
      constexpr bool per_edge_mse = true;
#endif
      DirectionalCalibration::apply(center, vectors, count, encoded_dim(), beta,
                                    per_edge_mse, batch.f_add(),
                                    batch.f_rescale());
    }
#endif
#if defined(QG_RADIAL_CALIBRATION)
    if (cosine_) {
      rabitqlib::QGBatchDataMap<float> batch(codes, encoded_dim());
      double cc = 0;
      for (size_t j = 0; j < encoded_dim(); ++j)
        cc += double(center[j]) * center[j];
#if defined(QG_RADIAL_STATS)
      std::array<uint64_t, 4> counts{};
#endif
      for (size_t i = 0; i < count; ++i) {
        const float amplitude = -.5f * batch.f_rescale()[i];
        double rr = 0, cr = 0, ch = 0, rh = 0;
        for (size_t j = 0; j < encoded_dim(); ++j) {
          const double r = double(vectors[i * encoded_dim() + j]) - center[j];
          const double h = r > 0 ? amplitude : -amplitude;
          rr += r * r;
          cr += double(center[j]) * r;
          ch += double(center[j]) * h;
          rh += r * h;
        }
        const auto factors = RadialCalibration::fit(rr, cc, cr, ch, rh);
        batch.f_add()[i] = float(-factors.b);
        batch.f_rescale()[i] = float(double(batch.f_rescale()[i]) * factors.a);
#if defined(QG_RADIAL_STATS)
        ++counts[factors.kind];
#endif
      }
#if defined(QG_RADIAL_STATS)
      RadialCalibration::record(counts);
#endif
    }
#endif
#if defined(QG_LOWRANK)
    // e=h-r, h=(||r||^2/||r||_1) sign(r). Remove the projection of
    // (center-query)^T e; remaining ideal error lies in the orthogonal
    // complement.
    rabitqlib::QGBatchDataMap<float> native(codes, encoded_dim());
    float *correction = reinterpret_cast<float *>(
        codes + rabitqlib::QGBatchDataMap<float>::data_bytes(encoded_dim()));
    std::fill_n(correction, QG_LOWRANK * kBatchSize, 0.f);
    std::array<float, QG_LOWRANK> center_projection;
    for (size_t j = 0; j < QG_LOWRANK; ++j)
      center_projection[j] =
          dot(center, basis_.data() + j * encoded_dim(), encoded_dim());
    std::vector<float> error(encoded_dim());
    for (size_t i = 0; i < count; ++i) {
      const float amplitude = native.f_rescale()[i] * (cosine_ ? -.5f : -.25f);
      for (size_t d = 0; d < encoded_dim(); ++d) {
        const float r = vectors[i * encoded_dim() + d] - center[d];
        error[d] = (r > 0 ? amplitude : -amplitude) - r;
      }
      for (size_t j = 0; j < QG_LOWRANK; ++j) {
        const float value = dot(error.data(), basis_.data() + j * encoded_dim(),
                                encoded_dim()) *
                            (cosine_ ? 1.f : 2.f);
        correction[j * kBatchSize + i] = value;
        native.f_add()[i] -= center_projection[j] * value;
      }
    }
#endif
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
#if defined(QG_LOWRANK)
    for (size_t j = 0; j < QG_LOWRANK; ++j)
      query.projection[j] =
          dot(query.rotated.data(), basis_.data() + j * encoded_dim(),
              encoded_dim());
#endif
  }
  void scan(const char *codes, Query &query, float center_distance,
            float *distances) const {
    query.lookup.set_g_add(center_distance);
    ScanSymphonyQGBatch(codes, query.lookup, quantized_dim(), distances);
#if defined(QG_RADIAL_CALIBRATION)
    if (cosine_) apply_radial(codes, center_distance, distances);
#endif
#if defined(QG_LOWRANK)
    apply_correction(codes, query, distances);
#endif
#if defined(QG_RESIDUAL_TWO_STAGE)
    std::array<float, kBatchSize> correction;
    query.lookup.set_g_add(0);
    ScanSymphonyQGBatch(
        codes + rabitqlib::QGBatchDataMap<float>::data_bytes(quantized_dim()),
        query.lookup, quantized_dim(), correction.data());
    for (size_t i = 0; i < kBatchSize; ++i) distances[i] += correction[i];
#endif
  }

 private:
#if defined(QG_RADIAL_CALIBRATION)
  __attribute__((target("avx512f,fma"))) void apply_radial(
      const char *codes, float center_distance, float *distances) const {
    rabitqlib::ConstQGBatchDataMap<float> batch(codes, encoded_dim());
    const __m512 center = _mm512_set1_ps(center_distance);
    for (size_t i = 0; i < kBatchSize; i += 16)
      _mm512_storeu_ps(
          distances + i,
          _mm512_fnmadd_ps(_mm512_loadu_ps(batch.f_add() + i), center,
                           _mm512_loadu_ps(distances + i)));
  }
#endif
#if defined(QG_LOWRANK)
  __attribute__((target("avx512f,fma"))) static float dot(const float *a,
                                                          const float *b,
                                                          size_t n) {
    __m512 sum[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(),
                     _mm512_setzero_ps(), _mm512_setzero_ps()};
    size_t i = 0;
    for (; i + 64 <= n; i += 64)
      for (size_t j = 0; j < 4; ++j)
        sum[j] = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16 * j),
                                 _mm512_loadu_ps(b + i + 16 * j), sum[j]);
    float value = _mm512_reduce_add_ps(_mm512_add_ps(
        _mm512_add_ps(sum[0], sum[1]), _mm512_add_ps(sum[2], sum[3])));
    for (; i < n; ++i) value += a[i] * b[i];
    return value;
  }
  __attribute__((target("avx512f,fma"))) void apply_correction(
      const char *codes, const Query &query, float *distances) const {
    const float *correction = reinterpret_cast<const float *>(
        codes + rabitqlib::QGBatchDataMap<float>::data_bytes(encoded_dim()));
    for (size_t i = 0; i < kBatchSize; i += 16) {
      __m512 value = _mm512_loadu_ps(distances + i);
      for (size_t j = 0; j < QG_LOWRANK; ++j)
        value = _mm512_fmadd_ps(
            _mm512_set1_ps(query.projection[j]),
            _mm512_loadu_ps(correction + j * kBatchSize + i), value);
      _mm512_storeu_ps(distances + i, value);
    }
  }
  std::vector<float> basis_;
#endif
  SymphonyQGRotation rotation_;
  bool cosine_;
};
#endif
}  // namespace zvec::turbo
