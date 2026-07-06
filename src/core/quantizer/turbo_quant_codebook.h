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
// TurboQuant codebook: optimal scalar quantizer for the Beta distribution
// that arises from random rotation of unit-norm vectors (Lemma 1 of the paper).
// Centroids are found via the Lloyd-Max (continuous 1-D k-means) algorithm.

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace zvec {
namespace core {

/*! TurboQuant codebook for a given (dimension, bit-width) pair.
 *
 * Each coordinate of a randomly-rotated unit-norm vector follows the
 * Beta distribution f_X(x) = Gamma(d/2)/(sqrt(pi)*Gamma((d-1)/2)) *
 * (1-x^2)^((d-3)/2) on [-1, 1], converging to N(0, 1/d) in high
 * dimensions.  The optimal b-bit scalar quantizer for this distribution
 * is obtained by solving a continuous 1-D k-means problem (Eq. 4 in
 * the paper) via the Lloyd-Max algorithm.
 */
class TurboQuantCodebook {
 public:
  //! Construct and train a codebook for the given (dim, bits) pair.
  explicit TurboQuantCodebook(size_t dimension, int bits)
      : dimension_(dimension), bits_(bits) {
    num_centroids_ = 1u << bits;
    train();
  }

  //! Quantize a single rotated coordinate to its nearest centroid index.
  uint32_t quantize(float x) const {
    if (num_centroids_ <= 1) return 0;
    if (x <= boundaries_[0]) return 0;
    if (x >= boundaries_[num_centroids_ - 2]) return num_centroids_ - 1;
    auto it = std::lower_bound(boundaries_.begin(), boundaries_.end(), x);
    return static_cast<uint32_t>(it - boundaries_.begin());
  }

  //! Dequantize: return the centroid value for a given index.
  float dequantize(uint32_t idx) const {
    return centroids_[idx];
  }

  //! Return the centroids (sorted ascending).
  const std::vector<float> &centroids() const {
    return centroids_;
  }

  //! Return the boundaries (midpoints between consecutive centroids).
  const std::vector<float> &boundaries() const {
    return boundaries_;
  }

  //! Number of centroids (2^bits).
  size_t num_centroids() const {
    return num_centroids_;
  }

  //! Bit-width.
  int bits() const {
    return bits_;
  }

  //! Dimension.
  size_t dimension() const {
    return dimension_;
  }

  //! Get or create a cached codebook for (dim, bits).
  static std::shared_ptr<TurboQuantCodebook> get(size_t dimension, int bits) {
    uint64_t key =
        (static_cast<uint64_t>(dimension) << 8) | static_cast<uint64_t>(bits);
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto &cache = codebook_cache();
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    }
    auto cb = std::make_shared<TurboQuantCodebook>(dimension, bits);
    cache[key] = cb;
    return cb;
  }

 private:
  //! Beta distribution PDF for a coordinate of a random point on the
  //! unit hypersphere in R^d (Lemma 1).
  double beta_pdf(double x) const {
    constexpr double kPi = 3.14159265358979323846;
    double abs_sq = 1.0 - x * x;
    if (abs_sq <= 0.0) return 0.0;
    double d = static_cast<double>(dimension_);
    double log_coeff = std::lgamma(d / 2.0) - 0.5 * std::log(kPi) -
                       std::lgamma((d - 1.0) / 2.0);
    double exponent = (d - 3.0) / 2.0;
    return std::exp(log_coeff + exponent * std::log(abs_sq));
  }

  //! Numerical integration of f(x) over [a, b] using composite Simpson's rule.
  double integrate(std::function<double(double)> f, double a, double b,
                   int n = 2000) const {
    if (b <= a) return 0.0;
    if (n % 2 == 1) ++n;
    double h = (b - a) / n;
    double s = f(a) + f(b);
    for (int i = 1; i < n; i += 2) {
      s += 4.0 * f(a + i * h);
    }
    for (int i = 2; i < n; i += 2) {
      s += 2.0 * f(a + i * h);
    }
    return s * h / 3.0;
  }

  //! Train the codebook using the Lloyd-Max algorithm.
  void train() {
    centroids_.resize(num_centroids_);
    boundaries_.resize(num_centroids_ - 1);

    // For high d, beta ≈ N(0, 1/d), so the distribution is concentrated
    // in [-sigma*4, sigma*4] where sigma = 1/sqrt(d).
    double sigma = 1.0 / std::sqrt(static_cast<double>(dimension_));
    double range_min = -1.0;
    double range_max = 1.0;
    // For very high dimensions the distribution is tightly concentrated,
    // so restrict the integration/initialization range.
    double effective_range = std::min(1.0, 6.0 * sigma);
    range_min = -effective_range;
    range_max = effective_range;

    // Initialize centroids as quantiles of the distribution.
    // Use inverse-CDF sampling via numerical CDF.
    int grid_n = 8000;
    double grid_h = (range_max - range_min) / grid_n;
    std::vector<double> x_vals(grid_n + 1), pdf_vals(grid_n + 1),
        cdf_vals(grid_n + 1);
    double cdf_total = 0.0;
    for (int i = 0; i <= grid_n; ++i) {
      x_vals[i] = range_min + i * grid_h;
      pdf_vals[i] = beta_pdf(x_vals[i]);
      if (i > 0) {
        cdf_total += 0.5 * (pdf_vals[i] + pdf_vals[i - 1]) * grid_h;
      }
      cdf_vals[i] = cdf_total;
    }
    // Normalize CDF
    if (cdf_total > 0.0) {
      for (auto &v : cdf_vals) v /= cdf_total;
    }

    // Initialize centroids at equal-CDF-mass quantiles
    for (size_t i = 0; i < num_centroids_; ++i) {
      double target_cdf = (i + 0.5) / num_centroids_;
      // Find x where CDF ≈ target_cdf
      auto it = std::lower_bound(cdf_vals.begin(), cdf_vals.end(), target_cdf);
      size_t idx = std::distance(cdf_vals.begin(), it);
      idx = std::min(idx, static_cast<size_t>(grid_n));
      centroids_[i] = static_cast<float>(x_vals[idx]);
    }
    std::sort(centroids_.begin(), centroids_.end());

    // Lloyd-Max iterations
    constexpr int kMaxIters = 200;
    constexpr double kTolerance = 1e-9;
    for (int iter = 0; iter < kMaxIters; ++iter) {
      // Update boundaries as midpoints
      for (size_t i = 0; i < num_centroids_ - 1; ++i) {
        boundaries_[i] = (centroids_[i] + centroids_[i + 1]) * 0.5f;
      }

      // Update centroids via continuous k-means:
      // c_i = integral(x * f(x), [b_{i-1}, b_i]) /
      //       integral(f(x), [b_{i-1}, b_i])
      double max_change = 0.0;
      for (size_t i = 0; i < num_centroids_; ++i) {
        double lo =
            (i == 0) ? range_min : static_cast<double>(boundaries_[i - 1]);
        double hi = (i == num_centroids_ - 1)
                        ? range_max
                        : static_cast<double>(boundaries_[i]);
        auto f_pdf = [this](double x) { return beta_pdf(x); };
        auto f_xpdf = [this](double x) { return x * beta_pdf(x); };

        double mass = integrate(f_pdf, lo, hi);
        double moment = integrate(f_xpdf, lo, hi);
        if (mass > 1e-15) {
          double new_c = moment / mass;
          double change = std::abs(new_c - centroids_[i]);
          max_change = std::max(max_change, change);
          centroids_[i] = static_cast<float>(new_c);
        }
      }
      if (max_change < kTolerance) break;
    }

    // Final boundaries
    for (size_t i = 0; i < num_centroids_ - 1; ++i) {
      boundaries_[i] = (centroids_[i] + centroids_[i + 1]) * 0.5f;
    }
  }

  // --- Cache infrastructure ---
  using CacheMap =
      std::unordered_map<uint64_t, std::shared_ptr<TurboQuantCodebook>>;
  static CacheMap &codebook_cache() {
    static CacheMap inst;
    return inst;
  }
  static std::mutex &cache_mutex() {
    static std::mutex m;
    return m;
  }

  size_t dimension_{0};
  int bits_{0};
  size_t num_centroids_{0};
  std::vector<float> centroids_{};
  std::vector<float> boundaries_{};
};

/*! Bit-packing utilities for TurboQuant indices.
 *
 * Packs b-bit indices into a byte array.  Indices are stored in
 * little-endian bit order: the first index occupies the least
 * significant bits of the first byte.
 */
class TurboQuantPacking {
 public:
  //! Return the number of bytes needed to pack `count` indices of `bits`
  //! bits each.
  static size_t packed_bytes(size_t count, int bits) {
    return (count * bits + 7) / 8;
  }

  //! Pack `count` b-bit indices from `indices` into `out`.
  static void pack(const uint32_t *indices, size_t count, int bits,
                   uint8_t *out) {
    std::fill(out, out + packed_bytes(count, bits), 0);
    for (size_t i = 0; i < count; ++i) {
      uint32_t val = indices[i] & ((1u << bits) - 1);
      size_t bit_pos = i * bits;
      size_t byte_pos = bit_pos / 8;
      size_t bit_off = bit_pos % 8;
      out[byte_pos] |= (val << bit_off) & 0xFF;
      if (bit_off + bits > 8) {
        out[byte_pos + 1] |= (val >> (8 - bit_off)) & 0xFF;
      }
      if (bit_off + bits > 16) {
        out[byte_pos + 2] |= (val >> (16 - bit_off)) & 0xFF;
      }
    }
  }

  //! Unpack `count` b-bit indices from `data` into `out`.
  static void unpack(const uint8_t *data, size_t count, int bits,
                     uint32_t *out) {
    uint32_t mask = (1u << bits) - 1u;
    for (size_t i = 0; i < count; ++i) {
      size_t bit_pos = i * bits;
      size_t byte_pos = bit_pos / 8;
      size_t bit_off = bit_pos % 8;
      uint32_t val = data[byte_pos] >> bit_off;
      if (bit_off + bits > 8) {
        val |= static_cast<uint32_t>(data[byte_pos + 1]) << (8 - bit_off);
      }
      if (bit_off + bits > 16) {
        val |= static_cast<uint32_t>(data[byte_pos + 2]) << (16 - bit_off);
      }
      out[i] = val & mask;
    }
  }
};

}  // namespace core
}  // namespace zvec
