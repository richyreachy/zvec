#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace zvec::turbo {
// Research-only prebuild calibration: keep -center.dot(residual) unchanged,
// shrink only the one-bit directional reconstruction. No extra code bytes or
// query arithmetic. Coordinates are already in the codec rotation frame.
struct DirectionalCalibration {
  static double mse_beta(double squared_norm, double l1_norm, size_t dim) {
    if (!(squared_norm > 0) || !dim) return 1.;
    return std::clamp(l1_norm * l1_norm / (double(dim) * squared_norm), .5, 1.);
  }
  template <class Add, class Scale>
  static void apply(const float *center, const float *vectors, size_t count,
                    size_t dim, double beta, bool per_edge_mse, Add add,
                    Scale scale) {
    if (!per_edge_mse && beta == 1.) return;
    for (size_t i = 0; i < count; ++i) {
      double cr = 0, rr = 0, l1 = 0;
      for (size_t j = 0; j < dim; ++j) {
        const double r = double(vectors[i * dim + j]) - center[j];
        cr += double(center[j]) * r;
        if (per_edge_mse) {
          rr += r * r;
          l1 += std::abs(r);
        }
      }
      const double b = per_edge_mse ? mse_beta(rr, l1, dim) : beta;
      if (b == 1.) continue;
      add[i] = float(b * double(add[i]) - (1. - b) * cr);
      scale[i] = float(b * double(scale[i]));
    }
  }
};
}  // namespace zvec::turbo
