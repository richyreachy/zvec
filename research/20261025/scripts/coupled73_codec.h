#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "turbo/distance/common/symphony_qg_distance.h"
namespace coupled73 {
inline void encode(const float *c, const float *v, const int8_t *signs,
                   size_t count, size_t d, char *codes) {
  if (count > 32 || d == 0 || d % 64)
    throw std::invalid_argument("coupled73 shape");
  const size_t bytes = rabitqlib::QGBatchDataMap<float>::data_bytes(d);
  std::fill_n(codes, bytes, 0);
  rabitqlib::QGBatchDataMap<float> batch(codes, d);
  std::vector<uint8_t> compact(count * d / 8, 0);
  double cn = 0;
  for (size_t k = 0; k < d; ++k) {
    if (!std::isfinite(c[k])) throw std::invalid_argument("nonfinite center");
    cn += double(c[k]) * c[k];
  }
  for (size_t i = 0; i < count; ++i) {
    double rr = 0, u = 0, cs = 0, vn = 0;
    for (size_t k = 0; k < d; ++k) {
      const int s = signs[i * d + k];
      if (s != 1 && s != -1) throw std::invalid_argument("invalid sign");
      if (!std::isfinite(v[i * d + k]))
        throw std::invalid_argument("nonfinite vector");
      double r = double(v[i * d + k]) - c[k];
      rr += r * r;
      u += r * s;
      cs += double(c[k]) * s;
      vn += double(v[i * d + k]) * v[i * d + k];
      if (s > 0) compact[i * d / 8 + k / 8] |= uint8_t(1u << (7 - k % 8));
    }
    if (rr > 0 && !(u > 0))
      throw std::invalid_argument("nonpositive denominator");
    double alpha = rr ? rr / u : 0;
    double add = .5 * rr + alpha * cs + .5 * (cn - vn);
    batch.f_add()[i] = float(add);
    batch.f_rescale()[i] = float(-2 * alpha);
    if (!std::isfinite(batch.f_add()[i]) ||
        !std::isfinite(batch.f_rescale()[i]))
      throw std::invalid_argument("nonfinite factor");
  }
  rabitqlib::fastscan::pack_codes(d, compact.data(), count, batch.bin_code());
}
}  // namespace coupled73
