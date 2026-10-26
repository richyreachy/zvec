#pragma once
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
namespace zvec::core {
// Offline research objective F(S)=sum_t max_{s in S} gain(s,t).
// Nonnegative gains make F monotone submodular; cardinality greedy applies.
inline std::vector<uint32_t> QuantizedGraphGreedyCover(
    const std::vector<float> &gain, size_t candidates, size_t targets,
    size_t budget) {
  std::vector<float> covered(targets, 0.f);
  std::vector<char> taken(candidates, 0);
  std::vector<uint32_t> selected;
  selected.reserve(std::min(budget, candidates));
  for (size_t step = 0; step < std::min(budget, candidates); ++step) {
    size_t best = candidates;
    double best_gain = -1;
    for (size_t s = 0; s < candidates; ++s)
      if (!taken[s]) {
        double improvement = 0;
        for (size_t t = 0; t < targets; ++t)
          improvement += std::max(0.f, gain[s * targets + t] - covered[t]);
        if (improvement > best_gain) {
          best = s;
          best_gain = improvement;
        }
      }
    if (best == candidates) break;
    taken[best] = 1;
    selected.push_back(uint32_t(best));
    for (size_t t = 0; t < targets; ++t)
      covered[t] = std::max(covered[t], gain[best * targets + t]);
  }
  return selected;
}
inline float QuantizedGraphRelativeRadius(float target2, float edge2,
                                          float reconstruction2, float true_dot,
                                          float estimate_dot, bool quantized) {
  if (!(target2 > 0 && edge2 > 0)) return 0;
  float radius = (2 * true_dot - edge2) / (2 * std::sqrt(edge2));
  if (quantized) {
    if (!(reconstruction2 > 0)) return 0;
    radius = std::min(
        radius, (2 * estimate_dot - edge2) / (2 * std::sqrt(reconstruction2)));
  }
  return std::max(0.f, radius / std::sqrt(target2));
}
__attribute__((target("avx512f,fma"))) inline std::array<float, 2>
QuantizedGraphCoverDots(const float *query, const float *r, const float *h,
                        size_t n) {
  __m512 a[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                 _mm512_setzero_ps()},
         b[4] = {a[0], a[1], a[2], a[3]};
  size_t i = 0;
  for (; i + 64 <= n; i += 64)
    for (size_t j = 0; j < 4; ++j) {
      __m512 q = _mm512_loadu_ps(query + i + 16 * j);
      a[j] = _mm512_fmadd_ps(q, _mm512_loadu_ps(r + i + 16 * j), a[j]);
      b[j] = _mm512_fmadd_ps(q, _mm512_loadu_ps(h + i + 16 * j), b[j]);
    }
  float x = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a[0], a[1]),
                                               _mm512_add_ps(a[2], a[3]))),
        y = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(b[0], b[1]),
                                               _mm512_add_ps(b[2], b[3])));
  for (; i < n; ++i) {
    x += query[i] * r[i];
    y += query[i] * h[i];
  }
  return {x, y};
}
}  // namespace zvec::core
