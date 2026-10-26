#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <random>
#include <set>
#include <algorithm/quantized_graph/quantized_graph_cover.h>
using namespace zvec::core;
int main() {
  std::mt19937 rng(49026);
  std::uniform_real_distribution<float> uniform(0, 1);
  std::normal_distribution<float> normal(0, 1);
  size_t trials = 0, balls = 0;
  double worst_ratio = 1;
  for (int trial = 0; trial < 1000; ++trial) {
    const size_t n = 8, t = 11, k = 3;
    std::vector<float> gain(n * t);
    for (float &g : gain) g = uniform(rng);
    auto objective = [&](const std::vector<uint32_t> &s) {
      double f = 0;
      for (size_t j = 0; j < t; ++j) {
        float v = 0;
        for (auto i : s) v = std::max(v, gain[i * t + j]);
        f += v;
      }
      return f;
    };
    auto selected = QuantizedGraphGreedyCover(gain, n, t, k);
    assert(selected.size() == k);
    assert(std::set<uint32_t>(selected.begin(), selected.end()).size() == k);
    double best = 0;
    for (uint32_t a = 0; a < n; ++a)
      for (uint32_t b = a + 1; b < n; ++b)
        for (uint32_t c = b + 1; c < n; ++c)
          best = std::max(best, objective({a, b, c}));
    double actual = objective(selected);
    worst_ratio = std::min(worst_ratio, actual / best);
    assert(actual + 1e-5 >= best * (1 - std::pow(1 - 1. / k, k)));
    ++trials;
  }
  for (size_t d : {16, 64, 768})
    for (int trial = 0; trial < 1000; ++trial) {
      std::vector<float> r(d), h(d), target(d), eta(d);
      double rr = 0, l1 = 0, tt = 0;
      for (size_t j = 0; j < d; ++j) {
        r[j] = normal(rng);
        target[j] = r[j] + .7f * normal(rng);
        rr += double(r[j]) * r[j];
        tt += double(target[j]) * target[j];
        l1 += std::abs(r[j]);
      }
      double amplitude = rr / l1, hh = 0;
      for (size_t j = 0; j < d; ++j) {
        h[j] = r[j] > 0 ? amplitude : -amplitude;
        hh += double(h[j]) * h[j];
      }
      auto dot = QuantizedGraphCoverDots(target.data(), r.data(), h.data(), d);
      double direct[2] = {0, 0};
      for (size_t j = 0; j < d; ++j) {
        direct[0] += double(target[j]) * r[j];
        direct[1] += double(target[j]) * h[j];
      }
      for (int k = 0; k < 2; ++k)
        assert(std::abs(dot[k] - direct[k]) < 1e-5 * (1 + std::abs(direct[k])));
      double rho =
          QuantizedGraphRelativeRadius(tt, rr, hh, dot[0], dot[1], true) *
          std::sqrt(tt);
      if (rho <= 0) continue;
      for (int sample = 0; sample < 4; ++sample) {
        double norm = 0;
        for (float &v : eta) {
          v = normal(rng);
          norm += double(v) * v;
        }
        double mt = -rr, mh = -rr;
        for (size_t j = 0; j < d; ++j) {
          double q = target[j] + eta[j] / std::sqrt(norm) * rho * .999;
          mt += 2 * q * r[j];
          mh += 2 * q * h[j];
        }
        assert(mt >= -1e-4 * (1 + rr));
        assert(mh >= -1e-4 * (1 + rr));
        ++balls;
      }
    }
  std::cout << "cover greedy-vs-exhaustive trials=" << trials
            << " min_ratio=" << worst_ratio << " local-ball checks=" << balls
            << " passed\n";
}
