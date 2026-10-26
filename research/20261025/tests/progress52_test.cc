#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <random>
#include <algorithm/quantized_graph/quantized_graph_cover.h>
using namespace zvec::core;
int main() {
  const float epsilon = .001f;
  const float tiny_radius = QuantizedGraphRelativeRadius(
      1, epsilon * epsilon, epsilon * epsilon, epsilon, epsilon, false);
  const float direct_radius =
      QuantizedGraphRelativeRadius(1, 1, 1, 1, 1, false);
  const float tiny_progress = QuantizedGraphRelativeProgress(
      1, epsilon * epsilon, epsilon, epsilon, false);
  const float direct_progress =
      QuantizedGraphRelativeProgress(1, 1, 1, 1, false);
  assert(tiny_radius > direct_radius && tiny_progress < .003 &&
         direct_progress == 1);
  assert(QuantizedGraphGreedyCover({tiny_radius, direct_radius}, 2, 1, 1)[0] ==
         0);
  assert(QuantizedGraphGreedyCover({tiny_progress, direct_progress}, 2, 1,
                                   1)[0] == 1);
  std::mt19937 rng(52026);
  std::normal_distribution<float> normal(0, 1);
  size_t checks = 0;
  double worst = 0;
  for (size_t d : {1, 16, 64, 768})
    for (int trial = 0; trial < 1000; ++trial) {
      std::vector<float> r(d), t(d), h(d);
      double rr = 0, tt = 0, l1 = 0;
      for (size_t i = 0; i < d; ++i) {
        r[i] = normal(rng);
        t[i] = r[i] + normal(rng);
        rr += double(r[i]) * r[i];
        tt += double(t[i]) * t[i];
        l1 += std::abs(r[i]);
      }
      const float amplitude = l1 > 0 ? rr / l1 : 0;
      for (size_t i = 0; i < d; ++i) h[i] = r[i] > 0 ? amplitude : -amplitude;
      auto dots = QuantizedGraphCoverDots(t.data(), r.data(), h.data(), d);
      double exact_distance = 0, estimated_distance = tt + rr;
      for (size_t i = 0; i < d; ++i) {
        exact_distance += std::pow(double(t[i]) - r[i], 2);
        estimated_distance -= 2 * double(t[i]) * h[i];
      }
      for (bool quantized : {false, true}) {
        double expected = std::clamp(
            (tt - (quantized ? std::max(exact_distance, estimated_distance)
                             : exact_distance)) /
                tt,
            0., 1.);
        double actual =
            QuantizedGraphRelativeProgress(tt, rr, dots[0], dots[1], quantized);
        worst = std::max(worst, std::abs(actual - expected));
        assert(std::abs(actual - expected) < 2e-5);
        ++checks;
        if (actual > 0) assert(exact_distance <= tt * (1 - actual) + 3e-5 * tt);
      }
    }
  assert(QuantizedGraphRelativeProgress(0, 1, 1, 1, true) == 0);
  assert(QuantizedGraphRelativeProgress(1, 0, 0, 0, true) == 0);
  std::cout << "progress identity/contraction checks=" << checks
            << " worst_gap=" << worst << " radius counterexample passed\n";
}
