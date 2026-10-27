#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#include "algorithm/quantized_graph/quantized_graph_half_center.h"
int main() {
  std::mt19937 rng(20261025);
  std::normal_distribution<float> normal;
  size_t checks = 0;
  for (size_t n : {1, 15, 16, 65, 768, 960, 1024})
    for (size_t trial = 0; trial < 3000; ++trial) {
      std::vector<float> u(n), q(n), a(n);
      std::vector<uint16_t> code(n);
      for (auto &x : u) x = normal(rng);
      for (auto &x : q) x = normal(rng);
      auto normalize = [](auto &v) {
        double norm = 0;
        for (float x : v) norm += double(x) * x;
        for (auto &x : v) x /= std::sqrt(norm);
      };
      normalize(u);
      normalize(q);
      if (trial % 4 == 0) q = u;
      if (trial % 4 == 1) {
        q = u;
        q[0] += 1e-5f;
        normalize(q);
      }
      if (trial % 100 == 0) std::fill(u.begin(), u.end(), 0.f);
      if (trial % 101 == 0) std::fill(q.begin(), q.end(), 0.f);
      float error, norm2;
      assert(zvec::core::HalfCenter::encode(u.data(), n, code.data(), error,
                                            norm2));
      double qnorm2 = 0, real_dot = 0;
      float dot = 0;
      for (size_t i = 0; i < n; ++i) {
        qnorm2 += double(q[i]) * q[i];
        real_dot += double(q[i]) * u[i];
        dot = std::fma(q[i], u[i], dot);
      }
      const float approx =
          zvec::core::HalfCenter::cosine_distance(q.data(), code.data(), n);
      const float lower = zvec::core::SQ8Center::cosine_lower_bound(
          approx, error, norm2, std::nextafter(float(qnorm2), INFINITY), n);
      const float upper =
          std::nextafter(float(2. * double(approx) - lower), INFINITY);
      assert(lower <= 1.f - dot);
      assert(double(lower) <= 1. - real_dot);
      assert(upper >= 1.f - dot);
      assert(double(upper) >= 1. - real_dot);
      ++checks;
    }
  std::cout << "FP16 cosine interval checks=" << checks << " passed\n";
}
