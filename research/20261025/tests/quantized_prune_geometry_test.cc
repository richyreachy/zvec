#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h>
int main() {
#if RABITQ_SUPPORTED
  std::mt19937 rng(2601044);
  std::normal_distribution<float> normal(0, .1);
  size_t score_checks = 0, radius_checks = 0;
  double max_gap = 0;
  auto normalize = [](float *x, size_t n) {
    double s = 0;
    for (size_t j = 0; j < n; ++j) s += double(x[j]) * x[j];
    for (size_t j = 0; j < n; ++j) x[j] /= std::sqrt(s);
  };
  for (size_t d : {64, 768, 960})
    for (bool cosine : {false, true}) {
      zvec::turbo::SymphonyQGBlockQuantizer codec(d, cosine);
      const size_t p = codec.encoded_dim(), n = 32;
      std::vector<float> u(d), raw(n * d), center, values(n * p), tmp,
          residual(n * p), hat(n * p);
      std::vector<double> norm2(n);
      for (auto &x : u) x = normal(rng);
      if (cosine) normalize(u.data(), d);
      codec.transform(u.data(), center);
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) raw[i * d + j] = u[j] + normal(rng);
        if (cosine) normalize(raw.data() + i * d, d);
        codec.transform(raw.data() + i * d, tmp);
        std::copy(tmp.begin(), tmp.end(), values.begin() + i * p);
        double l1 = 0;
        for (size_t j = 0; j < p; ++j) {
          float r = tmp[j] - center[j];
          residual[i * p + j] = r;
          norm2[i] += double(r) * r;
          l1 += std::abs(double(r));
        }
        double b = l1 > 0 ? norm2[i] / l1 : 0;
        for (size_t j = 0; j < p; ++j)
          hat[i * p + j] = float(residual[i * p + j] > 0 ? b : -b);
      }
      std::vector<char> codes(codec.batch_bytes());
      codec.encode(center.data(), values.data(), n, codes.data());
      rabitqlib::ConstQGBatchDataMap<float> batch(codes.data(), p);
      for (size_t i = 0; i < n; ++i) {
        decltype(codec)::Query query;
        codec.prepare_query(raw.data() + i * d, query);
        double center_distance = cosine ? 1 : 0;
        for (size_t k = 0; k < d; ++k) {
          if (cosine)
            center_distance -= double(raw[i * d + k]) * u[k];
          else {
            double z = double(raw[i * d + k]) - u[k];
            center_distance += z * z;
          }
        }
        for (size_t j = 0; j < n; ++j) {
          double lattice = 0, inner = 0;
          for (size_t k = 0; k < p; ++k) {
            lattice += query.rotated[k] * (residual[j * p + k] > 0 ? .5 : -.5);
            inner += double(residual[i * p + k]) * hat[j * p + k];
          }
          double native = batch.f_add()[j] + center_distance +
                          batch.f_rescale()[j] * lattice;
          double geometric =
              (norm2[i] + norm2[j] - 2 * inner) * (cosine ? .5 : 1.);
          double gap = std::abs(native - geometric);
          max_gap = std::max(max_gap, gap);
          assert(gap < 2e-4 * (1 + norm2[i] + norm2[j]));
          ++score_checks;
        }
      }
    }
  for (size_t trial = 0; trial < 10000; ++trial) {
    std::vector<double> s(16), r(16), h(16);
    double ns = 0, l1 = 0, nr = 0, nh = 0, rs = 0, rh = 0;
    for (size_t j = 0; j < 16; ++j) {
      s[j] = normal(rng);
      r[j] = s[j] + .4 * normal(rng);
      ns += s[j] * s[j];
      l1 += std::abs(s[j]);
      nr += r[j] * r[j];
    }
    for (size_t j = 0; j < 16; ++j) {
      h[j] = (s[j] > 0 ? 1 : -1) * ns / l1;
      nh += h[j] * h[j];
      rs += r[j] * s[j];
      rh += r[j] * h[j];
    }
    const double mt = 2 * rs - ns, me = 2 * rh - ns;
    if (mt <= 0 || me <= 0) continue;
    double rt = mt / (2 * std::sqrt(ns)), re = me / (2 * std::sqrt(nh)),
           radius = std::min(rt, re);
    for (bool target_h : {false, true}) {
      const auto &v = target_h ? h : s;
      double nv = std::sqrt(target_h ? nh : ns), dot_s = 0, dot_h = 0;
      for (size_t j = 0; j < 16; ++j) {
        double q = r[j] - .999 * radius * v[j] / nv;
        dot_s += q * s[j];
        dot_h += q * h[j];
      }
      assert(ns - 2 * dot_s < 0 && ns - 2 * dot_h < 0);
      ++radius_checks;
    }
    const auto &v = rt < re ? s : h;
    double nv = std::sqrt(rt < re ? ns : nh), dot = 0;
    for (size_t j = 0; j < 16; ++j)
      dot += (r[j] - 1.001 * radius * v[j] / nv) * v[j];
    assert(ns - 2 * dot > 0);
    ++radius_checks;
  }
  std::cout << "native ideal score checks=" << score_checks
            << " max_gap=" << max_gap
            << " local-radius checks=" << radius_checks << " passed\n";
#endif
}
