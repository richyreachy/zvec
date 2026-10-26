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
  std::mt19937 rng(71026);
  std::normal_distribution<float> normal(0, .1);
  size_t checks = 0;
  double worst_endpoint = 0, worst_lut = 0;
  for (size_t d : {64, 768, 960})
    for (bool cosine : {false, true})
      for (size_t count : {1, 7, 32}) {
        zvec::turbo::SymphonyQGBlockQuantizer codec(d, cosine);
        const size_t p = codec.encoded_dim();
        std::vector<float> u(d), raw(count * d), center, values(count * p), tmp;
        for (auto &x : u) x = normal(rng);
        auto normalize = [](float *x, size_t n) {
          double s = 0;
          for (size_t j = 0; j < n; ++j) s += double(x[j]) * x[j];
          if (s > 0)
            for (size_t j = 0; j < n; ++j) x[j] /= std::sqrt(s);
        };
        if (cosine) normalize(u.data(), d);
        codec.transform(u.data(), center);
        for (size_t i = 0; i < count; ++i) {
          for (size_t j = 0; j < d; ++j) raw[i * d + j] = u[j] + normal(rng);
          if (cosine) normalize(raw.data() + i * d, d);
          codec.transform(raw.data() + i * d, tmp);
          std::copy(tmp.begin(), tmp.end(), values.begin() + i * p);
        }
        std::vector<char> codes(codec.batch_bytes());
        codec.encode(center.data(), values.data(), count, codes.data());
        rabitqlib::ConstQGBatchDataMap<float> a(codes.data(), p),
            b(codes.data() + codec.batch_bytes() / 2, p);
        for (size_t i = 0; i < count; ++i) {
          decltype(codec)::Query query;
          codec.prepare_query(raw.data() + i * d, query);
          double exact = cosine ? 1 : 0;
          for (size_t j = 0; j < d; ++j) {
            if (cosine)
              exact -= double(u[j]) * raw[i * d + j];
            else {
              double r = double(u[j]) - raw[i * d + j];
              exact += r * r;
            }
          }
          double l1 = 0;
          for (size_t j = 0; j < p; ++j)
            l1 += std::abs(double(values[i * p + j] - center[j]));
          const float amplitude = float(l1 / p);
          double dot1 = 0, dot2 = 0;
          for (size_t j = 0; j < p; ++j) {
            float r = values[i * p + j] - center[j],
                  e = r - (r > 0 ? amplitude : -amplitude);
            dot1 += query.rotated[j] * (r > 0 ? .5 : -.5);
            dot2 += query.rotated[j] * (e > 0 ? .5 : -.5);
          }
          double ideal = a.f_add()[i] + b.f_add()[i] + exact +
                         a.f_rescale()[i] * dot1 + b.f_rescale()[i] * dot2;
          worst_endpoint = std::max(worst_endpoint, std::abs(ideal));
          assert(std::abs(ideal) < 1e-4 * (1 + exact));
          float scores[32];
          codec.scan(codes.data(), query, float(exact), scores);
          double gap = std::abs(double(scores[i]) - ideal);
          worst_lut = std::max(worst_lut, gap);
          double bound =
              (std::abs(a.f_rescale()[i]) + std::abs(b.f_rescale()[i])) *
                  query.lookup.delta() * (p / 4) * .5 +
              1e-4 * (1 + exact);
          assert(gap <= bound);
          for (size_t j = 0; j < count; ++j) assert(std::isfinite(scores[j]));
          ++checks;
        }
      }
  std::cout << "two-stage endpoint and LUT checks=" << checks
            << " worst_endpoint=" << worst_endpoint
            << " worst_lut=" << worst_lut << " passed\n";
#endif
}
