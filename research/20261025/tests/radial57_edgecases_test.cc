#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>
#include <turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h>
int main() {
#if RABITQ_SUPPORTED
  constexpr size_t d = 768;
  zvec::turbo::SymphonyQGBlockQuantizer codec(d, true);
  size_t checks = 0;
  std::vector<float> c(d, 0), v(d, 0), q(d, 0), rc, rv;
  std::vector<char> codes(codec.batch_bytes());
  for (int scenario = 0; scenario < 4; ++scenario) {
    std::fill(c.begin(), c.end(), 0);
    std::fill(v.begin(), v.end(), 0);
    if (scenario != 2) c[0] = 1;
    v[0] = scenario == 1 ? -1 : 1;
    if (scenario == 3) {
      v[1] = 1e-5f;
      double n = std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1]);
      for (float &x : v) x /= n;
    }
    codec.transform(c.data(), rc);
    codec.transform(v.data(), rv);
    codec.encode(rc.data(), rv.data(), 1, codes.data());
    rabitqlib::ConstQGBatchDataMap<float> batch(codes.data(), d);
    assert(std::isfinite(batch.f_add()[0]) &&
           std::isfinite(batch.f_rescale()[0]));
    for (int trial = 0; trial < 8; ++trial) {
      std::fill(q.begin(), q.end(), 0);
      q[0] = std::cos(trial * .4);
      q[1] = std::sin(trial * .4);
      decltype(codec)::Query query;
      codec.prepare_query(q.data(), query);
      double center = 1, truth = 1;
      for (size_t j = 0; j < d; ++j) {
        center -= double(query.rotated[j]) * rc[j];
        truth -= double(query.rotated[j]) * rv[j];
      }
      float result[32];
      codec.scan(codes.data(), query, float(center), result);
      assert(std::isfinite(result[0]));
      double bound =
          std::abs(batch.f_rescale()[0]) * query.lookup.delta() * (d / 4) * .5 +
          1e-4;
      assert(std::abs(result[0] - truth) <= bound);
      ++checks;
    }
  }
  std::cout << "radial actual-codec degenerate checks=" << checks
            << " passed\n";
#else
#error Required ISA disabled
#endif
}
