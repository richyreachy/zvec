#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm/quantized_graph/quantized_graph_half_center.h>
#include <algorithm/quantized_graph/quantized_graph_integer_center.h>
using zvec::core::IntegerCenter;
using zvec::core::SQ8Center;
int main() {
  assert(__builtin_cpu_supports("avx512vnni"));
  constexpr size_t d = 768;
  std::mt19937 rng(55026);
  std::normal_distribution<float> normal(0, 1);
  std::uniform_int_distribution<size_t> select(0, 999999);
  std::vector<uint8_t> codes(d);
  std::vector<int8_t> query_codes(d);
  size_t integer_checks = 0, interval_checks = 0, query_error_checks = 0;
  for (size_t n : {size_t(1), size_t(63), size_t(64), size_t(65), d})
    for (size_t t = 0; t < 100; ++t) {
      int64_t scalar = 0;
      for (size_t j = 0; j < n; ++j) {
        codes[j] = t < 2 ? 255 : uint8_t(rng());
        query_codes[j] = t == 0   ? 127
                         : t == 1 ? -127
                                  : int8_t(int(rng() % 255) - 127);
        scalar += int64_t(codes[j]) * query_codes[j];
      }
      assert(scalar == IntegerCenter::dot(codes.data(), query_codes.data(), n));
      ++integer_checks;
    }
  std::ifstream base(
      "/home/xingrui/bench/symphony_qg/cohere/original/data/base.f32",
      std::ios::binary),
      mf("/home/xingrui/bench/quantized_graph/research_20261025/mean54/"
         "mean.f32",
         std::ios::binary);
  assert(base && mf);
  std::vector<float> mean(d), c(d), q(d);
  mf.read(reinterpret_cast<char *>(mean.data()), d * sizeof(float));
  assert(mf);
  auto norm2 = [](const std::vector<float> &x) {
    double sum = 0;
    for (float v : x) sum += double(v) * v;
    return sum;
  };
  auto normalize = [&](std::vector<float> &x) {
    double n = std::sqrt(norm2(x));
    for (float &v : x) v /= n;
  };
  double mean_norm2 = norm2(mean);
  IntegerCenter::Query iq;
  double max_error = 0, min_slack = 1;
  for (size_t sample = 0; sample < 1000; ++sample) {
    base.seekg(select(rng) * d * sizeof(float));
    base.read(reinterpret_cast<char *>(c.data()), d * sizeof(float));
    assert(base);
    normalize(c);
    float e, n, o, s, b, be, cn, vn;
    double mc;
    assert(SQ8Center::encode_shifted(c.data(), mean.data(), d, codes.data(), e,
                                     n, o, s, b, be));
    IntegerCenter::annotate(c.data(), mean.data(), codes.data(), d, o, s, e, cn,
                            vn, mc);
    for (size_t t = 0; t < 8; ++t) {
      for (size_t j = 0; j < d; ++j)
        q[j] = t == 0   ? mean[j]
               : t == 1 ? -mean[j]
               : t < 6  ? c[j] + (.001f * (1 + t * 5)) * normal(rng)
                        : normal(rng);
      if (t) normalize(q);
      assert(IntegerCenter::prepare(q.data(), mean.data(), d, iq));
      double err2 = 0, qres2 = 0, qu2 = 0, actual_dot = 0;
      int32_t sum = 0;
      for (size_t j = 0; j < d; ++j) {
        double r = double(q[j]) - mean[j], u = double(iq.scale) * iq.codes[j];
        err2 += (r - u) * (r - u);
        qres2 += r * r;
        qu2 += u * u;
        sum += iq.codes[j];
        actual_dot += u * (double(s) * codes[j] + o);
      }
      assert(std::sqrt(err2) <= iq.error &&
             std::sqrt(qres2) <= iq.residual_norm &&
             std::sqrt(qu2) <= iq.coded_norm && sum == iq.code_sum);
      ++query_error_checks;
      double formula =
          double(iq.scale) *
          (double(s) * IntegerCenter::dot(codes.data(), iq.codes.data(), d) +
           double(o) * iq.code_sum);
      assert(std::abs(formula - actual_dot) < 1e-12);
      ++integer_checks;
      double truth = 1;
      float float_dot = 0;
      for (size_t j = 0; j < d; ++j) {
        truth -= double(q[j]) * c[j];
        float_dot += q[j] * c[j];
      }
      float approximate =
          IntegerCenter::distance(iq, codes.data(), d, o, s, mc, mean_norm2);
      float lower = IntegerCenter::lower_bound(approximate, iq, e, cn, vn, n,
                                               mc, mean_norm2, d);
      float upper = std::nextafter(float(2. * approximate - lower),
                                   std::numeric_limits<float>::infinity());
      assert(lower <= truth && truth <= upper);
      assert(lower <= 1.f - float_dot && 1.f - float_dot <= upper);
      interval_checks += 2;
      max_error = std::max(max_error, std::abs(approximate - truth));
      min_slack = std::min(min_slack, std::min(truth - lower, upper - truth));
    }
  }
  // Zero mean and exact zero residuals exercise the constant-code path.
  std::vector<float> zero(d, 0);
  assert(IntegerCenter::prepare(zero.data(), zero.data(), d, iq));
  assert(iq.code_sum == 0 && iq.scale == 0);
  std::cout << "integer center checks=" << integer_checks
            << " interval checks=" << interval_checks
            << " query error checks=" << query_error_checks
            << " max_distance_error=" << max_error
            << " min_interval_slack=" << min_slack << " passed\n";
}
