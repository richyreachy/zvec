#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm/quantized_graph/quantized_graph_half_center.h>
#include <turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h>
int main() {
#if RABITQ_SUPPORTED
  constexpr size_t d = 768;
  using zvec::core::SQ8Center;
  zvec::turbo::SymphonyQGBlockQuantizer codec(d, true);
  std::ifstream base(
      "/home/xingrui/bench/symphony_qg/cohere/original/data/base.f32",
      std::ios::binary);
  std::ifstream mf(
      "/home/xingrui/bench/quantized_graph/research_20261025/mean54/mean.f32",
      std::ios::binary);
  assert(base && mf);
  std::vector<float> mean(d), c(d), q(d), decoded(d), neighbor(d), rc, rn;
  mf.read(reinterpret_cast<char *>(mean.data()), d * sizeof(float));
  assert(mf);
  std::mt19937 rng(54026);
  std::normal_distribution<float> normal(0, 1);
  std::uniform_int_distribution<size_t> select(0, 999999);
  std::vector<uint8_t> codes(d), raw(d);
  std::vector<double> errors, raw_errors;
  size_t intervals = 0, identities = 0, anchors = 0;
  double max_identity = 0;
  auto norm2 = [](const std::vector<float> &x) {
    double sum = 0;
    for (float v : x) sum += double(v) * v;
    return sum;
  };
  auto normalize = [&](std::vector<float> &x) {
    double n = std::sqrt(norm2(x));
    for (float &v : x) v /= n;
  };
  const double mn = norm2(mean);
  for (size_t sample = 0; sample < 1000; ++sample) {
    base.seekg(select(rng) * d * sizeof(float));
    base.read(reinterpret_cast<char *>(c.data()), d * sizeof(float));
    assert(base);
    normalize(c);
    float e, n, o, s, b, be, e0, n0, o0, s0;
    assert(SQ8Center::encode_shifted(c.data(), mean.data(), d, codes.data(), e,
                                     n, o, s, b, be));
    assert(SQ8Center::encode(c.data(), d, raw.data(), e0, n0, o0, s0));
    errors.push_back(e);
    raw_errors.push_back(e0);
    SQ8Center::decode_shifted(codes.data(), mean.data(), d, decoded.data(), o,
                              s);
    double true_beta = 0, actual_error2 = 0;
    for (size_t j = 0; j < d; ++j) {
      double err = double(decoded[j]) - c[j];
      true_beta += double(mean[j]) * err;
      actual_error2 += err * err;
    }
    assert(std::sqrt(actual_error2) <= e);
    assert(std::abs(double(b) - true_beta) <= be);
    for (size_t trial = 0; trial < 8; ++trial) {
      for (size_t j = 0; j < d; ++j)
        q[j] = trial == 0   ? mean[j]
               : trial == 1 ? -mean[j]
               : trial < 6  ? c[j] + (.001f * (1 + trial * 5)) * normal(rng)
                            : normal(rng);
      normalize(q);
      double truth = 1, approx_real = 1, dotm = 0;
      for (size_t j = 0; j < d; ++j) {
        truth -= double(q[j]) * c[j];
        approx_real -= double(q[j]) * decoded[j];
        dotm += double(q[j]) * mean[j];
      }
      const float alpha = float(dotm / mn);
      double residual_norm = 0, residual_error_dot = 0;
      for (size_t j = 0; j < d; ++j) {
        double r = double(q[j]) - double(alpha) * mean[j];
        residual_norm += r * r;
        residual_error_dot += r * (double(decoded[j]) - c[j]);
      }
      const double identity_error = std::abs(
          approx_real + double(alpha) * true_beta - truth + residual_error_dot);
      assert(identity_error < 1e-12);
      max_identity = std::max(max_identity, identity_error);
      ++identities;
      const double pad =
          8. * (d + 4) * std::numeric_limits<double>::epsilon() *
          (std::sqrt(norm2(q)) + std::abs(double(alpha)) * std::sqrt(mn));
      float qr =
          std::nextafter(float(std::pow(std::sqrt(residual_norm) + pad, 2)),
                         std::numeric_limits<float>::infinity());
      float qn = std::nextafter(float(norm2(q)),
                                std::numeric_limits<float>::infinity());
      float approx = SQ8Center::cosine_distance_shifted(q.data(), codes.data(),
                                                        mean.data(), d, o, s);
      float lo = SQ8Center::cosine_lower_bound(approx, e, n, qn, d);
      float hi = std::nextafter(float(2. * approx - lo),
                                std::numeric_limits<float>::infinity());
      assert(lo <= truth && truth <= hi);
      ++intervals;
      float corrected = std::fma(alpha, b, approx);
      lo = SQ8Center::projected_lower_bound(corrected, e, n, qn, qr, alpha, b,
                                            be, d);
      hi = std::nextafter(float(2. * corrected - lo),
                          std::numeric_limits<float>::infinity());
      assert(lo <= truth && truth <= hi);
      ++intervals;
    }
    for (size_t j = 0; j < d; ++j) neighbor[j] = c[j] + .01f * normal(rng);
    normalize(neighbor);
    codec.transform(decoded.data(), rc);
    codec.transform(neighbor.data(), rn);
    std::vector<char> edge(codec.batch_bytes());
    codec.encode(rc.data(), rn.data(), 1, edge.data());
    rabitqlib::ConstQGBatchDataMap<float> batch(edge.data(), d);
    decltype(codec)::Query query;
    codec.prepare_query(neighbor.data(), query);
    double dist = 1, dot = 0;
    for (size_t j = 0; j < d; ++j) {
      dist -= double(query.rotated[j]) * rc[j];
      dot += query.rotated[j] * (rn[j] > rc[j] ? .5 : -.5);
    }
    double ideal = dist + batch.f_add()[0] + batch.f_rescale()[0] * dot;
    assert(std::abs(ideal) < 1e-4);
    float scores[32];
    codec.scan(edge.data(), query, float(dist), scores);
    assert(std::abs(scores[0] - ideal) <= std::abs(batch.f_rescale()[0]) *
                                                  query.lookup.delta() *
                                                  (d / 4) * .5 +
                                              1e-4);
    ++anchors;
  }
  // Degenerate zero shift and constant vector must remain finite.
  std::vector<float> zero(d, 0), constant(d, .125f);
  float e, n, o, s, b, be;
  assert(SQ8Center::encode_shifted(constant.data(), zero.data(), d,
                                   codes.data(), e, n, o, s, b, be));
  assert(b == 0);
  SQ8Center::decode_shifted(codes.data(), zero.data(), d, decoded.data(), o, s);
  assert(decoded == constant);
  std::sort(errors.begin(), errors.end());
  std::sort(raw_errors.begin(), raw_errors.end());
  std::cout << "mean-center interval checks=" << intervals
            << " projection identities=" << identities
            << " anchor/LUT checks=" << anchors
            << " raw_error_median=" << raw_errors[500]
            << " shifted_error_median=" << errors[500]
            << " max_identity_error=" << max_identity << " passed\n";
#else
#error Required ISA not enabled; refusing an empty test
#endif
}
