#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h>
using zvec::turbo::DirectionalCalibration;
int main() {
#if RABITQ_SUPPORTED
  constexpr size_t d = 768;
  zvec::turbo::SymphonyQGBlockQuantizer codec(d, true), l2(d, false);
  assert(codec.batch_bytes() ==
         rabitqlib::QGBatchDataMap<float>::data_bytes(d));
  std::mt19937 rng(62026);
  std::normal_distribution<float> normal(0, 1);
  std::uniform_int_distribution<size_t> select(0, 999999);
  std::ifstream base(
      "/home/xingrui/bench/symphony_qg/cohere/original/data/base.f32",
      std::ios::binary);
  assert(base);
  std::vector<float> c(d), v(d), q(d), rc, rv;
  std::vector<char> native(codec.batch_bytes()), adjusted(codec.batch_bytes()),
      l2raw(codec.batch_bytes()), l2got(codec.batch_bytes());
  auto dot = [](const auto &a, const auto &b) {
    double r = 0;
    for (size_t j = 0; j < a.size(); ++j) r += double(a[j]) * b[j];
    return r;
  };
  auto normalize = [&](auto &a) {
    double n = std::sqrt(dot(a, a));
    if (n > 0)
      for (float &x : a) x = float(x / n);
  };
  size_t factors = 0, luts = 0, endpoint = 0, identity = 0, l2checks = 0;
  double maxadd = 0, maxformula = 0, maxendpoint = 0;
  for (size_t sample = 0; sample < 1024; ++sample) {
    base.seekg(select(rng) * d * sizeof(float));
    base.read(reinterpret_cast<char *>(c.data()), d * sizeof(float));
    assert(base);
    normalize(c);
    if (sample % 2) {
      base.seekg(select(rng) * d * sizeof(float));
      base.read(reinterpret_cast<char *>(v.data()), d * sizeof(float));
      assert(base);
    } else
      for (size_t j = 0; j < d; ++j)
        v[j] = c[j] + (.001f + .001f * (sample % 50)) * normal(rng);
    normalize(v);
    bool unit = true;
    if (sample < 32) {
      switch (sample % 8) {
        case 0:
          v = c;
          break;
        case 1:
          for (size_t j = 0; j < d; ++j) v[j] = -c[j];
          break;
        case 2:
          std::fill(c.begin(), c.end(), 0);
          unit = false;
          break;
        case 3:
          std::fill(v.begin(), v.end(), 0);
          unit = false;
          break;
        case 4:
          std::fill(c.begin(), c.end(), 0);
          std::fill(v.begin(), v.end(), 0);
          unit = false;
          break;
        case 5:
          for (auto &x : c) x *= .3f;
          unit = false;
          break;
        case 6:
          for (auto &x : v) x *= .4f;
          unit = false;
          break;
        default:
          break;
      }
    }
    codec.transform(c.data(), rc);
    codec.transform(v.data(), rv);
    std::fill(native.begin(), native.end(), 0);
    std::fill(adjusted.begin(), adjusted.end(), 0);
    rabitqlib::quant::quantize_qg_batch(rv.data(), rc.data(), 1, d,
                                        native.data(), rabitqlib::METRIC_L2);
    l2raw = native;
    l2.encode(rc.data(), rv.data(), 1, l2got.data());
    assert(l2raw == l2got);
    ++l2checks;
    rabitqlib::QGBatchDataMap<float> raw(native.data(), d);
    zvec::turbo::ConvertSymphonyQGCosineFactors(rc.data(), rv.data(), 1, d,
                                                raw.f_add(), raw.f_rescale());
    double cr = 0, rr = 0, l1 = 0, ch = 0;
    double amp = -.5 * raw.f_rescale()[0];
    for (size_t j = 0; j < d; ++j) {
      double r = double(rv[j]) - rc[j];
      cr += double(rc[j]) * r;
      rr += r * r;
      l1 += std::abs(r);
      ch += double(rc[j]) * (r > 0 ? amp : -amp);
    }
    assert(std::isfinite(raw.f_add()[0]) && std::isfinite(raw.f_rescale()[0]));
    maxadd = std::max(maxadd, std::abs(raw.f_add()[0] - (ch - cr)));
    assert(maxadd < 2e-5);
#if defined(QG_DIRECTION_BETA)
    constexpr double fixed = QG_DIRECTION_BETA;
    const double beta = fixed;
#else
    const double beta = DirectionalCalibration::mse_beta(rr, l1, d);
#endif
    codec.encode(rc.data(), rv.data(), 1, adjusted.data());
    rabitqlib::ConstQGBatchDataMap<float> batch(adjusted.data(), d);
    assert(std::memcmp(raw.bin_code(), batch.bin_code(), d * 32 / 8) == 0);
    if (beta == 1) {
      assert(adjusted == native);
      ++identity;
    } else {
      assert(batch.f_add()[0] ==
             float(beta * double(raw.f_add()[0]) - (1 - beta) * cr));
      assert(batch.f_rescale()[0] == float(beta * double(raw.f_rescale()[0])));
    }
    ++factors;
    for (size_t trial = 0; trial < 8; ++trial) {
      if (trial == 0)
        q = c;
      else if (trial == 1)
        q = v;
      else {
        for (size_t j = 0; j < d; ++j)
          q[j] = trial % 2 ? normal(rng) : c[j] + .2f * normal(rng);
        normalize(q);
      }
      decltype(codec)::Query query;
      codec.prepare_query(q.data(), query);
      double center = 1 - dot(query.rotated, rc), sign_dot = 0, wh = 0;
      for (size_t j = 0; j < d; ++j) {
        double sign = rv[j] > rc[j] ? 1 : -1;
        sign_dot += query.rotated[j] * sign * .5;
        wh += (double(query.rotated[j]) - rc[j]) * amp * sign;
      }
      double ideal =
          center + batch.f_add()[0] + batch.f_rescale()[0] * sign_dot;
      double expected = center - cr - beta * wh;
      maxformula = std::max(maxformula, std::abs(ideal - expected));
      assert(maxformula < 3e-5);
      if (trial == 0) {
        assert(std::abs(ideal - (1 - dot(query.rotated, rv))) < 3e-5);
        ++endpoint;
      }
      if (trial == 1 && unit) {
        double e = std::abs(ideal - (1 - beta) * rr);
        maxendpoint = std::max(maxendpoint, e);
        assert(e < 3e-5);
        assert(ideal <= center + 3e-5);
        ++endpoint;
      }
      float scores[32];
      codec.scan(adjusted.data(), query, float(center), scores);
      double bound =
          std::abs(batch.f_rescale()[0]) * query.lookup.delta() * (d / 4) * .5 +
          1e-4;
      assert(std::isfinite(scores[0]));
      assert(std::abs(scores[0] - ideal) <= bound);
      ++luts;
    }
  }
  assert(DirectionalCalibration::mse_beta(0, 0, d) == 1);
  assert(DirectionalCalibration::mse_beta(1, 1, d) == .5);
  assert(DirectionalCalibration::mse_beta(d, d, d) == 1);
  std::cout << "factor_checks=" << factors << " LUT_checks=" << luts
            << " endpoint_checks=" << endpoint
            << " identity_blocks=" << identity
            << " unchanged_L2_blocks=" << l2checks
            << " max_original_add_error=" << maxadd
            << " max_formula_error=" << maxformula
            << " max_endpoint_error=" << maxendpoint << " passed\n";
#else
#error Required ISA disabled
#endif
}
