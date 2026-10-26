#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h>
using zvec::turbo::RadialCalibration;
int main() {
#if RABITQ_SUPPORTED
  constexpr size_t d = 768;
  zvec::turbo::SymphonyQGBlockQuantizer codec(d, true);
  assert(codec.batch_bytes() ==
         rabitqlib::QGBatchDataMap<float>::data_bytes(d));
  std::ifstream base(
      "/home/xingrui/bench/symphony_qg/cohere/original/data/base.f32",
      std::ios::binary);
  assert(base);
  std::mt19937 rng(57026);
  std::normal_distribution<float> normal(0, 1);
  std::uniform_int_distribution<size_t> select(0, 999999);
  std::vector<float> c(d), v(d), q(d), rc, rv, tangent(d), h(d), hp(d);
  std::vector<char> native(codec.batch_bytes()), adjusted(codec.batch_bytes());
  auto dot = [](const std::vector<float> &a, const std::vector<float> &b) {
    double x = 0;
    for (size_t j = 0; j < a.size(); ++j) x += double(a[j]) * b[j];
    return x;
  };
  auto normalize = [&](std::vector<float> &a) {
    double n = std::sqrt(dot(a, a));
    for (float &x : a) x /= n;
  };
  size_t constraints = 0, circle = 0, lut = 0, worse = 0, fallback = 0;
  double max_constraint = 0, max_circle_error = 0, native_mse = 0,
         adjusted_mse = 0;
  for (size_t sample = 0; sample < 1000; ++sample) {
    base.seekg(select(rng) * d * sizeof(float));
    base.read(reinterpret_cast<char *>(c.data()), d * sizeof(float));
    assert(base);
    normalize(c);
    if (sample % 2) {
      base.seekg(select(rng) * d * sizeof(float));
      base.read(reinterpret_cast<char *>(v.data()), d * sizeof(float));
      assert(base);
    } else {
      for (size_t j = 0; j < d; ++j)
        v[j] = c[j] + (.001f + .001f * (sample % 50)) * normal(rng);
    }
    normalize(v);
    codec.transform(c.data(), rc);
    codec.transform(v.data(), rv);
    rabitqlib::quant::quantize_qg_batch(rv.data(), rc.data(), 1, d,
                                        native.data(), rabitqlib::METRIC_L2);
    rabitqlib::QGBatchDataMap<float> raw(native.data(), d);
    zvec::turbo::ConvertSymphonyQGCosineFactors(rc.data(), rv.data(), 1, d,
                                                raw.f_add(), raw.f_rescale());
    double R = 0, C = 0, B = 0, H = 0, T = 0;
    float amplitude = -.5f * raw.f_rescale()[0];
    for (size_t j = 0; j < d; ++j) {
      double r = double(rv[j]) - rc[j];
      h[j] = r > 0 ? amplitude : -amplitude;
      R += r * r;
      C += double(rc[j]) * rc[j];
      B += double(rc[j]) * r;
      H += double(rc[j]) * h[j];
      T += r * h[j];
    }
    auto f = RadialCalibration::fit(R, C, B, H, T);
    fallback += f.kind == RadialCalibration::kRadialFallback;
    if (f.kind == RadialCalibration::kRegular) {
      double err = std::max(std::abs(f.a * T + f.b * B - R),
                            std::abs(f.a * H + f.b * C - B));
      assert(err < 1e-10);
      max_constraint = std::max(max_constraint, err);
      constraints += 2;
    }
    codec.encode(rc.data(), rv.data(), 1, adjusted.data());
    rabitqlib::ConstQGBatchDataMap<float> batch(adjusted.data(), d);
    assert(batch.f_add()[0] == float(-f.b));
    assert(batch.f_rescale()[0] == float(double(raw.f_rescale()[0]) * f.a));
    double e0 = 0, e1 = 0;
    for (size_t j = 0; j < d; ++j) {
      double r = double(rv[j]) - rc[j];
      hp[j] = float(f.a * h[j] + f.b * rc[j]);
      e0 += (h[j] - r) * (h[j] - r);
      e1 += (hp[j] - r) * (hp[j] - r);
    }
    worse += e1 > e0;
    native_mse += e0;
    adjusted_mse += e1;
    const double cv = dot(c, v);
    for (size_t j = 0; j < d; ++j) tangent[j] = float(v[j] - cv * c[j]);
    normalize(tangent);
    for (size_t trial = 0; trial < 8; ++trial) {
      double theta = (trial - 3.) * .45;
      for (size_t j = 0; j < d; ++j)
        q[j] = float(std::cos(theta) * c[j] + std::sin(theta) * tangent[j]);
      normalize(q);
      decltype(codec)::Query query;
      codec.prepare_query(q.data(), query);
      double center = 1, truth = 1, sign_dot = 0;
      for (size_t j = 0; j < d; ++j) {
        center -= double(query.rotated[j]) * rc[j];
        truth -= double(query.rotated[j]) * rv[j];
        sign_dot += query.rotated[j] * (rv[j] > rc[j] ? .5 : -.5);
      }
      double ideal = (1. - batch.f_add()[0]) * center + batch.f_add()[0] +
                     batch.f_rescale()[0] * sign_dot;
      if (f.kind == RadialCalibration::kRegular) {
        double err = std::abs(ideal - truth);
        assert(err < 3e-5);
        max_circle_error = std::max(max_circle_error, err);
        ++circle;
      }
      float scores[32];
      codec.scan(adjusted.data(), query, float(center), scores);
      double bound =
          std::abs(batch.f_rescale()[0]) * query.lookup.delta() * (d / 4) * .5 +
          1e-4;
      assert(std::abs(scores[0] - ideal) <= bound);
      ++lut;
    }
  }
  auto zero = RadialCalibration::fit(0, 1, 0, 0, 0);
  assert(zero.kind == RadialCalibration::kZeroResidual && zero.a == 0 &&
         zero.b == 0);
  auto origin = RadialCalibration::fit(1, 0, 0, 0, 1);
  assert(origin.kind == RadialCalibration::kZeroCenter && origin.a == 1 &&
         origin.b == 0);
  auto antipodal = RadialCalibration::fit(4, 1, -2, -2, 4);
  assert(antipodal.kind == RadialCalibration::kRadialFallback &&
         antipodal.a == 0 && antipodal.b == -2);
  auto singular = RadialCalibration::fit(1, 1, .5, 2, 1);
  assert(singular.kind == RadialCalibration::kRadialFallback);
  assert(std::abs(singular.a + singular.b * .5 - 1) > .1);
  std::cout << "radial constraints=" << constraints
            << " great_circle_checks=" << circle << " LUT_checks=" << lut
            << " degeneracy_checks=4 sample_fallback=" << fallback
            << " error_norm_worse=" << worse
            << " native_error2_mean=" << native_mse / 1000
            << " adjusted_error2_mean=" << adjusted_mse / 1000
            << " max_constraint_error=" << max_constraint
            << " max_circle_error=" << max_circle_error << " passed\n";
#else
#error Required ISA disabled
#endif
}
