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
  assert(base);
  std::mt19937 rng(53026);
  std::normal_distribution<float> normal(0, 1);
  std::uniform_int_distribution<size_t> select(0, 999999);
  std::vector<float> c(d), q(d), rc, rq, decoded(d), neighbor(d), rn;
  std::vector<uint8_t> raw_code(d), rot_code(d);
  std::vector<double> errors_raw, errors_rot;
  size_t interval_checks = 0, anchor_checks = 0;
  double worst_rotation_norm = 0;
  auto norm2 = [](const std::vector<float> &x) {
    double sum = 0;
    for (float v : x) sum += double(v) * v;
    return sum;
  };
  auto normalize = [&](std::vector<float> &x) {
    double n = std::sqrt(norm2(x));
    for (float &v : x) v /= n;
  };
  for (size_t sample = 0; sample < 1000; ++sample) {
    base.seekg(select(rng) * d * sizeof(float));
    base.read(reinterpret_cast<char *>(c.data()), d * sizeof(float));
    assert(base);
    normalize(c);
    codec.transform(c.data(), rc);
    float e0, n0, o0, s0, e, n, o, s;
    assert(SQ8Center::encode(c.data(), d, raw_code.data(), e0, n0, o0, s0));
    assert(SQ8Center::encode(rc.data(), d, rot_code.data(), e, n, o, s));
    errors_raw.push_back(e0);
    errors_rot.push_back(e);
    SQ8Center::decode(rot_code.data(), d, decoded.data(), o, s);
    double norm_gap = std::abs(std::sqrt(norm2(rc)) - std::sqrt(norm2(c)));
    worst_rotation_norm = std::max(worst_rotation_norm, norm_gap);
    assert(norm_gap < 32 * std::numeric_limits<float>::epsilon());
    for (size_t trial = 0; trial < 8; ++trial) {
      for (size_t j = 0; j < d; ++j)
        q[j] = trial < 6 ? c[j] + (.001f * (1 + trial * 5)) * normal(rng)
                         : normal(rng);
      normalize(q);
      codec.transform(q.data(), rq);
      double truth = 1;
      for (size_t j = 0; j < d; ++j) truth -= double(q[j]) * c[j];
      float approximate =
          SQ8Center::cosine_distance(rq.data(), rot_code.data(), d, o, s);
      float qnorm = std::nextafter(float(norm2(q)),
                                   std::numeric_limits<float>::infinity());
      float lower = SQ8Center::rotated_lower_bound(
          SQ8Center::cosine_lower_bound(approximate, e, n, qnorm, d), e, n,
          qnorm);
      float upper = std::nextafter(float(2. * approximate - lower),
                                   std::numeric_limits<float>::infinity());
      assert(lower <= truth && truth <= upper);
      ++interval_checks;
    }
    for (size_t j = 0; j < d; ++j) neighbor[j] = c[j] + .01f * normal(rng);
    normalize(neighbor);
    codec.transform(neighbor.data(), rn);
    std::vector<char> codes(codec.batch_bytes());
    codec.encode(decoded.data(), rn.data(), 1, codes.data());
    rabitqlib::ConstQGBatchDataMap<float> batch(codes.data(), d);
    decltype(codec)::Query query;
    codec.prepare_query(neighbor.data(), query);
    double center_distance = 1, dot = 0;
    for (size_t j = 0; j < d; ++j) {
      center_distance -= double(query.rotated[j]) * decoded[j];
      dot += query.rotated[j] * (rn[j] > decoded[j] ? .5 : -.5);
    }
    double ideal =
        center_distance + batch.f_add()[0] + batch.f_rescale()[0] * dot;
    assert(std::abs(ideal) < 1e-4);
    float scores[32];
    codec.scan(codes.data(), query, float(center_distance), scores);
    double bound =
        std::abs(batch.f_rescale()[0]) * query.lookup.delta() * (d / 4) * .5 +
        1e-4;
    assert(std::abs(scores[0] - ideal) <= bound);
    ++anchor_checks;
  }
  std::sort(errors_raw.begin(), errors_raw.end());
  std::sort(errors_rot.begin(), errors_rot.end());
  std::cout << "rotated-center interval checks=" << interval_checks
            << " anchor/LUT checks=" << anchor_checks
            << " raw_error_median=" << errors_raw[500]
            << " rotated_error_median=" << errors_rot[500]
            << " worst_rotation_norm_gap=" << worst_rotation_norm
            << " passed\n";
#endif
}
