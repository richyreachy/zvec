#define main graph65_program_main
#include "graph65_refine.cc"
#undef main
#define QG_BUCKET_WIDTH .016f
#include "core/algorithm/quantized_graph/quantized_graph_fusion.h"
#include "turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h"
namespace zvec::turbo {
bool symqg_mul(const float *, float *, size_t) {
  return false;
}
bool symqg_hadamard(float *, size_t) {
  return false;
}
}  // namespace zvec::turbo
int main(int argc, char **argv) try {
  check(argc == 5, "usage: geometry vectors ids moments output");
  Map vectors(argv[1]), ids(argv[2]), moments(argv[3]);
  check(vectors.size == 16 + 1000000ULL * 768 * 4 && ids.size % 16 == 0 &&
            moments.size == ids.size / 16 * 128,
        "input size");
  const float *v = reinterpret_cast<const float *>(vectors.p + 16);
  const double *old = reinterpret_cast<const double *>(moments.p);
  zvec::turbo::SymphonyQGBlockQuantizer codec(768, true);
  using F = zvec::core::QuantizedGraphFusion;
  size_t count = std::min(size_t{4096}, ids.size / 16);
  std::vector<std::array<double, 4>> output;
  output.reserve(count * 2);
  double max_ideal_delta = 0, max_factor_delta = 0, max_half_delta = 0;
  for (size_t i = 0; i < count; ++i) {
    uint32_t target = u32(ids.p + 16 * i);
    check(target < 1000000, "target");
    std::vector<float> x;
    codec.transform(v + size_t(target) * 768, x);
    for (size_t a = 0; a < 2; ++a) {
      uint32_t source = u32(ids.p + 16 * i + 4 * (a + 1));
      check(source < 1000000, "source");
      std::vector<float> center;
      codec.transform(v + size_t(source) * 768, center);
      uint16_t half;
      check(F::error_scale(center.data(), x.data(), 768, half), "scale");
      std::vector<char> codes(codec.batch_bytes());
      codec.encode(center.data(), x.data(), 1, codes.data());
      rabitqlib::ConstQGBatchDataMap<float> batch(codes.data(), 768);
      double scale = -.5 * batch.f_rescale()[0], actual = 0;
      for (size_t j = 0; j < 768; ++j) {
        double r = double(x[j]) - center[j];
        double e = (r > 0 ? scale : -scale) - r;
        actual += e * e;
      }
      double reference = old[i * 16 + a];
      uint16_t expected = F::encode(float(reference));
      check(half == expected, "half vs independent direct reconstruction");
      max_ideal_delta =
          std::max(max_ideal_delta,
                   std::abs(double(F::decode(half)) - F::decode(expected)));
      max_factor_delta =
          std::max(max_factor_delta, std::abs(actual - reference));
      max_half_delta =
          std::max(max_half_delta, std::abs(F::decode(half) - reference));
      output.push_back({double(half), F::decode(half), reference, actual});
    }
  }
  check(max_factor_delta < 1e-5, "native factor reconstruction");
  save(argv[4], output.data(), output.size() * sizeof(output[0]));
  std::printf(
      "FUSION69 GEOMETRY edges=%zu ideal_half_delta=%.17g "
      "native_factor_delta=%.17g half_error=%.17g\n",
      output.size(), max_ideal_delta, max_factor_delta, max_half_delta);
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
