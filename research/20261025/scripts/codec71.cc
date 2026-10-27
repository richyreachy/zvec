#define main graph65_program_main
#include "graph65_refine.cc"
#undef main
#include <fstream>
#include "turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h"
#include "rotation61_kernel.h"
int main(int argc, char **argv) try {
  check(argc == 5,
        "usage: codec71 rotated coefficients_or_dash rows.u32 output");
  Map input(argv[1]), rows(argv[3]);
  check(input.size == 2048ULL * 65 * 768 * 4 && rows.size % 4 == 0, "inputs");
  const float *raw = reinterpret_cast<const float *>(input.p);
  Rotation61 rotation{};
  bool apply = std::string(argv[2]) != "-";
  if (apply) {
    std::ifstream f(argv[2], std::ios::binary);
    f.read(reinterpret_cast<char *>(&rotation), sizeof(rotation));
    check(f.good() && f.peek() == EOF, "coefficients");
  }
  zvec::turbo::SymphonyQGBlockQuantizer codec(768, true);
  std::vector<float> x(65 * 768);
  std::vector<char> codes(codec.batch_bytes());
  std::array<float, 32> scores;
  std::vector<float> output;
  output.reserve(rows.size / 4 * 32 * 32);
  double max_scale_error = 0, max_add_error = 0;
  for (size_t i = 0; i < rows.size / 4; ++i) {
    auto row = u32(rows.p + 4 * i);
    check(row < 2048, "row");
    std::copy_n(raw + size_t(row) * 65 * 768, x.size(), x.data());
    if (apply)
      for (size_t j = 0; j < 65; ++j) rotation.apply(x.data() + j * 768);
    codec.encode(x.data(), x.data() + 768, 32, codes.data());
    rabitqlib::ConstQGBatchDataMap<float> batch(codes.data(), 768);
    double cn = 0;
    for (size_t j = 0; j < 768; ++j) cn += double(x[j]) * x[j];
    for (size_t lane = 0; lane < 32; ++lane) {
      const float *v = x.data() + (lane + 1) * 768;
      double norm = 0, l1 = 0, cs = 0, vn = 0;
      for (size_t j = 0; j < 768; ++j) {
        double r = double(v[j]) - x[j];
        norm += r * r;
        l1 += std::abs(r);
        cs += x[j] * (r > 0 ? 1. : -1.);
        vn += double(v[j]) * v[j];
      }
      double scale = l1 ? norm / l1 : 0;
      max_scale_error = std::max(max_scale_error,
                                 std::abs(batch.f_rescale()[lane] + 2 * scale));
      double add = .5 * norm + scale * cs + .5 * (cn - vn);
      max_add_error =
          std::max(max_add_error, std::abs(batch.f_add()[lane] - add));
    }
    for (size_t j = 33; j < 65; ++j) {
      const float *q = x.data() + j * 768;
      zvec::turbo::SymphonyQGBlockQuantizer::Query query;
      query.lookup.reset(q, 768);
      float dot = 0;
      for (size_t k = 0; k < 768; ++k) dot += q[k] * x[k];
      codec.scan(codes.data(), query, 1 - dot, scores.data());
      for (float score : scores) check(std::isfinite(score), "score");
      output.insert(output.end(), scores.begin(), scores.end());
    }
  }
  check(max_scale_error < 1e-5 && max_add_error < 1e-5, "factor agreement");
  save(argv[4], output.data(), output.size() * 4);
  std::printf(
      "CODEC71 COMPLETE centers=%zu scores=%zu max_scale_error=%.17g "
      "max_add_error=%.17g\n",
      rows.size / 4, output.size(), max_scale_error, max_add_error);
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
