#define main graph65_program_main
#include "graph65_refine.cc"
#undef main
#include "turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h"
#include "coupled73_codec.h"
int main(int argc, char **argv) try {
  check(argc == 6,
        "usage: codec74 rotated signs mode score_output codes_output");
  Map input(argv[1]), learned(argv[2]);
  check(input.size == 2048ULL * 65 * 768 * 4 &&
            learned.size == 2048ULL * 32 * 768,
        "inputs");
  const float *raw = reinterpret_cast<const float *>(input.p);
  std::string mode = argv[3];
  check(mode == "native" || mode == "precision_control" || mode == "learned",
        "mode");
  FILE *out = std::fopen(argv[4], "wbx"), *codeout = std::fopen(argv[5], "wbx");
  check(out && codeout, "exclusive outputs");
  zvec::turbo::SymphonyQGBlockQuantizer codec(768, true);
  std::vector<char> codes(codec.batch_bytes()), native(codec.batch_bytes());
  std::vector<int8_t> signs(32 * 768);
  std::array<float, 32> scores;
  double max_radial = 0, max_factor_diff = 0;
  size_t scalar_checks = 0;
  for (size_t row = 0; row < 2048; ++row) {
    const float *x = raw + row * 65 * 768;
    for (size_t i = 0; i < 32; ++i)
      for (size_t k = 0; k < 768; ++k)
        signs[i * 768 + k] = mode == "learned"
                                 ? reinterpret_cast<const int8_t *>(
                                       learned.p)[row * 32 * 768 + i * 768 + k]
                                 : (x[(i + 1) * 768 + k] > x[k] ? 1 : -1);
    if (mode == "native")
      codec.encode(x, x + 768, 32, codes.data());
    else
      coupled73::encode(x, x + 768, signs.data(), 32, 768, codes.data());
    rabitqlib::ConstQGBatchDataMap<float> batch(codes.data(), 768);
    if (mode == "precision_control") {
      codec.encode(x, x + 768, 32, native.data());
      check(std::memcmp(native.data(), codes.data(), 3072) == 0,
            "original signs same bits");
      rabitqlib::ConstQGBatchDataMap<float> nb(native.data(), 768);
      for (size_t i = 0; i < 32; ++i) {
        max_factor_diff =
            std::max(max_factor_diff,
                     std::abs(double(nb.f_add()[i]) - batch.f_add()[i]));
        max_factor_diff = std::max(
            max_factor_diff,
            std::abs(double(nb.f_rescale()[i]) - batch.f_rescale()[i]));
      }
    }
    for (size_t i = 0; i < 32; ++i) {
      double norm = 0, u = 0;
      for (size_t k = 0; k < 768; ++k) {
        double r = double(x[(i + 1) * 768 + k]) - x[k];
        norm += r * r;
        u += r * signs[i * 768 + k];
      }
      max_radial =
          std::max(max_radial, std::abs(-.5 * batch.f_rescale()[i] * u - norm));
    }
    check(std::fwrite(codes.data(), 1, codes.size(), codeout) == codes.size(),
          "code write");
    for (size_t j = 33; j < 65; ++j) {
      const float *q = x + j * 768;
      zvec::turbo::SymphonyQGBlockQuantizer::Query query;
      query.lookup.reset(q, 768);
      float dot = 0;
      for (size_t k = 0; k < 768; ++k) dot += q[k] * x[k];
      float center = 1 - dot;
      codec.scan(codes.data(), query, center, scores.data());
      for (size_t i = 0; i < 32; ++i) {
        check(std::isfinite(scores[i]), "finite score");
        if (row < 64) {
          uint32_t sum = 0;
          for (size_t book = 0; book < 192; ++book) {
            unsigned nibble = 0;
            for (size_t k = 0; k < 4; ++k)
              if (signs[i * 768 + 4 * book + k] > 0) nibble |= 1u << (3 - k);
            sum += query.lookup.lut()[book * 16 + nibble];
          }
          float lookup = std::fma(query.lookup.delta(), float(sum),
                                  query.lookup.sum_vl_lut());
          lookup += query.lookup.k1xsumq();
          float expected =
              std::fma(batch.f_rescale()[i], lookup, batch.f_add()[i] + center);
          check(std::memcmp(&expected, &scores[i], 4) == 0,
                "scalar LUT/FMA bit identity");
          ++scalar_checks;
        }
      }
      check(std::fwrite(scores.data(), 4, 32, out) == 32, "score write");
    }
  }
  check(std::fclose(out) == 0 && std::fclose(codeout) == 0, "close");
  check(max_radial < 1e-5 && max_factor_diff < 1e-5, "numeric tolerances");
  std::printf(
      "{\"mode\":\"%s\",\"centers\":2048,\"scores\":2097152,\"batch_bytes\":%"
      "zu,\"scalar_lut_fma_bit_identical_scores\":%zu,\"max_radial_error\":%."
      "17g,\"max_native_factor_difference\":%.17g}\n",
      mode.c_str(), codec.batch_bytes(), scalar_checks, max_radial,
      max_factor_diff);
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
