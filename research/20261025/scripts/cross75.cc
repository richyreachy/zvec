#define main graph65_program_main
#include "graph65_refine.cc"
#undef main
#include "turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h"
int main(int argc, char **argv) try {
  check(argc == 8,
        "usage: cross75 rotated cross_rotated pairs codes signs mode output");
  Map vectors(argv[1]), queries(argv[2]), pairs(argv[3]), codes(argv[4]),
      signs(argv[5]);
  check(vectors.size == 2048ULL * 65 * 768 * 4 &&
            queries.size == 1024ULL * 16 * 768 * 4 &&
            pairs.size == 1024 * 2 * 4 && codes.size == 2048 * 3328 &&
            signs.size == 2048ULL * 32 * 768,
        "shape");
  std::string mode = argv[6];
  check(mode == "native" || mode == "precision_control" || mode == "coupled" ||
            mode == "independent",
        "mode");
  FILE *out = std::fopen(argv[7], "wbx");
  check(out, "exclusive output");
  zvec::turbo::SymphonyQGBlockQuantizer codec(768, true);
  std::array<float, 32> scores;
  size_t checked = 0;
  for (size_t p = 0; p < 1024; ++p)
    for (size_t b = 0; b < 2; ++b) {
      size_t row = u32(pairs.p + (p * 2 + b) * 4);
      check(row < 2048, "row");
      const float *x =
          reinterpret_cast<const float *>(vectors.p) + row * 65 * 768;
      const char *code = reinterpret_cast<const char *>(codes.p) + row * 3328;
      rabitqlib::ConstQGBatchDataMap<float> batch(code, 768);
      for (size_t j = 0; j < 16; ++j) {
        const float *q =
            reinterpret_cast<const float *>(queries.p) + (p * 16 + j) * 768;
        zvec::turbo::SymphonyQGBlockQuantizer::Query lookup;
        lookup.lookup.reset(q, 768);
        float dot = 0;
        for (size_t k = 0; k < 768; ++k) dot += q[k] * x[k];
        float center = 1 - dot;
        codec.scan(code, lookup, center, scores.data());
        for (size_t i = 0; i < 32; ++i) {
          check(std::isfinite(scores[i]), "finite");
          if (p < 32) {
            uint32_t total = 0;
            for (size_t book = 0; book < 192; ++book) {
              unsigned nibble = 0;
              for (size_t k = 0; k < 4; ++k) {
                size_t dim = 4 * book + k;
                int sign = (mode == "coupled" || mode == "independent")
                               ? reinterpret_cast<const int8_t *>(
                                     signs.p)[row * 32 * 768 + i * 768 + dim]
                               : (x[(i + 1) * 768 + dim] > x[dim] ? 1 : -1);
                if (sign > 0) nibble |= 1u << (3 - k);
              }
              total += lookup.lookup.lut()[book * 16 + nibble];
            }
            float val = std::fma(lookup.lookup.delta(), float(total),
                                 lookup.lookup.sum_vl_lut());
            val += lookup.lookup.k1xsumq();
            float expected =
                std::fma(batch.f_rescale()[i], val, batch.f_add()[i] + center);
            check(std::memcmp(&expected, &scores[i], 4) == 0,
                  "scalar cross LUT/FMA");
            ++checked;
          }
        }
        check(std::fwrite(scores.data(), 4, 32, out) == 32, "write");
      }
    }
  check(std::fclose(out) == 0, "close");
  std::printf(
      "{\"mode\":\"%s\",\"pairs\":1024,\"scores\":1048576,\"scalar_lut_fma_bit_"
      "identical_scores\":%zu,\"not_full_search\":true}\n",
      mode.c_str(), checked);
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
