#define main graph65_program_main
#include "graph65_refine.cc"
#undef main
#include "turbo/preprocessor/symphony_qg_rotator/symphony_qg_rotator.h"
namespace zvec::turbo {
bool symqg_mul(const float *, float *, size_t) {
  return false;
}
bool symqg_hadamard(float *, size_t) {
  return false;
}
}  // namespace zvec::turbo
int main(int argc, char **argv) try {
  check(argc == 4, "usage: rotate71 vectors ids output");
  Map v(argv[1]), ids(argv[2]);
  check(v.size == 16 + 1000000ULL * 768 * 4 && u32(v.p) == 0x51475664 &&
            ids.size % (65 * 4) == 0,
        "inputs");
  auto raw = reinterpret_cast<const float *>(v.p + 16);
  zvec::turbo::SymphonyQGRotation rotation(768);
  FILE *out = std::fopen(argv[3], "wbx");
  check(out, "exclusive output");
  std::vector<float> row;
  size_t count = ids.size / 4;
  double max_norm_error = 0;
  for (size_t i = 0; i < count; ++i) {
    auto id = u32(ids.p + 4 * i);
    check(id < 1000000, "id");
    rotation.rotate(raw + size_t(id) * 768, row);
    double before = 0, after = 0;
    for (size_t j = 0; j < 768; ++j) {
      check(std::isfinite(row[j]), "finite");
      before += double(raw[size_t(id) * 768 + j]) * raw[size_t(id) * 768 + j];
      after += double(row[j]) * row[j];
    }
    max_norm_error = std::max(max_norm_error, std::abs(after - before));
    check(std::fwrite(row.data(), 4, 768, out) == 768, "write");
  }
  check(std::fclose(out) == 0 && max_norm_error < 2e-6, "close/norm");
  std::printf("ROTATE71 COMPLETE vectors=%zu max_norm2_error=%.17g\n", count,
              max_norm_error);
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
