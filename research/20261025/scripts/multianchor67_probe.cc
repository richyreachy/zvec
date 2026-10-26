#define main graph65_program_main
#include "graph65_refine.cc"
#undef main
#include "turbo/preprocessor/symphony_qg_rotator/symphony_qg_rotator.h"
// Use the production rotation's scalar fallback for this geometry-only probe.
namespace zvec::turbo {
bool symqg_mul(const float *, float *, size_t) {
  return false;
}
bool symqg_hadamard(float *, size_t) {
  return false;
}
}  // namespace zvec::turbo
int main(int argc, char **argv) try {
  check(argc == 4, "usage: probe vectors ids output_directory");
  Map vectors(argv[1]), ids(argv[2]);
  std::string out = argv[3];
  check(vectors.size == 16 + 1000000ULL * 768 * 4 &&
            u32(vectors.p) == 0x51475664 && ids.size % (7 * 4) == 0,
        "input format");
  const float *v = reinterpret_cast<const float *>(vectors.p + 16);
  size_t count = ids.size / 28;
  zvec::turbo::SymphonyQGRotation rotation(768);
  check(rotation.padded_dim() == 768, "rotation size");
  std::vector<std::array<double, 16>> moments(count);
  FILE *ef = std::fopen((out + "/error_vectors.f64").c_str(), "wbx"),
       *df = std::fopen((out + "/proxy_deltas.f64").c_str(), "wbx");
  check(ef && df, "exclusive files");
  double endpoint_max = 0, identity_max = 0;
  for (size_t i = 0; i < count; ++i) {
    std::array<std::vector<float>, 7> x;
    for (size_t j = 0; j < 7; ++j) {
      U id = u32(ids.p + i * 28 + j * 4);
      check(id < 1000000, "id");
      rotation.rotate(v + size_t(id) * D, x[j]);
    }
    std::array<std::array<double, D>, 2> e{}, h{}, r{};
    auto &m = moments[i];
    m.fill(0);
    for (size_t a = 0; a < 2; ++a) {
      double norm = 0, l1 = 0;
      for (size_t k = 0; k < D; ++k) {
        r[a][k] = double(x[0][k]) - x[a + 1][k];
        norm += r[a][k] * r[a][k];
        l1 += std::abs(r[a][k]);
      }
      double scale = l1 > 0 ? norm / l1 : 0, endpoint = 0;
      for (size_t k = 0; k < D; ++k) {
        h[a][k] = r[a][k] > 0 ? scale : -scale;
        e[a][k] = h[a][k] - r[a][k];
        m[a] += e[a][k] * e[a][k];
        endpoint += r[a][k] * e[a][k];
      }
      endpoint_max = std::max(endpoint_max, std::abs(endpoint));
      m[3 + a] = norm;
    }
    for (size_t k = 0; k < D; ++k) m[2] += e[0][k] * e[1][k];
    check(std::fwrite(e.data(), sizeof(e), 1, ef) == 1, "write errors");
    for (size_t q = 0; q < 4; ++q) {
      std::array<double, D> delta{};
      for (size_t k = 0; k < D; ++k) {
        delta[k] = double(x[q + 3][k]) - x[0][k];
        m[13 + q % 3] += 0;
      }
      check(std::fwrite(delta.data(), sizeof(delta), 1, df) == 1,
            "write delta");
      for (size_t a = 0; a < 2; ++a) {
        double error = 0, direct = 0;
        for (size_t k = 0; k < D; ++k) {
          error -= delta[k] * e[a][k];
          direct -= (double(x[q + 3][k]) - x[a + 1][k]) * e[a][k];
        }
        m[5 + q * 2 + a] = error;
        identity_max = std::max(identity_max, std::abs(error - direct));
      }
    }
  }
  check(std::fclose(ef) == 0 && std::fclose(df) == 0, "close");
  save(out + "/moments.f64", moments.data(),
       moments.size() * sizeof(moments[0]));
  std::printf("PROBE67 targets=%zu endpoint_max=%.17g identity_max=%.17g\n",
              count, endpoint_max, identity_max);
  check(endpoint_max < 1e-12 && identity_max < 1e-12, "identity");
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
