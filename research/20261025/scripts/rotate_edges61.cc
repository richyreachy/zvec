#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "turbo/preprocessor/symphony_qg_rotator/symphony_qg_rotator.h"
int main(int argc, char **argv) {
  assert(argc == 3);
  std::ifstream f(argv[1], std::ios::binary);
  uint32_t h[6];
  f.read(reinterpret_cast<char *>(h), 24);
  assert(f.good() && h[0] == 0x51474561 && h[1] == 1 && h[3] == 768 &&
         h[5] == 3);
  std::filesystem::path out = argv[2];
  assert(!std::filesystem::exists(out / "residuals.f32"));
  std::ofstream rfile(out / "residuals.f32", std::ios::binary),
      gfile(out / "local_gaps.f64", std::ios::binary);
  zvec::turbo::SymphonyQGRotation rotation(768);
  for (size_t i = 0; i < h[2]; ++i) {
    std::array<char, 64> metadata;
    std::array<std::array<float, 768>, 4> raw;
    f.read(metadata.data(), 64);
    f.read(reinterpret_cast<char *>(raw.data()), 4 * 768 * 4);
    assert(f.good());
    std::vector<float> center;
    rotation.rotate(raw[0].data(), center);
    for (size_t j = 1; j < 4; ++j) {
      std::vector<float> v;
      rotation.rotate(raw[j].data(), v);
      for (size_t k = 0; k < 768; ++k) v[k] -= center[k];
      rfile.write(reinterpret_cast<char *>(v.data()), 768 * 4);
    }
    double gap = 0;
    for (size_t k = 0; k < 768; ++k)
      gap += double(raw[3][k]) * (double(raw[2][k]) - raw[1][k]);
    gfile.write(reinterpret_cast<char *>(&gap), 8);
  }
  assert(f.peek() == EOF);
  rfile.close();
  gfile.close();
  assert(rfile.good() && gfile.good());
  std::cout << "ROTATE61 COMPLETE samples=" << h[2] << "\n";
}
