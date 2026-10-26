#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "turbo/preprocessor/symphony_qg_rotator/symphony_qg_rotator.h"
int main(int argc, char **argv) {
  assert(argc == 4);
  std::filesystem::path input = argv[1], output = argv[2];
  assert(!std::filesystem::exists(output));
  std::filesystem::create_directory(output);
  std::ifstream cf(input / "centers.f32", std::ios::binary),
      tf(input / "targets.i32", std::ios::binary),
      nf(input / "counts.u32", std::ios::binary), mf(argv[3], std::ios::binary);
  assert(cf.good() && tf.good() && nf.good() && mf.good());
  std::array<float, 768> mean{};
  mf.read(reinterpret_cast<char *>(mean.data()), 3072);
  assert(mf.good());
  size_t groups = std::filesystem::file_size(input / "counts.u32") / 4;
  zvec::turbo::SymphonyQGRotation rotate(768);
  std::vector<float> rm;
  rotate.rotate(mean.data(), rm);
  assert(rm.size() == 768);
  std::array<size_t, 3> dims{768, 384, 256};
  std::array<std::ofstream, 3> files;
  for (size_t i = 0; i < 3; ++i) {
    files[i].open(output / ("router" + std::to_string(dims[i]) + ".bin"),
                  std::ios::binary);
    uint32_t h[] = {0x51475259, 2, uint32_t(dims[i]), uint32_t(groups),
                    1000000};
    files[i].write(reinterpret_cast<char *>(h), 20);
    files[i].write(reinterpret_cast<char *>(mean.data()), 3072);
    uint32_t tag = 0x53514701;
    files[i].write(reinterpret_cast<char *>(&tag), 4);
    files[i].write(reinterpret_cast<char *>(rm.data()), 3072);
  }
  for (size_t g = 0; g < groups; ++g) {
    uint32_t count;
    std::array<int32_t, 16> targets;
    std::array<std::array<float, 768>, 16> centers, residuals;
    std::array<float, 16> md{};
    nf.read(reinterpret_cast<char *>(&count), 4);
    tf.read(reinterpret_cast<char *>(targets.data()), 64);
    cf.read(reinterpret_cast<char *>(centers.data()), 16 * 3072);
    assert(nf.good() && tf.good() && cf.good());
    for (size_t j = 0; j < 16; ++j) {
      std::vector<float> rot;
      rotate.rotate(centers[j].data(), rot);
      double dot = 0;
      for (size_t d = 0; d < 768; ++d) {
        residuals[j][d] = rot[d] - rm[d];
        dot += double(mean[d]) * centers[j][d];
      }
      md[j] = float(dot);
    }
    for (size_t i = 0; i < 3; ++i) {
      size_t dim = dims[i];
      std::array<float, 16> scales, offsets;
      std::vector<uint8_t> codes(16 * dim);
      for (size_t j = 0; j < 16; ++j) {
        auto begin = residuals[j].begin();
        auto bounds = std::minmax_element(begin, begin + dim);
        float lo = *bounds.first, scale = (*bounds.second - lo) / 255.f;
        offsets[j] = lo;
        scales[j] = scale;
        float inv = scale > 0 ? 1.f / scale : 0.f;
        for (size_t d = 0; d < dim; ++d)
          codes[j * dim + d] = uint8_t(std::clamp(
              std::nearbyint((residuals[j][d] - lo) * inv), 0.f, 255.f));
      }
      auto &f = files[i];
      f.write(reinterpret_cast<char *>(&count), 4);
      f.write(reinterpret_cast<char *>(targets.data()), 64);
      f.write(reinterpret_cast<char *>(scales.data()), 64);
      f.write(reinterpret_cast<char *>(offsets.data()), 64);
      f.write(reinterpret_cast<char *>(md.data()), 64);
      f.write(reinterpret_cast<char *>(codes.data()), codes.size());
      assert(f.good());
    }
  }
  for (auto &f : files) {
    f.close();
    assert(f.good());
  }
  std::cout << "CONVERT60 COMPLETE groups=" << groups
            << ";fixed tree, no query inputs\n";
}
