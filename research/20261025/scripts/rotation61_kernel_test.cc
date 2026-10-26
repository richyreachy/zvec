#include "rotation61_kernel.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
void scalar(float *v, const Rotation61 &r) {
  size_t strides[] = {1, 16, 384};
  for (size_t layer = 0; layer < 3; ++layer) {
    size_t stride = strides[layer], j = 0;
    for (size_t i = 0; i < 768; i += 2 * stride)
      for (size_t k = 0; k < stride; ++k, ++j) {
        float a = v[i + k], b = v[i + k + stride], c = r.layers[layer].c[j],
              s = r.layers[layer].s[j];
        v[i + k] = std::fma(a, c, -b * s);
        v[i + k + stride] = std::fma(b, c, a * s);
      }
  }
}
int main(int argc, char **argv) {
  assert(argc == 2);
  Rotation61 r{}, identity{};
  std::ifstream f(argv[1], std::ios::binary);
  f.read(reinterpret_cast<char *>(&r), sizeof(r));
  assert(f.good() && f.peek() == EOF);
  for (auto &layer : identity.layers)
    std::fill(std::begin(layer.c), std::end(layer.c), 1.f);
  std::mt19937 rng(61);
  std::normal_distribution<float> normal;
  std::vector<std::array<float, 768>> input(1000);
  double maxnorm = 0;
  size_t coords = 0;
  for (auto &v : input) {
    double before = 0;
    for (float &x : v) {
      x = normal(rng);
      before += double(x) * x;
    }
    auto a = v, b = v, c = v;
    r.apply(a.data());
    scalar(b.data(), r);
    identity.apply(c.data());
    assert(std::memcmp(a.data(), b.data(), sizeof(a)) == 0);
    assert(std::memcmp(c.data(), v.data(), sizeof(c)) == 0);
    double after = 0;
    for (float x : a) after += double(x) * x;
    maxnorm = std::max(maxnorm, std::abs(after / before - 1));
    assert(std::abs(after / before - 1) < 2e-6);
    coords += 768;
  }
  std::cout
      << "scalar_coordinate_checks=" << coords
      << " identity_vectors=1000 norm_vectors=1000 max_relative_norm2_error="
      << maxnorm << " passed\n";
  std::cout << "round,copy_ns,copy_and_transform_ns,incremental_ns,checksum\n";
  double checksum = 0;
  for (int round = 0; round < 3; ++round) {
    double ns[2];
    for (int order = 0; order < 2; ++order) {
      int transformed = (order + round) % 2;
      alignas(64) float buffer[768];
      auto start = std::chrono::steady_clock::now();
      for (size_t repeat = 0; repeat < 100; ++repeat)
        for (const auto &v : input) {
          std::memcpy(buffer, v.data(), sizeof(buffer));
          asm volatile("" : "+m"(buffer));
          if (transformed) r.apply(buffer);
          asm volatile("" : "+m"(buffer));
          checksum += buffer[0];
        }
      auto stop = std::chrono::steady_clock::now();
      ns[transformed] =
          std::chrono::duration<double, std::nano>(stop - start).count() /
          100000;
    }
    std::cout << round << ',' << ns[0] << ',' << ns[1] << ',' << ns[1] - ns[0]
              << ',' << checksum << '\n';
  }
}
