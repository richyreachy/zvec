#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "algorithm/quantized_graph/quantized_graph_integer_router.h"
using zvec::core::IntegerCenter;
using Router = zvec::core::QuantizedGraphIntegerRouter;
struct Graph {
  bool valid(uint32_t id) const {
    return id < 1000000;
  }
};
void write_model(const std::string &path, const std::vector<float> &mean,
                 const std::vector<Router::Group> &groups) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  uint32_t h[] = {0x51475259, 1, 768, uint32_t(groups.size()), 1000000};
  f.write(reinterpret_cast<char *>(h), sizeof(h));
  f.write(reinterpret_cast<const char *>(mean.data()), 768 * 4);
  for (const auto &g : groups) {
    f.write(reinterpret_cast<const char *>(&g.count), 4);
    f.write(reinterpret_cast<const char *>(g.target), 64);
    f.write(reinterpret_cast<const char *>(g.scale), 64);
    f.write(reinterpret_cast<const char *>(g.offset), 64);
    f.write(reinterpret_cast<const char *>(g.mean_dot), 64);
    f.write(reinterpret_cast<const char *>(g.codes), 16 * 768);
  }
}
uint32_t reference(const Router &r, const IntegerCenter::Query &q,
                   size_t beam) {
  std::vector<uint32_t> current{0};
  std::pair<float, uint32_t> best{INFINITY, UINT32_MAX};
  while (!current.empty()) {
    std::vector<std::pair<float, uint32_t>> next;
    for (auto id : current) {
      const auto &g = r.group(id);
      for (size_t j = 0; j < g.count; ++j) {
        int32_t dot = 0;
        for (size_t d = 0; d < 768; ++d)
          dot += int32_t(g.codes[j][d]) * q.codes[d];
        float score = -std::fma(
            q.scale,
            std::fma(g.scale[j], float(dot), g.offset[j] * float(q.code_sum)),
            g.mean_dot[j]);
        if (g.target[j] < 0)
          best = std::min(best, {score, uint32_t(-int64_t(g.target[j]) - 1)});
        else
          next.push_back({score, uint32_t(g.target[j])});
      }
    }
    std::sort(next.begin(), next.end());
    current.clear();
    for (size_t j = 0; j < std::min(beam, next.size()); ++j)
      current.push_back(next[j].second);
  }
  return best.second;
}
int main(int argc, char **argv) {
  assert(argc == 3);
  std::string model = argv[1], tmp = argv[2];
  assert(!std::filesystem::exists(tmp));
  std::mt19937 rng(59);
  std::normal_distribution<float> normal;
  std::vector<float> mean(768);
  IntegerCenter::Query q;
  q.codes.resize(768);
  q.scale = .001f;
  size_t integer_checks = 0, formula_checks = 0;
  for (size_t trial = 0; trial < 1000; ++trial) {
    Router::Group g;
    g.count = uint32_t(1 + trial % 16);
    q.code_sum = 0;
    for (auto &v : q.codes) {
      v = int8_t(int(rng() % 255) - 127);
      q.code_sum += v;
    }
    for (size_t j = 0; j < 16; ++j) {
      g.scale[j] = .0001f + float(rng() % 1000) * 1e-6f;
      g.offset[j] = normal(rng) * .01f;
      g.mean_dot[j] = normal(rng) * .1f;
      for (auto &v : g.codes[j]) v = uint8_t(rng());
    }
    float score[16];
    int32_t dots[16];
    Router::score(g, q, score, dots);
    for (size_t j = 0; j < 16; ++j) {
      int32_t ref = 0;
      for (size_t d = 0; d < 768; ++d)
        ref += int32_t(g.codes[j][d]) * q.codes[d];
      assert(ref == dots[j]);
      ++integer_checks;
      if (j < g.count) {
        double fp64 = -double(g.mean_dot[j]) -
                      double(q.scale) * (double(g.scale[j]) * ref +
                                         double(g.offset[j]) * q.code_sum);
        assert(std::abs(score[j] - fp64) <= 1e-6 * (1 + std::abs(fp64)));
        ++formula_checks;
      } else
        assert(std::isinf(score[j]) && score[j] > 0);
    }
  }
  // Known two-level tree and loader failures; every failure must clear state.
  std::vector<Router::Group> gs(2);
  gs[0].count = 2;
  gs[0].target[0] = 1;
  gs[0].target[1] = -1;
  gs[0].mean_dot[1] = -2;
  gs[1].count = 2;
  gs[1].target[0] = -2;
  gs[1].target[1] = -3;
  gs[1].mean_dot[0] = 1;
  gs[1].mean_dot[1] = 2;
  Router router;
  write_model(tmp, mean, gs);
  assert(router.load(tmp.c_str(), mean, Graph{}, 1000000));
  assert(router.choose(q, 1) == 2);
  assert(router.choose(q, 4) == 2);
  size_t loader_checks = 2;
  for (int kind = 0; kind < 7; ++kind) {
    auto bad = gs;
    if (kind == 0) bad[0].target[0] = 0;
    if (kind == 1) bad[0].target[0] = 2;
    if (kind == 2) bad[0].target[1] = 1;
    if (kind == 3) bad[0].count = 17;
    if (kind == 4) bad[1].scale[0] = INFINITY;
    if (kind == 5) bad[1].target[0] = -1000001;
    if (kind == 6) bad[1].target[0] = INT32_MIN;
    write_model(tmp, mean, bad);
    assert(!router.load(tmp.c_str(), mean, Graph{}, 1000000));
    assert(router.empty());
    ++loader_checks;
  }
  write_model(tmp, mean, gs);
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::app);
    f.put('x');
  }
  assert(!router.load(tmp.c_str(), mean, Graph{}, 1000000));
  ++loader_checks;
  write_model(tmp, mean, gs);
  std::filesystem::resize_file(tmp, 4000);
  assert(!router.load(tmp.c_str(), mean, Graph{}, 1000000));
  ++loader_checks;
  write_model(tmp, mean, gs);
  mean[0] = .1f;
  assert(!router.load(tmp.c_str(), mean, Graph{}, 1000000));
  ++loader_checks;
  assert(router.load(nullptr, mean, Graph{}, 1000000) && router.empty());
  ++loader_checks;
  std::ifstream f(model, std::ios::binary);
  assert(f.good());
  f.seekg(20);
  f.read(reinterpret_cast<char *>(mean.data()), 768 * 4);
  assert(router.load(model.c_str(), mean, Graph{}, 1000000));
  size_t route_checks = 0;
  for (size_t t = 0; t < 250; ++t) {
    std::vector<float> v(768);
    double norm = 0;
    for (auto &x : v) {
      x = normal(rng);
      norm += double(x) * x;
    }
    for (auto &x : v) x /= std::sqrt(norm);
    assert(IntegerCenter::prepare(v.data(), mean.data(), 768, q));
    for (size_t b : {1, 2, 4}) {
      assert(router.choose(q, b) == reference(router, q, b));
      ++route_checks;
    }
  }
  std::filesystem::remove(tmp);
  std::cout << "integer_products=" << integer_checks
            << " fp64_formulas=" << formula_checks
            << " loader_checks=" << loader_checks
            << " actual_model_scalar_route_checks=" << route_checks
            << " passed\n";
}
