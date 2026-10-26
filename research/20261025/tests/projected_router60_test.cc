#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "algorithm/quantized_graph/quantized_graph_integer_router.h"
#include "turbo/preprocessor/symphony_qg_rotator/symphony_qg_rotator.h"
using Router = zvec::core::QuantizedGraphIntegerRouter;
using Query = zvec::core::IntegerCenter::Query;
struct Graph {
  bool valid(uint32_t id) const {
    return id < 1000000;
  }
};
uint32_t reference(const Router &r, const Query &q, size_t beam) {
  std::vector<uint32_t> current{0};
  std::pair<float, uint32_t> best{INFINITY, UINT32_MAX};
  while (!current.empty()) {
    std::vector<std::pair<float, uint32_t>> next;
    for (auto id : current) {
      const auto &g = r.group(id);
      for (size_t j = 0; j < g.count; ++j) {
        int32_t dot = 0;
        for (size_t d = 0; d < Router::kDim; ++d)
          dot += int32_t(g.codes[j][d]) * q.codes[d];
        float score = -std::fma(
            q.scale * Router::kFactor,
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
  assert(argc == 4);
  std::vector<float> mean(768), rm;
  std::ifstream f(argv[2], std::ios::binary);
  f.read(reinterpret_cast<char *>(mean.data()), 3072);
  assert(f.good());
  zvec::turbo::SymphonyQGRotation rotate(768);
  rotate.rotate(mean.data(), rm);
  Router r;
  assert(r.load(argv[1], mean, Graph{}, 1000000, &rm));
  std::mt19937 rng(60);
  std::normal_distribution<float> normal;
  size_t codechecks = 0, dotchecks = 0, formulachecks = 0, routechecks = 0,
         rotationchecks = 0;
  for (size_t t = 0; t < 500; ++t) {
    std::vector<float> a(768), b(768), ra, rb;
    double an = 0, bn = 0;
    for (size_t j = 0; j < 768; ++j) {
      a[j] = normal(rng);
      b[j] = normal(rng);
      an += double(a[j]) * a[j];
      bn += double(b[j]) * b[j];
    }
    for (size_t j = 0; j < 768; ++j) {
      a[j] /= std::sqrt(an);
      b[j] /= std::sqrt(bn);
    }
    rotate.rotate(a.data(), ra);
    rotate.rotate(b.data(), rb);
    double original = 0, rotated = 0;
    for (size_t j = 0; j < 768; ++j) {
      original += double(a[j]) * b[j];
      rotated += double(ra[j]) * rb[j];
    }
    assert(std::abs(original - rotated) < 2e-7);
    ++rotationchecks;
    Query q;
    assert(r.prepare_projected(ra.data(), q));
    float maximum = 0;
    for (size_t j = 0; j < Router::kDim; ++j)
      maximum = std::max(maximum, std::abs(ra[j] - rm[j]));
    float scale = maximum / 127.f;
    assert(scale == q.scale);
    float inverse = scale > 0 ? 1.f / scale : 0;
    int sum = 0;
    for (size_t j = 0; j < Router::kDim; ++j) {
      int code = int(
          std::clamp(std::nearbyint((ra[j] - rm[j]) * inverse), -127.f, 127.f));
      assert(code == q.codes[j]);
      sum += code;
      ++codechecks;
    }
    assert(sum == q.code_sum);
    const auto &g = r.group(rng() % r.group_count());
    float scores[16];
    int32_t dots[16];
    Router::score(g, q, scores, dots);
    for (size_t j = 0; j < 16; ++j) {
      int32_t ref = 0;
      for (size_t d = 0; d < Router::kDim; ++d)
        ref += int32_t(g.codes[j][d]) * q.codes[d];
      assert(ref == dots[j]);
      ++dotchecks;
      if (j < g.count) {
        double exact =
            -double(g.mean_dot[j]) -
            double(q.scale) * Router::kFactor *
                (double(g.scale[j]) * ref + double(g.offset[j]) * q.code_sum);
        assert(std::abs(scores[j] - exact) <= 1e-6 * (1 + std::abs(exact)));
        ++formulachecks;
      } else
        assert(std::isinf(scores[j]));
    }
    if (t < 100)
      for (size_t beam : {1, 2, 4}) {
        assert(r.choose(q, beam) == reference(r, q, beam));
        ++routechecks;
      }
  }
  Query zero;
  assert(r.prepare_projected(rm.data(), zero));
  assert(zero.scale == 0 && zero.code_sum == 0);
  assert(std::all_of(zero.codes.begin(), zero.codes.end(),
                     [](int8_t x) { return x == 0; }));
  auto bad = rm;
  bad[0] = std::nextafter(bad[0], INFINITY);
  assert(!r.load(argv[1], mean, Graph{}, 1000000, &bad));
  assert(r.empty());
  assert(!r.load(argv[1], mean, Graph{}, 1000000));
  assert(!r.load(argv[3], mean, Graph{}, 1000000, &rm));
  assert(r.load(nullptr, mean, Graph{}, 1000000, &rm) && r.empty());
  std::cout << "dimension=" << Router::kDim << " query_codes=" << codechecks
            << " integer_dots=" << dotchecks
            << " fp64_formulas=" << formulachecks
            << " scalar_routes=" << routechecks
            << " rotation_dots=" << rotationchecks
            << " zero_residual_and_loader_checks=5 passed\n";
}
