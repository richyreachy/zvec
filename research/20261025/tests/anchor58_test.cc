#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <random>
#include "algorithm/quantized_graph/quantized_graph_router.h"
struct Graph {
  struct Vector {
    const float *p = nullptr;
    const void *data() const {
      return p;
    }
  };
  std::vector<std::vector<float>> points;
  bool valid(uint32_t id) const {
    return id < points.size();
  }
  int get_vector(uint32_t id, Vector &v) const {
    v.p = points.at(id).data();
    return 0;
  }
};
struct Codec {
  size_t dim;
  void transform(const void *p, std::vector<float> &v) const {
    const float *x = static_cast<const float *>(p);
    v.assign(x, x + dim);
  }
};
struct Context {
  std::vector<float> q;
  size_t calls = 0;
  float distance(const void *p) {
    ++calls;
    const float *x = static_cast<const float *>(p);
    double s = 0;
    for (size_t j = 0; j < q.size(); ++j) {
      double r = double(q[j]) - x[j];
      s += r * r;
    }
    return float(s);
  }
};
int main() {
  std::mt19937 rng(58026);
  std::normal_distribution<float> normal;
  size_t checks = 0;
  for (size_t count : {size_t(73), size_t(4096)}) {
    constexpr size_t dim = 33;
    Graph graph;
    graph.points.assign(count, std::vector<float>(dim));
    std::vector<double> mean(dim, 0);
    for (auto &p : graph.points)
      for (size_t j = 0; j < dim; ++j) {
        p[j] = normal(rng);
        mean[j] += p[j] / double(count);
      }
    zvec::core::QuantizedGraphRouter router;
    assert(router.build(graph, Codec{dim}, mean, count, dim) == 0);
    assert(router.anchor_count() == count);
    for (size_t t = 0; t < 100; ++t) {
      Context ctx;
      ctx.q.resize(dim);
      for (float &x : ctx.q) x = normal(rng);
      uint32_t entry = t % count;
      float original = ctx.distance(graph.points[entry].data());
      float retained = -1;
      std::vector<float> scores;
      const size_t before = ctx.calls;
      assert(router.choose(ctx, graph, ctx.q.data(), entry, dim, 0, 4, scores,
                           &retained) == 0);
      assert(std::isnan(retained) && entry == t % count && ctx.calls == before);
      ++checks;
      assert(router.choose(ctx, graph, ctx.q.data(), entry, dim, count, 4,
                           scores, &retained) == 0);
      assert(retained == original);
      assert(ctx.calls == before + 5);
      ++checks;
      float best = original;
      for (const auto &p : graph.points)
        best = std::min(best, ctx.distance(p.data()));
      assert(ctx.distance(graph.points[entry].data()) <=
             best + 1e-4f * (1 + best));
      ++checks;
    }
  }
  std::cout
      << "expanded-router "
         "full-dimensional/retained-original-distance/disabled-route checks="
      << checks << " passed\n";
}
