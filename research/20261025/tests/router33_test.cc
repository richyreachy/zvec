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
  float distance(const void *p) {
    const float *x = static_cast<const float *>(p);
    double s = 0;
    for (size_t j = 0; j < q.size(); ++j) {
      double z = double(q[j]) - x[j];
      s += z * z;
    }
    return float(s);
  }
};
int main() {
  std::mt19937 gen(33);
  std::normal_distribution<float> normal;
  size_t checks = 0;
  for (size_t dim : {1, 15, 16, 65, 768}) {
    Graph g;
    g.points.assign(73, std::vector<float>(dim));
    std::vector<double> centroid(dim);
    for (auto &p : g.points)
      for (size_t j = 0; j < dim; ++j) {
        p[j] = normal(gen);
        centroid[j] += p[j] / 73.;
      }
    zvec::core::QuantizedGraphRouter router;
    assert(router.build(g, Codec{dim}, centroid, 73, dim) == 0);
    for (size_t t = 0; t < 100; ++t) {
      Context c;
      c.q.resize(dim);
      for (auto &x : c.q) x = normal(gen);
      std::vector<float> scores;
      for (size_t count : {5, 16, 73})
        for (size_t projected : {size_t{1}, dim})
          for (size_t shortlist : {1, 4, 16}) {
            uint32_t entry = uint32_t(t % 73);
            float original = c.distance(g.points[entry].data());
            assert(router.choose(c, g, c.q.data(), entry, projected, count,
                                 shortlist, scores) == 0);
            assert(c.distance(g.points[entry].data()) <= original);
            if (count == 73 && projected == dim) {
              float best = original;
              for (const auto &p : g.points)
                best = std::min(best, c.distance(p.data()));
              assert(c.distance(g.points[entry].data()) <=
                     best + 1e-4f * (1 + best));
            }
            ++checks;
          }
    }
  }
  std::cout << "router seed-distance and full-projection checks=" << checks
            << " passed\n";
}
