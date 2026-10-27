#include <array>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "algorithm/quantized_graph/quantized_graph_edge_export.h"
struct Graph {
  struct Vector {
    const float *p = nullptr;
    const void *data() const {
      return p;
    }
  };
  std::vector<std::array<float, 768>> v =
      std::vector<std::array<float, 768>>(73);
  std::array<std::array<uint32_t, 4>, 73> adj{};
  Graph() {
    for (size_t i = 0; i < 73; ++i) {
      v[i][i] = 1;
      for (size_t j = 0; j < 4; ++j) adj[i][j] = (i + j + 1) % 73;
    }
  }
  bool valid(uint32_t id) const {
    return id < 73;
  }
  uint64_t key(uint32_t id) const {
    return id + 1000;
  }
  int get_vector(uint32_t id, Vector &out) const {
    out.p = v[id].data();
    return 0;
  }
};
int main(int argc, char **argv) {
  assert(argc == 2);
  std::filesystem::path p = argv[1];
  assert(!std::filesystem::exists(p));
  Graph g;
  auto neighbors = [&](uint32_t id) {
    return std::make_pair(g.adj[id].data(), size_t{4});
  };
  using Export = zvec::core::QuantizedGraphEdgeExport;
  assert(Export::write(g, 73, 768, p.c_str(), neighbors));
  auto bytes = std::filesystem::file_size(p);
  assert(!Export::write(g, 73, 768, p.c_str(), neighbors));
  assert(std::filesystem::file_size(p) == bytes);
  std::ifstream f(p, std::ios::binary);
  uint32_t h[6];
  f.read(reinterpret_cast<char *>(h), 24);
  assert(h[0] == 0x51474561 && h[1] == 1 && h[2] == 73 && h[3] == 768 &&
         h[4] == 73 && h[5] == 3);
  size_t checks = 0;
  for (size_t i = 0; i < 73; ++i) {
    std::array<uint32_t, 4> ids;
    std::array<uint64_t, 4> keys;
    uint32_t degree;
    std::array<uint32_t, 3> offsets;
    std::array<std::array<float, 768>, 4> vectors;
    f.read(reinterpret_cast<char *>(ids.data()), 16);
    f.read(reinterpret_cast<char *>(keys.data()), 32);
    f.read(reinterpret_cast<char *>(&degree), 4);
    f.read(reinterpret_cast<char *>(offsets.data()), 12);
    f.read(reinterpret_cast<char *>(vectors.data()), 4 * 768 * 4);
    assert(f.good());
    assert(ids[0] == (uint64_t(i) * 2654435761ULL + 7193) % 73);
    assert(degree == 4);
    for (size_t j = 0; j < 4; ++j) {
      assert(keys[j] == ids[j] + 1000);
      for (size_t k = 0; k < 768; ++k)
        assert(vectors[j][k] == (k == ids[j] ? 1.f : 0.f));
      ++checks;
    }
    for (size_t j = 0; j < 3; ++j) {
      assert(offsets[j] < 4);
      assert(ids[j + 1] == g.adj[ids[0]][offsets[j]]);
      ++checks;
    }
  }
  assert(f.peek() == EOF);
  f.close();
  std::filesystem::remove(p);
  assert(!Export::write(g, 73, 384, p.c_str(), neighbors));
  assert(!std::filesystem::exists(p));
  assert(!Export::write(g, 0, 768, p.c_str(), neighbors));
  auto short_list = [&](uint32_t id) {
    return std::make_pair(g.adj[id].data(), size_t{2});
  };
  assert(!Export::write(g, 73, 768, p.c_str(), short_list));
  assert(!std::filesystem::exists(p));
  g.v[0][0] = NAN;
  assert(!Export::write(g, 73, 768, p.c_str(), neighbors));
  std::filesystem::remove(p);
  std::cout << "edge-export identity/vector/actual-neighbor checks=" << checks
            << " plus exclusive-file,dimension,empty,degree,nonfinite checks "
               "passed\n";
}
