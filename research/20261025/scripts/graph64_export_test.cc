#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "algorithm/quantized_graph/quantized_graph_export.h"
struct Graph {
  struct Vector {
    const float *p = nullptr;
    const void *data() const {
      return p;
    }
  };
  std::array<std::array<float, 768>, 7> values{};
  std::array<std::array<uint32_t, 4>, 7> adj{};
  int bad = -1, duplicate_key = -1, null_id = -1;
  Graph() {
    for (size_t i = 0; i < 7; ++i) {
      values[i][i] = 1;
      for (size_t j = 0; j < 4; ++j) adj[i][j] = (i + j + 1) % 7;
    }
  }
  bool valid(uint32_t id) const {
    return id < 7 && int(id) != bad;
  }
  uint64_t key(uint32_t id) const {
    return int(id) == duplicate_key ? 0 : (id * 3) % 7;
  }
  int get_vector(uint32_t id, Vector &v) const {
    v.p = int(id) == null_id ? nullptr : values[id].data();
    return 0;
  }
};
int main(int argc, char **argv) {
  assert(argc == 2);
  namespace fs = std::filesystem;
  fs::path root = argv[1];
  assert(fs::create_directory(root));
  auto gp = root / "graph.bin", vp = root / "vectors.f32";
  using Export = zvec::core::QuantizedGraphExport;
  Graph g;
  auto ns = [&](uint32_t id) {
    return std::make_pair(g.adj[id].data(), size_t{4});
  };
  auto write = [&]() {
    return Export::write(g, 7, 768, 2, gp.c_str(), vp.c_str(), ns);
  };
  auto clean = [&]() {
    fs::remove(gp);
    fs::remove(vp);
  };
  assert(write());
  assert(fs::file_size(gp) == 32 + 7 * 140);
  assert(fs::file_size(vp) == 16 + 7 * 768 * 4);
  assert(!write());
  std::ifstream f(gp, std::ios::binary), v(vp, std::ios::binary);
  uint32_t gh[8], vh[4];
  f.read(reinterpret_cast<char *>(gh), 32);
  v.read(reinterpret_cast<char *>(vh), 16);
  assert(gh[0] == 0x51474764 && gh[1] == 1 && gh[2] == 7 && gh[3] == 768 &&
         gh[4] == 32 && gh[5] == 2);
  assert(vh[0] == 0x51475664 && vh[1] == 1 && vh[2] == 7 && vh[3] == 768);
  size_t checks = 0;
  for (size_t i = 0; i < 7; ++i) {
    uint64_t key;
    uint32_t degree, adj[32];
    std::array<float, 768> x;
    f.read(reinterpret_cast<char *>(&key), 8);
    f.read(reinterpret_cast<char *>(&degree), 4);
    f.read(reinterpret_cast<char *>(adj), 128);
    v.read(reinterpret_cast<char *>(x.data()), 3072);
    assert(f.good() && v.good());
    assert(key == g.key(i) && degree == 4);
    for (size_t j = 0; j < 32; ++j)
      assert(adj[j] == (j < 4 ? g.adj[i][j] : UINT32_MAX));
    for (size_t j = 0; j < 768; ++j) {
      assert(x[j] == g.values[i][j]);
      ++checks;
    }
  }
  assert(f.peek() == EOF && v.peek() == EOF);
  f.close();
  v.close();
  clean();
  assert(!Export::write(g, 0, 768, 0, gp.c_str(), vp.c_str(), ns));
  assert(!Export::write(g, 7, 384, 0, gp.c_str(), vp.c_str(), ns));
  assert(!Export::write(g, 7, 768, 7, gp.c_str(), vp.c_str(), ns));
  assert(!fs::exists(gp));
  {
    std::ofstream keep(vp);
    keep << "preserve";
  }
  assert(!write());
  assert(fs::file_size(vp) == 8);
  clean();
  g.adj[0][1] = g.adj[0][0];
  assert(!write());
  clean();
  g = Graph{};
  g.adj[0][0] = 0;
  assert(!write());
  clean();
  g = Graph{};
  g.adj[0][0] = 99;
  assert(!write());
  clean();
  g = Graph{};
  g.duplicate_key = 1;
  assert(!write());
  clean();
  g = Graph{};
  g.null_id = 1;
  assert(!write());
  clean();
  g = Graph{};
  g.bad = 1;
  assert(!write());
  clean();
  g = Graph{};
  g.values[1][0] = NAN;
  assert(!write());
  clean();
  g = Graph{};
  g.values[1][1] = 2;
  assert(!write());
  clean();
  g = Graph{};
  auto too_many = [&](uint32_t id) {
    return std::make_pair(g.adj[id].data(), size_t{33});
  };
  assert(!Export::write(g, 7, 768, 2, gp.c_str(), vp.c_str(), too_many));
  clean();
  assert(fs::remove(root));
  std::cout << "graph64 export vector checks=" << checks
            << " keys/degrees/edges/padding/headers passed;14 rejection cases "
               "passed\n";
}
