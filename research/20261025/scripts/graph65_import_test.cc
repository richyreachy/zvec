#include <unistd.h>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <string>
#include "core/algorithm/quantized_graph/quantized_graph_import.h"
struct Graph {
  int invalid = -1;
  bool duplicate = false;
  bool valid(uint32_t i) const {
    return i < 40 && int(i) != invalid;
  }
  uint64_t key(uint32_t i) const {
    return duplicate ? 999 : 100000ULL + 17 * i;
  }
};
int main() {
  using zvec::core::QuantizedGraphImport;
  Graph g;
  QuantizedGraphImport m;
  std::string path =
      "/tmp/graph65_import_test_" + std::to_string(getpid()) + ".bin";
  std::vector<char> good(32 + 40 * 140, 0);
  uint32_t h[8] = {0x51474764, 1, 40, 768, 32, 7, 0, 0};
  std::memcpy(good.data(), h, 32);
  for (uint32_t i = 0; i < 40; ++i) {
    uint64_t key = g.key(i);
    uint32_t degree = 32;
    std::memcpy(good.data() + 32 + i * 140, &key, 8);
    std::memcpy(good.data() + 40 + i * 140, &degree, 4);
    for (uint32_t j = 0; j < 32; ++j) {
      uint32_t x = (i + j + 1) % 40;
      std::memcpy(good.data() + 44 + i * 140 + j * 4, &x, 4);
    }
  }
  auto write = [&](const std::vector<char> &bytes) {
    FILE *f = fopen(path.c_str(), "wb");
    assert(f);
    assert(fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size());
    assert(fclose(f) == 0);
  };
  write(good);
  assert(m.load(path.c_str(), g, 40, 768, 32));
  assert(m.entry == 7 && m.rows.size() == 40);
  for (uint32_t i = 0; i < 40; ++i)
    for (uint32_t j = 0; j < 32; ++j) assert(m.rows[i][j] == (i + j + 1) % 40);
  size_t rejections = 0;
  auto reject = [&](std::vector<char> bytes) {
    write(bytes);
    assert(!m.load(path.c_str(), g, 40, 768, 32));
    assert(m.rows.empty() && m.entry == UINT32_MAX);
    ++rejections;
  };
  auto mutate = [&](size_t pos, uint32_t value) {
    auto b = good;
    std::memcpy(b.data() + pos, &value, 4);
    reject(b);
  };
  mutate(0, 0);
  mutate(4, 2);
  mutate(8, 39);
  mutate(12, 767);
  mutate(16, 31);
  mutate(20, 40);
  mutate(24, 1);
  mutate(28, 1);
  mutate(32, 0);
  mutate(40, 31);
  mutate(44, 0);
  mutate(44, 40);
  mutate(48, 1);
  auto b = good;
  b.pop_back();
  reject(b);
  b = good;
  b.push_back(0);
  reject(b);
  reject(std::vector<char>(8));
  write(good);
  g.invalid = 2;
  assert(!m.load(path.c_str(), g, 40, 768, 32));
  ++rejections;
  g.invalid = -1;
  g.duplicate = true;
  b = good;
  for (size_t i = 0; i < 40; ++i) {
    uint64_t key = 999;
    std::memcpy(b.data() + 32 + i * 140, &key, 8);
  }
  reject(b);
  g.duplicate = false;
  write(good);
  assert(!m.load(path.c_str(), g, 40, 768, 64));
  ++rejections;
  assert(!m.load(path.c_str(), g, 40, 767, 32));
  ++rejections;
  assert(!m.load("/nonexistent/graph65", g, 40, 768, 32));
  ++rejections;
  assert(m.load(path.c_str(), g, 40, 768, 32));
  assert(m.load(nullptr, g, 40, 768, 32));
  assert(m.rows.empty());
  assert(std::filesystem::remove(path));
  std::printf(
      "PASS 1280 neighbor checks, %zu rejection assertions, absent path clears "
      "override\n",
      rejections);
}
