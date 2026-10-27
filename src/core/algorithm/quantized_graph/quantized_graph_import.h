#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <vector>

namespace zvec::core {
// Research-only immutable adjacency override; file layout matches graph export.
// An absent path preserves the original construction. Only full 32-edge rows
// are accepted, so subsequent block pruning cannot change imported neighbors.
struct QuantizedGraphImport {
  std::vector<std::array<uint32_t, 32>> rows;
  uint32_t entry{UINT32_MAX};
  void clear() {
    rows.clear();
    entry = UINT32_MAX;
  }
  template <class Graph>
  bool load(const char *path, const Graph &entity, size_t docs, size_t dim,
            size_t degree) {
    clear();
    if (!path) return true;
    if (!docs || docs > UINT32_MAX || dim != 768 || degree != 32) return false;
    FILE *file = std::fopen(path, "rb");
    if (!file) return false;
    uint32_t h[8]{};
    bool ok = std::fread(h, sizeof(h), 1, file) == 1 && h[0] == 0x51474764 &&
              h[1] == 1 && h[2] == docs && h[3] == dim && h[4] == degree &&
              h[5] < docs && h[6] == 0 && h[7] == 0;
    // Validate length before allocating any adjacency storage.
    if (ok) {
      ok = std::fseek(file, 0, SEEK_END) == 0 &&
           std::ftell(file) == static_cast<long>(32 + docs * 140) &&
           std::fseek(file, 32, SEEK_SET) == 0;
    }
    std::vector<std::array<uint32_t, 32>> loaded;
    if (ok) loaded.resize(docs);
    std::unordered_set<uint64_t> keys;
    for (size_t i = 0; i < docs && ok; ++i) {
      uint64_t key = 0;
      uint32_t count = 0;
      auto &row = loaded[i];
      ok = std::fread(&key, sizeof(key), 1, file) == 1 &&
           std::fread(&count, sizeof(count), 1, file) == 1 &&
           std::fread(row.data(), sizeof(row), 1, file) == 1 &&
           entity.valid(uint32_t(i)) && key == entity.key(uint32_t(i)) &&
           keys.insert(key).second && count == 32;
      for (size_t j = 0; j < 32 && ok; ++j)
        ok = row[j] < docs && row[j] != i && entity.valid(row[j]) &&
             std::find(row.begin(), row.begin() + j, row[j]) == row.begin() + j;
    }
    ok = ok && std::fgetc(file) == EOF && !std::ferror(file);
    const int rc = std::fclose(file);
    if (!ok || rc != 0) return false;
    rows.swap(loaded);
    entry = h[5];
    return true;
  }
};
}  // namespace zvec::core
