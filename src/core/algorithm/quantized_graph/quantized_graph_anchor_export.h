#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <unordered_set>
#include <vector>

namespace zvec::core {
// Research-only little-endian sample export; called before any query. Records
// are internal ID, external key, then normalized float32 vector coordinates.
struct QuantizedGraphAnchorExport {
  template <class Graph>
  static bool write(const Graph &entity, size_t count, size_t dim,
                    const char *path) {
    if (!path || dim != 768 || count > UINT32_MAX) return false;
    std::vector<uint32_t> ids;
    std::unordered_set<uint32_t> seen;
    for (size_t i = 0; i < std::min(size_t{65536}, count); ++i) {
      const uint32_t id =
          uint32_t((uint64_t(i) * 2654435761ULL + 7193) % count);
      if (entity.valid(id) && seen.insert(id).second) ids.push_back(id);
    }
    if (ids.empty()) return false;
    FILE *f = std::fopen(path, "wbx");
    if (!f) return false;
    const uint32_t header[] = {0x51474159, 1, uint32_t(ids.size()),
                               uint32_t(dim), uint32_t(count)};
    bool ok = std::fwrite(header, sizeof(header), 1, f) == 1;
    for (uint32_t id : ids) {
      if (!ok) break;
      typename Graph::Vector raw;
      if (entity.get_vector(id, raw) || !raw.data()) {
        ok = false;
        break;
      }
      const float *v = static_cast<const float *>(raw.data());
      double norm = 0;
      for (size_t j = 0; j < dim; ++j) norm += double(v[j]) * v[j];
      if (!std::isfinite(norm) || std::abs(norm - 1.) > 1e-4) {
        ok = false;
        break;
      }
      const uint64_t key = entity.key(id);
      ok = std::fwrite(&id, sizeof(id), 1, f) == 1 &&
           std::fwrite(&key, sizeof(key), 1, f) == 1 &&
           std::fwrite(v, sizeof(float), dim, f) == dim;
    }
    ok = !std::ferror(f) && ok;
    return std::fclose(f) == 0 && ok;
  }
};
}  // namespace zvec::core
