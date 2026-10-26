#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <unordered_set>
#include <vector>

namespace zvec::core {
// Database-only diagnostic. Samples the final graph, not the source HNSW
// neighbors. Three distinct neighbors permit independent local comparisons.
struct QuantizedGraphEdgeExport {
  template <class Graph, class GetNeighbors>
  static bool write(const Graph &entity, size_t docs, size_t dim,
                    const char *path, GetNeighbors get_neighbors) {
    if (!path || dim != 768 || docs > UINT32_MAX || !docs) return false;
    struct Sample {
      std::array<uint32_t, 4> ids;
      uint32_t degree;
      std::array<uint32_t, 3> offsets;
    };
    std::vector<Sample> samples;
    std::unordered_set<uint32_t> seen;
    std::mt19937 rng(20261026);
    for (size_t i = 0; i < std::min(docs, size_t{65536}); ++i) {
      const uint32_t id = uint32_t((uint64_t(i) * 2654435761ULL + 7193) % docs);
      if (!entity.valid(id) || !seen.insert(id).second) continue;
      const auto neighbors = get_neighbors(id);
      std::vector<uint32_t> offsets;
      std::unordered_set<uint32_t> unique{id};
      for (size_t j = 0; j < neighbors.second; ++j) {
        const uint32_t other = neighbors.first[j];
        if (other < docs && entity.valid(other) && unique.insert(other).second)
          offsets.push_back(uint32_t(j));
      }
      if (offsets.size() < 3) continue;
      std::shuffle(offsets.begin(), offsets.end(), rng);
      samples.push_back(
          {{id, neighbors.first[offsets[0]], neighbors.first[offsets[1]],
            neighbors.first[offsets[2]]},
           uint32_t(neighbors.second),
           {offsets[0], offsets[1], offsets[2]}});
    }
    if (samples.empty()) return false;
    FILE *f = std::fopen(path, "wbx");
    if (!f) return false;
    const uint32_t header[] = {
        0x51474561,     1, uint32_t(samples.size()), uint32_t(dim),
        uint32_t(docs), 3};
    bool ok = std::fwrite(header, sizeof(header), 1, f) == 1;
    for (const auto &s : samples) {
      if (!ok) break;
      std::array<uint64_t, 4> keys;
      for (size_t j = 0; j < 4; ++j) keys[j] = entity.key(s.ids[j]);
      ok = std::fwrite(s.ids.data(), sizeof(s.ids), 1, f) == 1 &&
           std::fwrite(keys.data(), sizeof(keys), 1, f) == 1 &&
           std::fwrite(&s.degree, sizeof(s.degree), 1, f) == 1 &&
           std::fwrite(s.offsets.data(), sizeof(s.offsets), 1, f) == 1;
      for (uint32_t id : s.ids) {
        if (!ok) break;
        typename Graph::Vector raw;
        if (entity.get_vector(id, raw) || !raw.data()) {
          ok = false;
          break;
        }
        const float *v = static_cast<const float *>(raw.data());
        double norm = 0;
        for (size_t d = 0; d < dim; ++d) norm += double(v[d]) * v[d];
        if (!std::isfinite(norm) || std::abs(norm - 1.) > 1e-4) {
          ok = false;
          break;
        }
        ok = std::fwrite(v, sizeof(float), dim, f) == dim;
      }
    }
    ok = !std::ferror(f) && ok;
    return std::fclose(f) == 0 && ok;
  }
};
}  // namespace zvec::core
