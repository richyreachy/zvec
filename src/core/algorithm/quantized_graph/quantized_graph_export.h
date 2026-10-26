#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <unordered_set>

namespace zvec::core {
// Database-only research export of the final graph and its internal-ID vectors.
// Exclusive creation preserves existing artifacts; partial failed files remain
// for inspection. Integer fields use this machine's native little-endian order.
struct QuantizedGraphExport {
  template <class Graph, class GetNeighbors>
  static bool write(const Graph &entity, size_t docs, size_t dim,
                    uint32_t entry, const char *graph_path,
                    const char *vector_path, GetNeighbors neighbors) {
    if (!graph_path || !vector_path || !docs || docs > UINT32_MAX ||
        dim != 768 || entry >= docs)
      return false;
    FILE *graph = std::fopen(graph_path, "wbx");
    if (!graph) return false;
    FILE *vectors = std::fopen(vector_path, "wbx");
    if (!vectors) {
      std::fclose(graph);
      return false;
    }
    const uint32_t gh[8] = {
        0x51474764, 1, uint32_t(docs), uint32_t(dim), 32, entry, 0, 0};
    const uint32_t vh[4] = {0x51475664, 1, uint32_t(docs), uint32_t(dim)};
    bool ok = std::fwrite(gh, sizeof(gh), 1, graph) == 1 &&
              std::fwrite(vh, sizeof(vh), 1, vectors) == 1;
    std::unordered_set<uint64_t> keys;
    for (size_t i = 0; i < docs && ok; ++i) {
      const uint32_t id = uint32_t(i);
      if (!entity.valid(id)) {
        ok = false;
        break;
      }
      const uint64_t key = entity.key(id);
      if (!keys.insert(key).second) {
        ok = false;
        break;
      }
      auto ns = neighbors(id);
      if (ns.second > 32) {
        ok = false;
        break;
      }
      const uint32_t degree = uint32_t(ns.second);
      std::array<uint32_t, 32> adj;
      adj.fill(UINT32_MAX);
      for (size_t j = 0; j < degree; ++j) {
        uint32_t target = ns.first[j];
        if (target >= docs || target == id || !entity.valid(target) ||
            std::find(adj.begin(), adj.begin() + j, target) !=
                adj.begin() + j) {
          ok = false;
          break;
        }
        adj[j] = target;
      }
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
      ok = std::fwrite(&key, sizeof(key), 1, graph) == 1 &&
           std::fwrite(&degree, sizeof(degree), 1, graph) == 1 &&
           std::fwrite(adj.data(), sizeof(adj), 1, graph) == 1 &&
           std::fwrite(v, sizeof(float), dim, vectors) == dim;
    }
    ok = !std::ferror(graph) && !std::ferror(vectors) && ok;
    const int vg = std::fclose(graph), vv = std::fclose(vectors);
    return ok && vg == 0 && vv == 0;
  }
};
}  // namespace zvec::core
