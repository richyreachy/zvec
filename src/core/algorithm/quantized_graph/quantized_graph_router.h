#pragma once
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
namespace zvec::core {
// Experimental query-independent anchor sampler. The projection is supplied
// by the codec's existing linear transform; no query or GT trains this router.
class QuantizedGraphRouter {
 public:
  void clear() {
    ids.clear();
    norms.clear();
    matrix.clear();
    mean.clear();
  }
  template <class Graph, class Codec>
  int build(const Graph &entity, const Codec &codec,
            const std::vector<double> &centroid, size_t count, size_t dim) {
    clear();
    if (!count) return 0;
    for (size_t i = 0; i < std::min(size_t{1024}, count); ++i) {
      const uint32_t id =
          uint32_t((uint64_t(i) * 2654435761ULL + 7193) % count);
      if (entity.valid(id) &&
          std::find(ids.begin(), ids.end(), id) == ids.end())
        ids.push_back(id);
    }
    std::vector<float> center(centroid.begin(), centroid.end()), rotated;
    codec.transform(center.data(), mean);
    norms.resize(ids.size());
    matrix.resize(ids.size() * mean.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      typename Graph::Vector raw;
      int rc = entity.get_vector(ids[i], raw);
      if (rc) return rc;
      if (!raw.data()) return -1;
      const float *v = static_cast<const float *>(raw.data());
      double norm = 0;
      for (size_t j = 0; j < dim; ++j) {
        double x = double(v[j]) - centroid[j];
        norm += x * x;
      }
      norms[i] = float(norm);
      codec.transform(v, rotated);
      for (size_t j = 0; j < mean.size(); ++j)
        matrix[j * ids.size() + i] = rotated[j] - mean[j];
    }
    return 0;
  }
  bool empty() const {
    return ids.empty();
  }
  template <class Query>
  static auto rotated(const Query &q, int) -> decltype(q.rotated.data()) {
    return q.rotated.data();
  }
  template <class Query>
  static const float *rotated(const Query &, long) {
    return nullptr;
  }
  __attribute__((target("avx512f,fma"))) void score(
      const float *q, size_t dimensions, size_t count,
      std::vector<float> &scores) const {
    scores.resize(count);
    const float factor = 2.f * float(mean.size()) / float(dimensions);
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
      __m512 sum = _mm512_loadu_ps(norms.data() + i);
      for (size_t j = 0; j < dimensions; ++j) {
        const __m512 coefficient = _mm512_set1_ps(factor * (q[j] - mean[j]));
        sum = _mm512_fnmadd_ps(
            coefficient, _mm512_loadu_ps(matrix.data() + j * ids.size() + i),
            sum);
      }
      _mm512_storeu_ps(scores.data() + i, sum);
    }
    for (; i < count; ++i) {
      float s = norms[i];
      for (size_t j = 0; j < dimensions; ++j)
        s -= factor * (q[j] - mean[j]) * matrix[j * ids.size() + i];
      scores[i] = s;
    }
  }
  template <class Context, class Graph>
  int choose(Context &ctx, const Graph &entity, const float *rotated_query,
             uint32_t &entry, size_t dimensions, size_t count, size_t shortlist,
             std::vector<float> &scores) const {
    count = std::min(count, ids.size());
    dimensions = std::min(dimensions, mean.size());
    shortlist = std::min({shortlist, count, size_t{16}});
    if (!count || !dimensions || !shortlist) return 0;
    score(rotated_query, dimensions, count, scores);
    std::array<std::pair<float, uint32_t>, 16> top;
    top.fill({std::numeric_limits<float>::infinity(), UINT32_MAX});
    for (size_t i = 0; i < count; ++i) {
      if (scores[i] >= top[shortlist - 1].first) continue;
      size_t j = shortlist - 1;
      while (j && scores[i] < top[j - 1].first) {
        top[j] = top[j - 1];
        --j;
      }
      top[j] = {scores[i], uint32_t(i)};
    }
    typename Graph::Vector raw;
    int rc = entity.get_vector(entry, raw);
    if (rc) return rc;
    if (!raw.data()) return -1;
    float best = ctx.distance(raw.data());
    for (size_t i = 0; i < shortlist; ++i) {
      const uint32_t candidate = ids[top[i].second];
      rc = entity.get_vector(candidate, raw);
      if (rc) return rc;
      if (!raw.data()) return -1;
      const float d = ctx.distance(raw.data());
      if (d < best) {
        best = d;
        entry = candidate;
      }
    }
    return 0;
  }

 private:
  std::vector<uint32_t> ids;
  std::vector<float> norms, matrix, mean;
};
}  // namespace zvec::core
