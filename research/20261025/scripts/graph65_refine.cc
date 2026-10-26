// Offline deterministic graph refinement. No search queries or ground truth.
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <immintrin.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using U = uint32_t;
constexpr size_t D = 768, K = 32;
constexpr U EMPTY = UINT32_MAX;
struct Map {
  int fd;
  size_t size;
  const char *p;
  explicit Map(const std::string &path) {
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("open " + path);
    struct stat st {};
    if (fstat(fd, &st)) throw std::runtime_error("stat");
    size = st.st_size;
    p = static_cast<const char *>(
        mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (p == MAP_FAILED) throw std::runtime_error("mmap");
  }
  ~Map() {
    munmap(const_cast<char *>(p), size);
    close(fd);
  }
};
U u32(const char *p) {
  U x;
  std::memcpy(&x, p, 4);
  return x;
}
void check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
double l2(const float *a, const float *b) {
  __m512d s0 = _mm512_setzero_pd(), s1 = s0;
  for (size_t j = 0; j < D; j += 16) {
    auto d0 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(a + j)),
                            _mm512_cvtps_pd(_mm256_loadu_ps(b + j)));
    auto d1 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(a + j + 8)),
                            _mm512_cvtps_pd(_mm256_loadu_ps(b + j + 8)));
    s0 = _mm512_fmadd_pd(d0, d0, s0);
    s1 = _mm512_fmadd_pd(d1, d1, s1);
  }
  return _mm512_reduce_add_pd(_mm512_add_pd(s0, s1));
}
using Row = std::array<U, 33>;
struct Item {
  U id;
  double r;
};
struct Scratch {
  std::vector<Item> pool;
  std::vector<float> coords;
  std::vector<size_t> seeds, angular, nearest;
  std::vector<double> min_chord;
  std::vector<unsigned char> used;
  double pair(size_t i, size_t j) const {
    return l2(coords.data() + i * D, coords.data() + j * D);
  }
  double chord(size_t i, size_t j) const {
    double c = (pool[i].r + pool[j].r - pair(i, j)) /
               (2 * std::sqrt(pool[i].r * pool[j].r));
    return std::clamp(2 - 2 * c, 0., 4.);
  }
  void prune() {
    seeds.clear();
    for (size_t i = 0; i < pool.size() && seeds.size() < K; ++i) {
      bool keep = true;
      for (size_t j : seeds)
        if (pair(i, j) < pool[i].r) {
          keep = false;
          break;
        }
      if (keep) seeds.push_back(i);
    }
  }
  void fills() {
    used.assign(pool.size(), 0);
    for (auto i : seeds) used[i] = 1;
    nearest = seeds;
    for (size_t i = 0; i < pool.size() && nearest.size() < K; ++i)
      if (!used[i]) nearest.push_back(i);
    angular = seeds;
    min_chord.assign(pool.size(), 4.);
    auto update = [&](size_t j) {
      if (pool[j].r == 0) return;
      for (size_t i = 0; i < pool.size(); ++i)
        if (!used[i] && pool[i].r > 0)
          min_chord[i] = std::min(min_chord[i], chord(i, j));
    };
    for (auto j : seeds) update(j);
    while (angular.size() < K) {
      size_t best = pool.size();
      double score = -1;
      // Pool is stable sorted by distance and ID; strict > gives deterministic
      // ties.
      for (size_t i = 0; i < pool.size(); ++i)
        if (!used[i] && pool[i].r > 0 && min_chord[i] > score) {
          best = i;
          score = min_chord[i];
        }
      if (best == pool.size())
        for (size_t i = 0; i < pool.size(); ++i)
          if (!used[i]) {
            best = i;
            break;
          }
      check(best < pool.size(), "insufficient candidates");
      angular.push_back(best);
      used[best] = 1;
      update(best);
    }
  }
  double coverage(const std::vector<size_t> &chosen) const {
    double radius = 0;
    for (size_t i = 0; i < pool.size(); ++i)
      if (pool[i].r > 0) {
        double best = 4;
        for (auto j : chosen)
          if (pool[j].r > 0) best = std::min(best, i == j ? 0. : chord(i, j));
        radius = std::max(radius, best);
      }
    return std::sqrt(radius);
  }
  Row row(const std::vector<size_t> &indices) const {
    Row r;
    r.fill(EMPTY);
    r[0] = indices.size();
    for (size_t j = 0; j < indices.size(); ++j) r[j + 1] = pool[indices[j]].id;
    return r;
  }
};
void save(const std::string &path, const void *p, size_t n) {
  FILE *f = std::fopen(path.c_str(), "wbx");
  check(f, "exclusive output open");
  bool ok = std::fwrite(p, 1, n, f) == n;
  ok = std::fclose(f) == 0 && ok;
  check(ok, "write output");
}
int main(int argc, char **argv) try {
  check(argc == 6,
        "usage: graph65_refine input_directory output_prefix count threads "
        "diagnostic_stride");
  std::string input = argv[1], prefix = argv[2];
  Map graph(input + "/full_graph.bin"), vectors(input + "/full_vectors.f32"),
      candidates(input + "/full_candidates.bin");
  check(graph.size >= 32 && vectors.size >= 16 && candidates.size >= 32,
        "short header");
  U n = u32(graph.p + 8), count = std::stoul(argv[3]),
    threads = std::stoul(argv[4]);
  // Sixth argument selects coverage diagnostics stride, bounded for full runs.
  U stride = std::stoul(argv[5]);
  check(n > 0 && count > 0 && count <= n && threads > 0 && stride > 0,
        "arguments");
  check(u32(graph.p) == 0x51474764 && u32(graph.p + 4) == 1 &&
            u32(graph.p + 12) == D && u32(graph.p + 16) == K &&
            u32(graph.p + 20) < n,
        "graph header");
  check(graph.size == 32 + size_t(n) * 140, "graph length");
  check(u32(vectors.p) == 0x51475664 && u32(vectors.p + 4) == 1 &&
            u32(vectors.p + 8) == n && u32(vectors.p + 12) == D &&
            vectors.size == 16 + size_t(n) * D * 4,
        "vectors header/size");
  check(u32(candidates.p) == 0x51474364 && u32(candidates.p + 4) == 1 &&
            u32(candidates.p + 8) == n && u32(candidates.p + 12) == D &&
            u32(candidates.p + 16) == n && u32(candidates.p + 20) == 200 &&
            candidates.size == 32 + size_t(n) * 1604,
        "candidates header/size");
  const float *v = reinterpret_cast<const float *>(vectors.p + 16);
  std::vector<U> inverse(n, EMPTY);
  for (U i = 0; i < n; ++i) {
    U q = u32(candidates.p + 32 + size_t(i) * 1604);
    check(q < n && inverse[q] == EMPTY, "candidate permutation");
    inverse[q] = i;
  }
  std::vector<Row> first(count), nearest(count), angular(count);
  std::vector<uint64_t> offsets(n + 1, 0);
  std::vector<U> incoming;
  // stats: pool before cap, pool after cap, seeds, reverse indegree, zero
  // residuals, nearest/angular coverage (NaN when unsampled), mean L2, max L2
  // for both fills.
  std::vector<std::array<double, 11>> stats(count);
  auto start = std::chrono::steady_clock::now();
  auto pass = [&](bool final) {
    std::atomic<U> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
    std::mutex error_mutex;
    auto work = [&]() {
      try {
        Scratch s;
        std::vector<U> ids;
        while (!failed.load()) {
          U id = next.fetch_add(1);
          if (id >= count) break;
          ids.clear();
          const char *c = candidates.p + 32 + size_t(inverse[id]) * 1604;
          for (size_t j = 0; j < 200; ++j) ids.push_back(u32(c + 4 + j * 4));
          const char *g = graph.p + 32 + size_t(id) * 140;
          U degree = u32(g + 8);
          check(degree <= K, "source degree");
          for (size_t j = 0; j < degree; ++j)
            ids.push_back(u32(g + 12 + j * 4));
          if (final)
            ids.insert(ids.end(), incoming.begin() + offsets[id],
                       incoming.begin() + offsets[id + 1]);
          std::sort(ids.begin(), ids.end());
          ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
          s.pool.clear();
          for (U x : ids) {
            check(x < n, "candidate ID");
            if (x != id)
              s.pool.push_back({x, l2(v + size_t(id) * D, v + size_t(x) * D)});
          }
          size_t before = s.pool.size();
          std::sort(s.pool.begin(), s.pool.end(), [](Item a, Item b) {
            return a.r < b.r || (a.r == b.r && a.id < b.id);
          });
          if (s.pool.size() > 750) s.pool.resize(750);
          check(s.pool.size() >= K, "pool too small");
          s.coords.resize(s.pool.size() * D);
          for (size_t j = 0; j < s.pool.size(); ++j)
            std::memcpy(s.coords.data() + j * D, v + size_t(s.pool[j].id) * D,
                        D * 4);
          s.prune();
          if (!final)
            first[id] = s.row(s.seeds);
          else {
            s.fills();
            nearest[id] = s.row(s.nearest);
            angular[id] = s.row(s.angular);
            auto &st = stats[id];
            st.fill(0);
            st[0] = before;
            st[1] = s.pool.size();
            st[2] = s.seeds.size();
            st[3] = offsets[id + 1] - offsets[id];
            for (auto item : s.pool) st[4] += item.r == 0;
            st[5] = st[6] = std::numeric_limits<double>::quiet_NaN();
            if (id % stride == 0) {
              st[5] = s.coverage(s.nearest);
              st[6] = s.coverage(s.angular);
            }
            for (auto j : s.nearest) {
              st[7] += s.pool[j].r / K;
              st[9] = std::max(st[9], s.pool[j].r);
            }
            for (auto j : s.angular) {
              st[8] += s.pool[j].r / K;
              st[10] = std::max(st[10], s.pool[j].r);
            }
          }
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!failure) failure = std::current_exception();
        failed = true;
      }
    };
    std::vector<std::thread> workers;
    for (U t = 0; t < threads; ++t) workers.emplace_back(work);
    for (auto &t : workers) t.join();
    if (failure) std::rethrow_exception(failure);
    std::printf(
        "PASS final=%d elapsed_seconds=%.9f\n", final,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    std::fflush(stdout);
  };
  pass(false);
  for (auto &r : first)
    for (U j = 0; j < r[0]; ++j) ++offsets[r[j + 1] + 1];
  for (U i = 0; i < n; ++i) offsets[i + 1] += offsets[i];
  incoming.resize(offsets[n]);
  auto cursor = offsets;
  for (U i = 0; i < count; ++i)
    for (U j = 0; j < first[i][0]; ++j) incoming[cursor[first[i][j + 1]]++] = i;
  cursor.clear();
  cursor.shrink_to_fit();
  save(prefix + "_first.u32", first.data(), first.size() * sizeof(Row));
  pass(true);
  save(prefix + "_nearest.u32", nearest.data(), nearest.size() * sizeof(Row));
  save(prefix + "_angular.u32", angular.data(), angular.size() * sizeof(Row));
  save(prefix + "_stats.f64", stats.data(), stats.size() * sizeof(stats[0]));
  if (count == n)
    for (auto mode : {0, 1}) {
      const auto &rows = mode ? angular : nearest;
      std::vector<char> out(graph.size);
      std::memcpy(out.data(), graph.p, 32);
      for (U i = 0; i < n; ++i) {
        std::memcpy(out.data() + 32 + size_t(i) * 140,
                    graph.p + 32 + size_t(i) * 140, 8);
        std::memcpy(out.data() + 40 + size_t(i) * 140, rows[i].data(), 132);
      }
      save(prefix + (mode ? "_angular.bin" : "_nearest.bin"), out.data(),
           out.size());
    }
  std::printf(
      "DONE count=%u threads=%u reverse_edges=%zu seconds=%.9f\n", count,
      threads, incoming.size(),
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
