// Stage78 real QuantizedGraph integration fixture. Synthetic, not optimized
// training.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "core/algorithm/quantized_graph/quantized_graph.h"
#include "core/algorithm/quantized_graph/sq8_graph_codec.h"
#include "core/algorithm/quantized_graph/symphony_qg_codec.h"
#include "coupled73_codec.h"

namespace {
constexpr size_t N = 40, D = 768, B = 3328;
using Base = zvec::turbo::SymphonyQGBlockQuantizer;
void require(bool yes, const std::string &message) {
  if (!yes) throw std::runtime_error(message);
}
std::vector<char> read(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  require(bool(f), "open " + p);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
void write(const std::string &p, const void *data, size_t n) {
  std::ofstream f(p, std::ios::binary);
  require(bool(f.write(static_cast<const char *>(data), n)), "write " + p);
}
template <class T>
T take(const std::vector<char> &b, size_t off) {
  T x;
  require(off + sizeof(x) <= b.size(), "fixture truncated");
  std::memcpy(&x, b.data() + off, sizeof(x));
  return x;
}
struct Graph {
  struct Vector {
    const float *values = nullptr;
    const void *data() const {
      return values;
    }
  };
  std::vector<float> vectors;
  std::array<std::vector<uint32_t>, N> links;
  std::array<uint64_t, N> keys;
  std::array<bool, N> live;
  uint32_t failed = UINT32_MAX;
  bool valid(uint32_t id) const {
    return id < N && live[id];
  }
  uint64_t key(uint32_t id) const {
    return keys.at(id);
  }
  const std::vector<uint32_t> &neighbors(uint32_t id) const {
    return links.at(id);
  }
  int get_vector(uint32_t id, Vector &out) const {
    if (id == failed) return zvec::core::IndexError_ReadData;
    out.values = vectors.data() + id * D;
    return 0;
  }
  explicit Graph(const std::string &dir) {
    const auto raw = read(dir + "/raw.f32"), graph = read(dir + "/graph.bin");
    require(raw.size() == 16 + N * D * 4, "raw size");
    vectors.resize(N * D);
    std::memcpy(vectors.data(), raw.data() + 16, N * D * 4);
    require(graph.size() == N * 140, "graph size");
    for (size_t i = 0; i < N; ++i) {
      keys[i] = take<uint64_t>(graph, i * 140);
      const auto degree = take<uint32_t>(graph, i * 140 + 8);
      require(degree <= 32 && keys[i] != i, "degree/key fixture");
      for (size_t j = 0; j < degree; ++j)
        links[i].push_back(take<uint32_t>(graph, i * 140 + 12 + j * 4));
      live[i] = true;
    }
  }
};
struct NoKeyGraph {
  using Vector = Graph::Vector;
  const Graph &g;
  bool valid(uint32_t i) const {
    return g.valid(i);
  }
  const std::vector<uint32_t> &neighbors(uint32_t i) const {
    return g.neighbors(i);
  }
  int get_vector(uint32_t i, Vector &v) const {
    return g.get_vector(i, v);
  }
};
struct Query {
  struct Results {
    size_t capacity = 64;
    std::vector<std::pair<uint32_t, float>> values;
    size_t limit() const {
      return capacity;
    }
    size_t size() const {
      return values.size();
    }
    void clear() {
      values.clear();
    }
    void emplace(uint32_t id, float d) {
      values.emplace_back(id, d);
      std::sort(values.begin(), values.end(), [](const auto &a, const auto &b) {
        return a.second < b.second ||
               (a.second == b.second && a.first < b.first);
      });
      if (values.size() > capacity) values.resize(capacity);
    }
  } results;
  std::vector<float> q;
  size_t scans = 0, expansions = 0;
  Results &reset_results() {
    scans = expansions = 0;
    return results;
  }
  size_t topk() const {
    return 10;
  }
  const void *query() const {
    return q.data();
  }
  bool reach_scan_limit() const {
    return scans >= 10000;
  }
  bool excluded(uint32_t) const {
    return false;
  }
  void on_expand() {
    ++expansions;
  }
  float distance(const void *raw) {
    ++scans;
    const auto *v = static_cast<const float *>(raw);
    float dot = 0;
    for (size_t j = 0; j < D; ++j) dot += q[j] * v[j];
    return 1.f - dot;
  }
};
std::vector<char> native_codes, imported_codes, native_signs, imported_signs;
std::array<size_t, N> degrees;
std::array<std::atomic<uint64_t>, N> coverage{};
std::atomic<uint64_t> scan_calls{0}, scalar_lanes{0};
bool importing = true;
struct AuditedCodec : Base {
  using Base::Base;
  void scan(const char *codes, Query &query, float center,
            float *scores) const {
    const auto &bank = importing ? imported_codes : native_codes;
    const auto &signbank = importing ? imported_signs : native_signs;
    size_t id = N;
    for (size_t i = 0; i < N; ++i)
      if (degrees[i] && std::memcmp(codes, bank.data() + i * B, B) == 0) {
        id = i;
        break;
      }
    require(id < N, "actual graph scan used code bytes outside expected bank");
    coverage[id].fetch_add(1, std::memory_order_relaxed);
    scan_calls.fetch_add(1, std::memory_order_relaxed);
    Base::scan(codes, query, center, scores);
    std::array<float, 32> reference;
    Base::scan(bank.data() + id * B, query, center, reference.data());
    require(std::memcmp(scores, reference.data(), 32 * sizeof(float)) == 0,
            "actual FastScan differs from independent expected code scan");
    rabitqlib::ConstQGBatchDataMap<float> factors(bank.data() + id * B, D);
    for (size_t lane = 0; lane < degrees[id]; ++lane) {
      uint32_t total = 0;
      for (size_t book = 0; book < D / 4; ++book) {
        unsigned nibble = 0;
        for (size_t k = 0; k < 4; ++k)
          if (signbank[(id * 32 + lane) * D + book * 4 + k] > 0)
            nibble |= 1u << (3 - k);
        total += query.lookup.lut()[book * 16 + nibble];
      }
      float value = std::fma(query.lookup.delta(), float(total),
                             query.lookup.sum_vl_lut());
      value += query.lookup.k1xsumq();
      const float expected = std::fma(factors.f_rescale()[lane], value,
                                      factors.f_add()[lane] + center);
      require(std::memcmp(&scores[lane], &expected, sizeof(float)) == 0,
              "actual FastScan differs from scalar LUT/FMA oracle at row " +
                  std::to_string(id));
      scalar_lanes.fetch_add(1, std::memory_order_relaxed);
    }
  }
};
// The inherited Query name above denotes Base::Query; graph query is ::Query.
using QGraph = zvec::core::QuantizedGraph<AuditedCodec>;
void prepare(const std::string &dir, const Graph &g) {
  Base codec(D, true);
  std::vector<float> rotated(N * D), one, neighbors(32 * D, 0.f);
  native_codes.assign(N * B, 0);
  imported_codes.assign(N * B, 0);
  native_signs.assign(N * 32 * D, 0);
  imported_signs.assign(N * 32 * D, 0);
  for (size_t i = 0; i < N; ++i) {
    codec.transform(g.vectors.data() + i * D, one);
    require(one.size() == D, "real rotation dimension");
    std::copy(one.begin(), one.end(), rotated.begin() + i * D);
  }
  size_t flips = 0;
  for (size_t i = 0; i < N; ++i) {
    const auto &links = g.links[i];
    std::fill(neighbors.begin(), neighbors.end(), 0.f);
    for (size_t lane = 0; lane < links.size(); ++lane) {
      const auto *c = rotated.data() + i * D,
                 *v = rotated.data() + links[lane] * D;
      std::copy_n(v, D, neighbors.data() + lane * D);
      size_t minimum = 0;
      double smallest = INFINITY, denom = 0;
      for (size_t j = 0; j < D; ++j) {
        const double r = double(v[j]) - c[j];
        const auto sign = int8_t(r > 0 ? 1 : -1);
        native_signs[(i * 32 + lane) * D + j] =
            imported_signs[(i * 32 + lane) * D + j] = sign;
        denom += std::abs(r);
        if (std::abs(r) < smallest) {
          smallest = std::abs(r);
          minimum = j;
        }
      }
      if (links.size() == 32) {
        require(
            denom - 2 * smallest > 0 && denom / (denom - 2 * smallest) <= 1.02,
            "synthetic flip denominator/cap");
        imported_signs[(i * 32 + lane) * D + minimum] *= -1;
        ++flips;
      }
    }
    if (!links.empty())
      codec.encode(rotated.data() + i * D, neighbors.data(), links.size(),
                   native_codes.data() + i * B);
    std::copy_n(native_codes.data() + i * B, B, imported_codes.data() + i * B);
    if (links.size() == 32) {
      coupled73::encode(
          rotated.data() + i * D, neighbors.data(),
          reinterpret_cast<const int8_t *>(imported_signs.data() + i * 32 * D),
          32, D, imported_codes.data() + i * B);
      require(std::memcmp(native_codes.data() + i * B,
                          imported_codes.data() + i * B, B) != 0,
              "synthetic full block must differ from native");
    }
  }
  write(dir + "/rot.f32", rotated.data(), rotated.size() * 4);
  write(dir + "/native.codes", native_codes.data(), native_codes.size());
  write(dir + "/imported.codes", imported_codes.data(), imported_codes.size());
  write(dir + "/native.signs", native_signs.data(), native_signs.size());
  write(dir + "/imported.signs", imported_signs.data(), imported_signs.size());
  std::cout << "{\"real_rotation_vectors\":40,\"synthetic_sign_flips\":"
            << flips << "}\n";
}
void enable(const std::string &dir, const std::string &sha) {
  setenv("QG_INDEPENDENT_CODE_FILE", (dir + "/model.bin").c_str(), 1);
  setenv("QG_INDEPENDENT_RAW_FILE", (dir + "/raw.f32").c_str(), 1);
  setenv("QG_INDEPENDENT_ROTATED_FILE", (dir + "/rot.f32").c_str(), 1);
  setenv("QG_INDEPENDENT_HEADER_FILE", (dir + "/header.bin").c_str(), 1);
  setenv("QG_INDEPENDENT_MODEL_SHA256", sha.c_str(), 1);
}
Query query_for(const Graph &g, size_t i) {
  Query q;
  q.q.assign(g.vectors.data() + i * D, g.vectors.data() + (i + 1) * D);
  return q;
}
void rejected_search(const QGraph &qg, const Graph &g) {
  auto q = query_for(g, 0);
  require(qg.search(0, g, q) == zvec::core::IndexError_Mismatch,
          "failed/cleared snapshot must reject normal search");
  require(qg.search(UINT32_MAX, g, q) == zvec::core::IndexError_Mismatch,
          "failed/cleared snapshot must reject invalid-entry early exit");
}
void scan_all(const QGraph &qg, const Graph &g, size_t threads) {
  for (auto &v : coverage) v.store(0);
  std::atomic<size_t> next{0};
  std::mutex mutex;
  std::string failure;
  auto run = [&]() {
    try {
      for (;;) {
        const auto i = next.fetch_add(1);
        if (i >= N * 2) break;
        auto q = query_for(g, (i * 13 + 7) % N);
        require(qg.search(i % N, g, q) == 0, "search status");
        require(q.results.size() > 0, "search returned no results");
      }
    } catch (const std::exception &e) {
      std::lock_guard<std::mutex> lock(mutex);
      failure = e.what();
    }
  };
  std::vector<std::thread> pool;
  for (size_t i = 1; i < threads; ++i) pool.emplace_back(run);
  run();
  for (auto &t : pool) t.join();
  require(failure.empty(), failure);
  for (size_t i = 0; i < N; ++i)
    require(!degrees[i] || coverage[i].load() > 0,
            "nonempty block never actually scanned: " + std::to_string(i));
}
void test(const std::string &dir, const std::string &sha, Graph &g) {
  native_codes = read(dir + "/native.codes");
  imported_codes = read(dir + "/imported.codes");
  native_signs = read(dir + "/native.signs");
  imported_signs = read(dir + "/imported.signs");
  require(native_codes.size() == N * B && imported_codes.size() == N * B,
          "code bank size");
  require(
      native_signs.size() == N * 32 * D && imported_signs.size() == N * 32 * D,
      "sign bank size");
  for (size_t i = 0; i < N; ++i) degrees[i] = g.links[i].size();
  std::vector<std::string> checks;
  auto pass = [&](const std::string &name) {
    checks.push_back(name);
    std::cerr << "PASS78 " << name << '\n';
  };
  enable(dir, sha);
  QGraph qg(D, AuditedCodec(D, true), 32, 0, true);
  require(qg.prebuild(g, N, 1) == 0, "one-thread prebuild");
  scan_all(qg, g, 1);
  pass("one_thread_prebuild_all_39_nonempty_blocks_real_scan");
  require(qg.prebuild(g, N, 8) == 0, "eight-thread repeated prebuild");
  scan_all(qg, g, 8);
  pass("eight_thread_repeated_prebuild_and_concurrent_queries");
  const auto maps = read("/proc/self/maps");
  const std::string maptext(maps.begin(), maps.end());
  require(maptext.find(dir + "/model.bin") == std::string::npos &&
              maptext.find(dir + "/rot.f32") == std::string::npos &&
              maptext.find(dir + "/raw.f32") == std::string::npos,
          "import mapped sources not released");
  pass("mapped_sources_released_before_queries");
  qg.clear();
  rejected_search(qg, g);
  pass("clear_rejects_queries_until_rebuild");
  require(qg.prebuild(g, N, 8) == 0, "rebuild after clear");
  scan_all(qg, g, 1);
  pass("clear_then_rebuild_restores_codes");
  auto failure = [&](const std::string &name, auto mutate, auto restore) {
    qg.clear();
    mutate();
    const int rc = qg.prebuild(g, N, 8);
    require(rc != 0, name + " must reject prebuild");
    rejected_search(qg, g);
    restore();
    require(qg.prebuild(g, N, 8) == 0, name + " repaired rebuild");
    scan_all(qg, g, 1);
    pass(name + "_fails_closed_then_recovers");
  };
  const float v0 = g.vectors[0];
  failure(
      "same_key_source_one_ulp",
      [&] { g.vectors[0] = std::nextafter(v0, INFINITY); },
      [&] { g.vectors[0] = v0; });
  const auto key0 = g.keys[0];
  failure(
      "center_external_key_change", [&] { g.keys[0] += 1; },
      [&] { g.keys[0] = key0; });
  failure(
      "neighbor_order_change", [&] { std::swap(g.links[0][0], g.links[0][1]); },
      [&] { std::swap(g.links[0][0], g.links[0][1]); });
  failure(
      "deleted_node", [&] { g.live[2] = false; }, [&] { g.live[2] = true; });
  failure(
      "get_vector_error", [&] { g.failed = 6; },
      [&] { g.failed = UINT32_MAX; });
  failure(
      "wrong_model_sha",
      [&] {
        setenv("QG_INDEPENDENT_MODEL_SHA256", std::string(64, '0').c_str(), 1);
      },
      [&] { setenv("QG_INDEPENDENT_MODEL_SHA256", sha.c_str(), 1); });
  zvec::core::QuantizedGraph<zvec::core::SQ8GraphCodec> sq(
      D, zvec::core::SQ8GraphCodec(D, true), 32, 0, true);
  require(sq.prebuild(g, N, 8) == zvec::core::IndexError_Mismatch,
          "SQ8 must reject imported model");
  auto sqquery = query_for(g, 0);
  require(sq.search(0, g, sqquery) == zvec::core::IndexError_Mismatch,
          "SQ8 imported failure must reject query");
  pass("sq8_import_rejected_and_query_closed");
  NoKeyGraph nokey{g};
  QGraph no_key_qg(D, AuditedCodec(D, true), 32, 0, true);
  require(no_key_qg.prebuild(nokey, N, 1) != 0,
          "no-key graph imported model rejection");
  pass("no_key_graph_compiles_and_import_rejects");
  unsetenv("QG_INDEPENDENT_CODE_FILE");
  importing = false;
  require(qg.prebuild(g, N, 8) == 0, "explicit model disable native prebuild");
  scan_all(qg, g, 8);
  pass("explicit_disable_rebuild_uses_native_codec");
  qg.clear();
  scan_all(qg, g, 8);
  pass("native_clear_restores_lazy_graph_build_and_concurrent_query");
  require(no_key_qg.prebuild(nokey, N, 8) == 0, "no-key native graph prebuild");
  auto generic_query = query_for(g, 0);
  require(no_key_qg.search(0, nokey, generic_query) == 0,
          "no-key native graph query");
  pass("no_key_native_prebuild_and_query");
  require(sq.prebuild(g, N, 8) == 0, "SQ8 native recovery prebuild");
  require(sq.search(0, g, sqquery) == 0, "SQ8 native recovery query");
  pass("sq8_native_recovery_prebuild_and_query");
  std::cout
      << "{\"passed\":true,\"synthetic_nodes\":40,\"dimensions\":768,\"full_"
         "blocks\":36,\"short_nonempty_blocks\":3,\"empty_blocks\":1,\"max_"
         "readonly_query_threads\":8,\"actual_fastscan_calls\":"
      << scan_calls.load()
      << ",\"bit_identical_scalar_lut_fma_lane_checks\":" << scalar_lanes.load()
      << ",\"checks\":[";
  for (size_t i = 0; i < checks.size(); ++i)
    std::cout << (i ? "," : "") << '\"' << checks[i] << '\"';
  std::cout << "],\"scope\":\"40 synthetic normalized vectors;real "
               "QuantizedGraph,Symphony rotation,FastScan;no optimized "
               "training or full1M model validation or Cohere timing\"}\n";
}
}  // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 3 || argc == 4,
            "usage: import78_integration prepare DIR | test DIR SHA256");
    Graph graph(argv[2]);
    if (std::string(argv[1]) == "prepare")
      prepare(argv[2], graph);
    else {
      require(argc == 4 && std::string(argv[1]) == "test", "bad mode");
      test(argv[2], argv[3], graph);
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL78 " << e.what() << '\n';
    return 1;
  }
}
