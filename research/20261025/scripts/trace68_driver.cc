#include <sys/resource.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include "vecs_reader.h"


namespace ci = zvec::core_interface;
using Clock = std::chrono::steady_clock;
constexpr int DIM = 768;
int TOPK = 10, M = 50, EFC = 500, DEGREE = 32, BUILD_THREADS = 8;
const size_t N =
    std::getenv("BENCH_N") ? std::stoul(std::getenv("BENCH_N")) : 1000000;
const std::string DATA = std::getenv("BENCH_DATA")
                             ? std::getenv("BENCH_DATA")
                             : "/home/xingrui/bench/dataset/cohere/1m/";
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void check(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
long rss_kib() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line))
    if (line.rfind("VmRSS:", 0) == 0) return std::stol(line.substr(6));
  return 0;
}
long peak_kib() {
  rusage r{};
  getrusage(RUSAGE_SELF, &r);
  return r.ru_maxrss;
}
ci::HNSWIndexParam params(const std::string &method) {
  ci::HNSWIndexParam p(ci::MetricType::kCosine, DIM, M, EFC);
  p.data_type = ci::DataType::DT_FP32;
  p.use_id_map = true;
  p.quantized_graph_codec = method == "hnsw" ? "" : method;
  return p;
}
// Expose the existing core degree parameter without changing the library.
// The public HNSW parameter currently only exposes the enable/disable switch.
class TunableHNSW : public ci::HNSWIndex {
 public:
  int initialize(const ci::HNSWIndexParam &p) {
    proxima_index_params_.set("proxima.hnsw.quantized_graph.degree",
                              uint32_t(DEGREE));
    return init(p);
  }
};
ci::Index::Pointer make_index(const std::string &method) {
  auto index = std::make_shared<TunableHNSW>();
  check(index->initialize(params(method)) == 0, "initialize index");
  return index;
}
struct Queries {
  std::vector<std::vector<float>> vectors;
  std::vector<std::vector<uint64_t>> gt;
  Queries() {
    std::ifstream in(DATA + "cohere_test_vector_1m.1000.new.txt");
    std::string line;
    while (std::getline(in, line)) {
      auto sep = line.find(';');
      check(sep != std::string::npos, "query separator");
      check(std::stoull(line.substr(0, sep)) == vectors.size(),
            "query IDs must be sequential");
      std::istringstream row(line.substr(sep + 1));
      std::vector<float> v;
      float x;
      while (row >> x) {
        check(std::isfinite(x), "nonfinite query");
        v.push_back(x);
      }
      check(v.size() == DIM, "query dimension");
      vectors.push_back(std::move(v));
    }
    std::ifstream truth(DATA + "neighbors.txt");
    while (std::getline(truth, line)) {
      auto sep = line.find(';');
      check(sep != std::string::npos, "ground truth separator");
      check(std::stoull(line.substr(0, sep)) == gt.size(),
            "GT query IDs must be sequential");
      std::istringstream row(line.substr(sep + 1));
      std::vector<uint64_t> v;
      uint64_t x;
      while (row >> x) {
        check(x < N, "GT ID outside base range");
        v.push_back(x);
      }
      check(v.size() >= TOPK, "too few GT neighbors");
      check(std::unordered_set<uint64_t>(v.begin(), v.end()).size() == v.size(),
            "duplicate GT IDs");
      gt.push_back(std::move(v));
    }
    check(vectors.size() == 1000 && vectors.size() == gt.size(),
          "expected 1000 queries and GT rows");
  }
};
void validate_base(zvec::core::VecsReader &base) {
  check(base.num_vecs() == N && base.index_meta().dimension() == DIM &&
            base.index_meta().data_type() == zvec::core::IndexMeta::DT_FP32,
        "expected 1M x 768 FP32");
  std::vector<bool> seen(base.num_vecs(), false);
  for (size_t i = 0; i < base.num_vecs(); ++i) {
    auto id = base.get_key(i);
    check(id < base.num_vecs() && !seen[id],
          "base IDs must be a permutation of 0..999999");
    seen[id] = true;
  }
}
void build(const std::string &path) {
  check(!std::filesystem::exists(path), "refusing to overwrite existing index");
  auto start = Clock::now();
  zvec::core::VecsReader base;
  check(base.load(DATA + "cohere_train_vector_1m.new.zvec.vecs"), "load base");
  validate_base(base);
  auto index = make_index("hnsw");
  check(bool(index), "create index");
  ci::StorageOptions storage;
  storage.type = ci::StorageOptions::StorageType::kMMAP;
  storage.create_new = true;
  check(index->open(path, storage) == 0, "open new index");
  auto insert_start = Clock::now();
  std::atomic<size_t> done{0};
  std::atomic<int> errors{0};
  std::vector<std::thread> workers;
  for (size_t t = 0; t < size_t(BUILD_THREADS); ++t)
    workers.emplace_back([&, t] {
      for (size_t i = t; i < base.num_vecs() && !errors.load();
           i += BUILD_THREADS) {
        ci::VectorData v{ci::DenseVector{base.get_vector(i)}};
        if (index->add(v, base.get_key(i)) != 0) {
          ++errors;
          break;
        }
        auto count = ++done;
        if (count % 10000 == 0)
          std::cout << "BUILD_PROGRESS " << count
                    << " seconds=" << seconds(insert_start) << std::endl;
      }
    });
  for (auto &w : workers) w.join();
  check(errors == 0 && done == base.num_vecs(), "build errors");
  double insert_s = seconds(insert_start);
  check(index->get_doc_count() == base.num_vecs(), "built count");
  check(index->flush() == 0, "flush index");
  check(index->close() == 0, "close index");
  std::cout << "BUILD_RESULT count=" << done << " insert_seconds=" << insert_s
            << " total_seconds=" << seconds(start)
            << " peak_rss_kib=" << peak_kib() << std::endl;
}
void verify() {
  Queries q;
  zvec::core::VecsReader base;
  check(base.load(DATA + "cohere_train_vector_1m.new.zvec.vecs"), "load base");
  validate_base(base);
  std::vector<double> norms(base.num_vecs());
  for (size_t i = 0; i < base.num_vecs(); ++i) {
    auto v = static_cast<const float *>(base.get_vector(i));
    double norm = 0;
    for (int d = 0; d < DIM; ++d) {
      check(std::isfinite(v[d]), "nonfinite base");
      norm += double(v[d]) * v[d];
    }
    check(norm > 0, "zero base vector");
    norms[i] = std::sqrt(norm);
  }
  size_t total_hits = 0;
  for (size_t qi = 0; qi < q.vectors.size(); qi += 100) {
    std::priority_queue<std::pair<double, uint64_t>> best;
    for (size_t i = 0; i < base.num_vecs(); ++i) {
      auto v = static_cast<const float *>(base.get_vector(i));
      double dot = 0;
      for (int d = 0; d < DIM; ++d) dot += double(v[d]) * q.vectors[qi][d];
      // Query norm is a positive constant and does not affect the ordering.
      auto candidate = std::make_pair(-dot / norms[i], base.get_key(i));
      if (best.size() < TOPK)
        best.push(candidate);
      else if (candidate < best.top()) {
        best.pop();
        best.push(candidate);
      }
    }
    size_t hits = 0;
    while (!best.empty()) {
      hits += std::find(q.gt[qi].begin(), q.gt[qi].begin() + TOPK,
                        best.top().second) != q.gt[qi].begin() + TOPK;
      best.pop();
    }
    total_hits += hits;
    std::cout << "GT_CHECK query=" << qi << " hits=" << hits << "/" << TOPK
              << std::endl;
  }
  check(total_hits == 10 * size_t(TOPK),
        "supplied GT differs from exact cosine on verification sample");
  std::cout << "VALIDATED base=N dimension=768 queries=1000 exact_sample_hits="
            << total_hits << "/" << 10 * TOPK << std::endl;
}
void search(const std::string &path, const std::string &method, int, int,
            const std::string &prefix) {
  const size_t count = std::getenv("TRACE68_QUERY_COUNT")
                           ? std::stoul(std::getenv("TRACE68_QUERY_COUNT"))
                           : 4096;
  check(count > 0 && count <= N, "query count");
  auto index = make_index(method);
  ci::StorageOptions storage;
  storage.type = ci::StorageOptions::StorageType::kMMAP;
  storage.read_only = true;
  check(index->open(path, storage) == 0, "open trace index");
  check(index->get_doc_count() == N, "count");
  std::ifstream graph(std::getenv("TRACE68_GRAPH_FILE"), std::ios::binary),
      vf(std::getenv("TRACE68_VECTOR_FILE"), std::ios::binary);
  uint32_t gh[8], vh[4];
  graph.read(reinterpret_cast<char *>(gh), 32);
  vf.read(reinterpret_cast<char *>(vh), 16);
  check(graph.good() && vf.good() && gh[0] == 0x51474764 && gh[2] == N &&
            gh[3] == DIM && vh[0] == 0x51475664 && vh[2] == N && vh[3] == DIM,
        "headers");
  std::vector<uint32_t> inverse(N, UINT32_MAX);
  std::vector<uint64_t> keys(N);
  for (size_t i = 0; i < N; ++i) {
    uint32_t degree, adj[32];
    graph.read(reinterpret_cast<char *>(&keys[i]), 8);
    graph.read(reinterpret_cast<char *>(&degree), 4);
    graph.read(reinterpret_cast<char *>(adj), 128);
    check(graph.good() && keys[i] < N && inverse[keys[i]] == UINT32_MAX,
          "keys");
    inverse[keys[i]] = i;
  }
  check(graph.peek() == EOF, "graph size");
  std::vector<float> vectors(N * DIM);
  vf.read(reinterpret_cast<char *>(vectors.data()), vectors.size() * 4);
  check(vf.good() && vf.peek() == EOF, "vector size");
  struct Record {
    uint32_t query;
    uint32_t ids[10];
    float distances[10];
  };
  static_assert(sizeof(Record) == 84, "layout");
  std::vector<Record> output(count);
  auto param = std::make_shared<ci::HNSWQueryParam>();
  param->ef_search = 800;
  param->topk = 10;
  size_t self_hits = 0;
  for (size_t i = 0; i < count; ++i) {
    uint32_t id = (uint64_t(i) * 2654435761ULL + 6819) % N;
    auto &r = output[i];
    r.query = id;
    ci::SearchResult result;
    check(index->search({ci::DenseVector{vectors.data() + size_t(id) * DIM}},
                        param, &result) == 0,
          "trace search");
    check(result.doc_list_.size() == 10, "result count");
    size_t j = 0;
    std::unordered_set<uint32_t> seen;
    for (const auto &doc : result.doc_list_) {
      check(doc.key() < N && std::isfinite(doc.score()), "result");
      uint32_t target = inverse[doc.key()];
      check(seen.insert(target).second, "duplicate result");
      r.ids[j] = target;
      r.distances[j] = doc.score();
      self_hits += target == id;
      ++j;
    }
  }
  FILE *out = std::fopen((prefix + "_results.bin").c_str(), "wbx");
  check(out, "exclusive output");
  uint32_t h[8] = {0x51474368,      1,  uint32_t(N), DIM,
                   uint32_t(count), 10, 800,         6819};
  check(std::fwrite(h, sizeof(h), 1, out) == 1 &&
            std::fwrite(output.data(), sizeof(Record), count, out) == count,
        "output");
  check(std::fclose(out) == 0, "close output");
  check(index->close() == 0, "close index");
  std::cout << "TRACE68 COMPLETE queries=" << count
            << " self_hits=" << self_hits
            << "; database proxies include self, no GT or retrieval QPS"
            << std::endl;
}

int main(int argc, char **argv) {
  try {
    zvec::ailego::LoggerBroker::SetLevel(zvec::ailego::Logger::LEVEL_INFO);
    check(argc >= 2,
          "usage: comparison verify | build INDEX | search INDEX QG ROUND "
          "SECONDS PREFIX");
    if (std::getenv("BENCH_TOPK")) TOPK = std::stoi(std::getenv("BENCH_TOPK"));
    if (std::getenv("BENCH_M")) M = std::stoi(std::getenv("BENCH_M"));
    if (std::getenv("BENCH_EFC")) EFC = std::stoi(std::getenv("BENCH_EFC"));
    if (std::getenv("BENCH_DEGREE"))
      DEGREE = std::stoi(std::getenv("BENCH_DEGREE"));
    if (std::getenv("BENCH_BUILD_THREADS"))
      BUILD_THREADS = std::stoi(std::getenv("BENCH_BUILD_THREADS"));
    check(TOPK == 10 || TOPK == 50 || TOPK == 100, "supported topk 10/50/100");
    check(DEGREE == 32 || DEGREE == 64, "supported degree 32/64");
    std::cout << "CONFIG topk=" << TOPK << " m=" << M
              << " efConstruction=" << EFC << " degree=" << DEGREE
              << " build_threads=" << BUILD_THREADS << std::endl;
    std::string mode = argv[1];
    if (mode == "verify")
      verify();
    else if (mode == "build" && argc == 3)
      build(argv[2]);
    else if (mode == "search" && argc == 7)
      search(argv[2], std::string(argv[3]), std::stoi(argv[4]),
             std::stoi(argv[5]), argv[6]);
    else
      throw std::runtime_error("invalid command line");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAILED: " << e.what() << std::endl;
    return 1;
  }
}
