#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "algorithm/quantized_graph/quantized_graph_integer_router.h"
#include "turbo/preprocessor/symphony_qg_rotator/symphony_qg_rotator.h"
using Router = zvec::core::QuantizedGraphIntegerRouter;
using Query = zvec::core::IntegerCenter::Query;
struct Graph {
  bool valid(uint32_t id) const {
    return id < 1000000;
  }
};
int main(int argc, char **argv) {
  assert(argc == 4);
  std::ifstream mf(argv[1], std::ios::binary);
  mf.seekg(20);
  std::vector<float> mean(768), rm;
  mf.read(reinterpret_cast<char *>(mean.data()), 3072);
  zvec::turbo::SymphonyQGRotation rotate(768);
  rotate.rotate(mean.data(), rm);
  Router router;
  assert(router.load(argv[1], mean, Graph{}, 1000000, &rm));
  std::vector<std::vector<float>> queries;
  std::ifstream qf(argv[2]);
  std::string line;
  while (std::getline(qf, line)) {
    std::istringstream row(line.substr(line.find(';') + 1));
    std::vector<float> q;
    float x;
    while (row >> x) q.push_back(x);
    assert(q.size() == 768);
    double norm = 0;
    for (float v : q) norm += double(v) * v;
    for (float &v : q) v /= std::sqrt(norm);
    queries.emplace_back();
    rotate.rotate(q.data(), queries.back());
  }
  assert(queries.size() == 1000);
  Query codes;
  uint64_t checksum = 0;
  size_t total = 0;
  std::ofstream trace(argv[3]);
  trace << "query_id,dimension,entry,groups\n";
  for (size_t i = 0; i < queries.size(); ++i) {
    assert(router.prepare_projected(queries[i].data(), codes));
    size_t groups;
    auto id = router.choose(codes, 1, &groups);
    total += groups;
    checksum += id;
    trace << i << ',' << Router::kDim << ',' << id << ',' << groups << '\n';
  }
  trace.close();
  auto start = std::chrono::steady_clock::now();
  for (size_t repeat = 0; repeat < 50; ++repeat)
    for (const auto &q : queries) {
      assert(router.prepare_projected(q.data(), codes));
      checksum += router.choose(codes, 1);
    }
  auto finish = std::chrono::steady_clock::now();
  double ns =
      std::chrono::duration<double, std::nano>(finish - start).count() / 50000;
  std::cout << "dimension,mean_groups,projected_encode_and_route_ns,checksum\n"
            << Router::kDim << ',' << double(total) / 1000 << ',' << ns << ','
            << checksum << '\n';
}
