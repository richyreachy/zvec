#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "algorithm/quantized_graph/quantized_graph_integer_router.h"
using zvec::core::IntegerCenter;
using Router = zvec::core::QuantizedGraphIntegerRouter;
struct Graph {
  bool valid(uint32_t id) const {
    return id < 1000000;
  }
};
int main(int argc, char **argv) {
  assert(argc == 3);
  std::ifstream file(argv[1], std::ios::binary);
  file.seekg(20);
  std::vector<float> mean(768);
  file.read(reinterpret_cast<char *>(mean.data()), 3072);
  Router r;
  assert(r.load(argv[1], mean, Graph{}, 1000000));
  std::ifstream queryfile(argv[2]);
  std::string line;
  std::vector<IntegerCenter::Query> queries;
  while (std::getline(queryfile, line)) {
    std::istringstream row(line.substr(line.find(';') + 1));
    std::vector<float> q;
    float x;
    while (row >> x) q.push_back(x);
    assert(q.size() == 768);
    double norm = 0;
    for (float v : q) norm += double(v) * v;
    for (float &v : q) v /= std::sqrt(norm);
    queries.emplace_back();
    assert(IntegerCenter::prepare(q.data(), mean.data(), 768, queries.back()));
  }
  assert(queries.size() == 1000);
  std::cout << "query_id,beam,entry,groups\n";
  for (size_t beam : {1, 2, 4})
    for (size_t i = 0; i < queries.size(); ++i) {
      size_t groups = 0;
      uint32_t entry = r.choose(queries[i], beam, &groups);
      std::cout << i << ',' << beam << ',' << entry << ',' << groups << '\n';
    }
}
