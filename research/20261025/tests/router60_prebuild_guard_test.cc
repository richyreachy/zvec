#include <cassert>
#include <cstdlib>
#include <vector>
#include "algorithm/quantized_graph/quantized_graph.h"
using namespace zvec::core;
struct PoisonCodec {
  static constexpr size_t kBatchSize = 3;
  using Query = std::vector<float>;
  size_t dim;
  size_t encoded_dim() const {
    return dim;
  }
  size_t batch_bytes() const {
    return dim * 3;
  }
  void transform(const void *, std::vector<float> &) const {
    std::abort();
  }
  void encode(const float *, const float *, size_t, char *) const {
    std::abort();
  }
};
struct PoisonGraph {
  struct Vector {
    const void *data() const {
      std::abort();
    }
  };
  bool valid(uint32_t) const {
    std::abort();
  }
  int get_vector(uint32_t, Vector &) const {
    std::abort();
  }
  std::vector<uint32_t> neighbors(uint32_t) const {
    std::abort();
  }
};
int main() {
  for (size_t dim : {0, 384, 768, 1024})
    for (bool cosine : {false, true}) {
      if (dim == 768 && cosine) continue;
      QuantizedGraph<PoisonCodec> graph(dim, PoisonCodec{dim}, 32, dim, cosine);
      assert(graph.prebuild(PoisonGraph{}, 1, 1) == IndexError_Mismatch);
    }
  std::puts(
      "projected-router incompatible prebuild checks=7 passed before "
      "graph/codec access");
}
