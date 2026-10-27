#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <map>
#include <random>
#include "algorithm/quantized_graph/quantized_graph_bucket.h"
int main() {
  zvec::core::QuantizedGraphBucket q;
  std::mt19937 gen(92187);
  size_t checks = 0;
  for (int round = 0; round < 100; ++round) {
    q.reset(800);
    float scale = round % 2 ? 0.5f : 20.f;
    q.set_scale(scale);
    std::map<uint32_t, float> ref;
    for (int i = 0; i < 4000; ++i) {
      if (ref.empty() || gen() % 3) {
        float d = (int(gen() % 10000) - 1000) * scale / 1000;
        uint32_t id = uint32_t(i) | (round % 3 == 0 ? 0x80000000U : 0);
        q.insert(id, d);
        ref[id] = d;
      } else {
        assert(q.has_next());
        float low = q.next_distance();
        uint32_t id = q.next_id_at(0);
        assert(q.pop() == id && ref.count(id));
        float v = ref.at(id);
        for (const auto &[key, d] : ref)
          assert(low <= d + std::abs(d) * 1e-6f + 1e-7f);
        float smallest = std::numeric_limits<float>::infinity();
        for (const auto &[key, d] : ref) smallest = std::min(smallest, d);
        if (low >= 0 && low < 2046 * scale * QG_BUCKET_WIDTH)
          assert(v - smallest <= scale * QG_BUCKET_WIDTH + 1e-5f);
        ref.erase(id);
        ++checks;
      }
    }
    while (!ref.empty()) {
      uint32_t id = q.pop();
      assert(ref.erase(id) == 1);
      ++checks;
    }
    assert(!q.has_next());
  }
  std::cout << "bucket property checks=" << checks << " passed\n";
}
