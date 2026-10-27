// Fixed-candidate arrival-order intervention on the frozen production bucket.
// No graph, query, learned score, ground truth, or retrieval timing is read.
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "core/algorithm/quantized_graph/quantized_graph_bucket.h"

#if defined(QG_FLAT_BUCKET) || defined(QG_QUEUE_STATS) || \
    defined(QG_BUCKET_CACHEMIN)
#error This fixture tests the ordinary production bucket only.
#endif

namespace {
using Bucket = zvec::core::QuantizedGraphBucket;
constexpr uint32_t entry = 0, a = 1, b = 2;
constexpr float entry_exact = 1.f, a_score = .5f, b_score = .501f;
void need(bool yes, const char *message) {
  if (!yes) throw std::runtime_error(message);
}
uint32_t bits(float value) {
  uint32_t out;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}
struct Result {
  std::array<uint32_t, 2> arrivals, pops;
  size_t bin;
  uint32_t lower_bits;
};
Result run(Bucket &bucket, const std::array<uint32_t, 2> &arrivals) {
  bucket.reset(64);
  need(!bucket.has_next() && bucket.width() == 1.f, "reset state");
  // QuantizedGraph::search seeds entry with max(), before its first expansion
  // computes the exact entry score and calls set_scale. Match that sequence.
  bucket.insert(entry, std::numeric_limits<float>::max());
  need(bucket.has_next() && bucket.next_id_at(0) == entry, "entry seed peek");
  need(bucket.pop() == entry && !bucket.has_next(),
       "entry pop before set_scale");
  bucket.set_scale(entry_exact);
  need(bits(bucket.width()) == bits(.016f), "entry-scaled width");
  const size_t bin = bucket.locate(a_score);
  need(bin == bucket.locate(b_score) && bin > 0 && bin < Bucket::N - 1,
       "fixed scores must share an ordinary bin");
  for (const auto id : arrivals) {
    need(id == a || id == b, "fixed candidate IDs only");
    bucket.insert(id, id == a ? a_score : b_score);
  }
  const auto lower = bits(bucket.next_distance());
  Result result{arrivals, {}, bin, lower};
  for (size_t i = 0; i < 2; ++i) {
    need(bucket.has_next() && bucket.first() == bin, "same bin remains first");
    need(bucket.next_id_at(0) == arrivals[1 - i], "peek must be last arrival");
    need(bits(bucket.next_distance()) == lower,
         "same lower bound for both candidates");
    result.pops[i] = bucket.pop();
    need(result.pops[i] == arrivals[1 - i], "pop must be last arrival");
  }
  need(!bucket.has_next(), "both candidates consumed");
  return result;
}
}  // namespace

int main() try {
  Bucket bucket;
  const Result ab = run(bucket, {a, b});
  const Result ba = run(bucket, {b, a});
  need(ab.bin == ba.bin && ab.lower_bits == ba.lower_bits,
       "intervention preserves bin and stop bound");
  need(ab.pops == std::array<uint32_t, 2>{b, a} &&
           ba.pops == std::array<uint32_t, 2>{a, b},
       "arrival reversal changes pop order");
  std::cout << "{\"passed\":true,\"production_bucket_used\":true,"
            << "\"entry_id\":0,\"entry_exact_bits\":" << bits(entry_exact)
            << ",\"width_bits\":" << bits(bucket.width())
            << ",\"fixed_candidates\":[{\"id\":1,\"score_bits\":"
            << bits(a_score) << "},{\"id\":2,\"score_bits\":" << bits(b_score)
            << "}],"
            << "\"shared_bucket_id\":" << ab.bin
            << ",\"shared_lower_bound_bits\":" << ab.lower_bits
            << ",\"arrival_ab\":[1,2],\"pop_ab\":[2,1],\"arrival_ba\":[2,1],"
               "\"pop_ba\":[1,2],"
            << "\"only_intervention_is_arrival_order\":true,\"same_candidate_"
               "ids_and_score_bits\":true,"
            << "\"entry_seed_popped_before_set_scale\":true,\"bucket_reused_"
               "after_reset\":true,"
            << "\"ground_truth_or_queries_read\":false,\"retrieval_or_qps_"
               "run\":false}\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "LIFO CONTROL FAILED " << error.what() << '\n';
  return 1;
}
