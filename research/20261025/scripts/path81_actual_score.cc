// Phase81 actual-code score exporter. DRAFT: not compiled or executed.
// Preserve the accepted phase80 production preparation/kernel bindings
// verbatim.
#define main phase80_reference_program_main
#include "path80_score.cc"
#undef main
#include <memory>
#include <set>
#include "core/algorithm/quantized_graph/quantized_graph_bucket.h"
#include "candidate81_codec.h"

namespace phase81 {
constexpr uint32_t Q = 4096, NC = 20, CAND = 137;
using Bucket = zvec::core::QuantizedGraphBucket;
struct StateInput {
  uint32_t query, internal_id, center, step;
};
static_assert(sizeof(StateInput) == 16, "state input layout");
struct Prefix {
  uint32_t query, kind, step, source;
};
static_assert(sizeof(Prefix) == 16, "metadata only prefix");
struct Boundary {
  uint32_t at_bits, previous_bits, next_bits;
  uint32_t previous_bucket, at_bucket, next_bucket;
  uint32_t flags, reserved;  // 1=finite, 2=previous checked, 4=next checked.
};
static_assert(sizeof(Boundary) == 32, "boundary layout");
struct Lane {
  uint32_t target, native_bits, truth_bits, native_bucket;
  uint32_t eligible_unexpanded, valid_candidates, reserved0, reserved1;
  Boundary lower, upper;
};
static_assert(sizeof(Lane) == 96, "lane layout");
struct StateHeader {
  uint32_t query, internal_id, center, step, degree, entry_bits, width_bits,
      center_bits;
  uint64_t native_event_index;
  uint32_t center_row, gate_flags, entry_id, phase, query_record_index,
      score_model_count;
};
static_assert(sizeof(StateHeader) == 64, "state header layout");
struct StateOutput {
  StateHeader header;
  std::array<Lane, K> lanes;
};
static_assert(sizeof(StateOutput) == 3136, "state output layout");
struct QueryOutput {
  uint32_t query, internal_id, entry_bits, width_bits;
  std::array<float, D + 1> prepared;
  std::array<float, D> rotated;
};
static_assert(sizeof(QueryOutput) == 6164, "query output layout");
struct ReadyQuery {
  QueryOutput output{};
  Codec::Query lookup;
  uint32_t output_index = 0;
};
struct Center {
  uint32_t id;
  std::array<uint32_t, K> neighbors;
  std::array<char, B> native;
};

float unbits(uint32_t value) {
  float x;
  std::memcpy(&x, &value, 4);
  return x;
}
float ordered_float(uint32_t value) {
  return unbits(value & 0x80000000U ? value ^ 0x80000000U : ~value);
}
Boundary transition(const Bucket &bucket, uint32_t target_bucket) {
  need(target_bucket >= 1 && target_bucket < Bucket::N,
       "finite transition target");
  need(std::isfinite(bucket.width()) && bucket.width() > 0,
       "finite positive transition width");
  // Total order on finite IEEE floats, including -0 before +0. No
  // NaNs/infinities.
  uint32_t low = 0x00800000U, high = 0xff7fffffU;
  if (bucket.locate(ordered_float(high)) < target_bucket) return {};
  need(bucket.locate(ordered_float(low)) < target_bucket,
       "transition bracketing");
  while (low < high) {
    const uint32_t middle = low + uint32_t((uint64_t(high) - low) / 2);
    if (bucket.locate(ordered_float(middle)) >= target_bucket)
      high = middle;
    else
      low = middle + 1;
  }
  const float at = ordered_float(low);
  const float previous =
      std::nextafter(at, -std::numeric_limits<float>::infinity());
  const float next = std::nextafter(at, std::numeric_limits<float>::infinity());
  need(std::isfinite(at) && std::isfinite(previous) && !std::isnan(next),
       "finite transition and predecessor; next may be +infinity");
  need(bucket.locate(previous) < target_bucket &&
           bucket.locate(at) >= target_bucket &&
           bucket.locate(next) >= target_bucket,
       "locate transition nextafter control");
  need(bucket.locate(at) == target_bucket,
       "transition skipped requested bucket");
  need(bits(previous) == bits(ordered_float(low - 1)),
       "ordered predecessor check");
  // Do not claim at == float(target_bucket-1)*width. Division rounding decides.
  return {bits(at),
          bits(previous),
          bits(next),
          uint32_t(bucket.locate(previous)),
          uint32_t(bucket.locate(at)),
          uint32_t(bucket.locate(next)),
          7,
          0};
}

struct Output {
  FILE *file = nullptr;
  explicit Output(const std::string &path) {
    file = std::fopen(path.c_str(), "wbx");
    need(file, "exclusive output: " + path);
  }
  void write(const void *p, size_t size) {
    need(std::fwrite(p, 1, size, file) == size, "write output");
  }
  void close() {
    FILE *f = file;
    file = nullptr;
    need(std::fclose(f) == 0, "close output");
  }
  ~Output() {
    if (file) std::fclose(file);
  }
};
std::array<uint32_t, 16> header(uint32_t magic, uint32_t record_bytes,
                                uint32_t records, uint32_t models,
                                uint32_t phase) {
  return {{magic, 1, 64, record_bytes, D, K, records, models, phase, 0, 0, 0, 0,
           0, 0, 0}};
}
}  // namespace phase81

int main(int argc, char **argv) try {
  using namespace phase81;
  need(argc == 11,
       "usage: path81_actual_score training|validation events graph raw "
       "rotated candidate_dir states.bin reference_results.bin "
       "selected_model_dir_or_dash exclusive_output_dir");
  const std::string mode = argv[1], candidate_dir = argv[6],
                    selected_dir = argv[9], out = argv[10];
  need(mode == "training" || mode == "validation", "explicit partition mode");
  const bool validation = mode == "validation";
  need(validation ? selected_dir != "-" : selected_dir == "-",
       "validation requires frozen selected model directory only");
  need(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4,
       "IEEE FP32");
  const uint32_t endian = 1;
  need(*reinterpret_cast<const char *>(&endian) == 1, "little endian");
  need(QG_BUCKET_WIDTH == .016f && Bucket::N == 2048,
       "frozen production bucket macros");
  Map events(argv[2]), graph(argv[3]), raw(argv[4]), rotated(argv[5]),
      states(argv[7]), mapping(argv[8]);
  Map candidates(candidate_dir + "/candidates.bin"),
      ids(candidate_dir + "/centers.u32"),
      neighbors(candidate_dir + "/neighbors.u32"),
      valid(candidate_dir + "/valid.u8"),
      native_blocks(candidate_dir + "/native_blocks.bin"),
      compact(candidate_dir + "/candidate_compact.u8");
  const uint32_t trace_header[4] = {0x51474d68, 1, 32, D};
  need(events.size >= 16 && (events.size - 16) % 32 == 0 &&
           std::memcmp(events.p, trace_header, 16) == 0,
       "native trace header");
  need(graph.size == 32 + 1000000ULL * 140 &&
           take<uint32_t>(graph.p) == 0x51474764 &&
           take<uint32_t>(graph.p + 4) == 1 &&
           take<uint32_t>(graph.p + 8) == 1000000 &&
           take<uint32_t>(graph.p + 12) == D &&
           take<uint32_t>(graph.p + 16) == K,
       "frozen graph layout");
  need(raw.size == 16 + 1000000ULL * D * 4 &&
           rotated.size == 1000000ULL * D * 4 &&
           take<uint32_t>(raw.p) == 0x51475664 &&
           take<uint32_t>(raw.p + 8) == 1000000 &&
           take<uint32_t>(raw.p + 12) == D,
       "frozen database vectors");
  const uint32_t bank_header[16] = {0x51474381, 1, 64, D, NC, CAND, K, B,
                                    0,          0, 0,  0, 0,  0,    0, 0};
  need(candidates.size == 64 + size_t(NC) * CAND * B &&
           std::memcmp(candidates.p, bank_header, 64) == 0,
       "candidate bank layout");
  need(ids.size == NC * 4 && neighbors.size == NC * K * 4 &&
           valid.size == NC * CAND * K && native_blocks.size == NC * B &&
           compact.size == NC * CAND * K * (D / 8),
       "candidate sidecar dimensions");
  const uint32_t mapping_header[8] = {0x51474368, 1,  1000000, D,
                                      Q,          10, 800,     6819};
  need(mapping.size == 32 + Q * 84 &&
           std::memcmp(mapping.p, mapping_header, 32) == 0,
       "DB driver query mapping header");
  need(states.size >= 32 && take<uint32_t>(states.p) == 0x51475381 &&
           take<uint32_t>(states.p + 4) == 1 &&
           take<uint32_t>(states.p + 8) == 16 &&
           take<uint32_t>(states.p + 16) == uint32_t(validation) &&
           take<uint32_t>(states.p + 20) == Q &&
           take<uint32_t>(states.p + 24) == D &&
           take<uint32_t>(states.p + 28) == 0,
       "partition states header");
  const uint32_t state_count = take<uint32_t>(states.p + 12);
  need(state_count == (validation ? 2183U : 6625U) &&
           states.size == 32 + size_t(state_count) * 16,
       "full frozen selected-state scope; no score-based subset");
  const uint32_t score_models = validation ? 3 : CAND;
  std::unique_ptr<Map> selected, selections;
  if (validation) {
    selected = std::make_unique<Map>(selected_dir + "/selected_models.bin");
    selections = std::make_unique<Map>(selected_dir + "/selected_indices.u32");
    const uint32_t selected_header[16] = {0x51474d81, 1, 64, D, NC, 2, K, B,
                                          0,          0, 0,  0, 0,  0, 0, 0};
    need(selected->size == 64 + size_t(2) * NC * B &&
             std::memcmp(selected->p, selected_header, 64) == 0 &&
             selections->size == 2 * NC * K * 4,
         "two frozen selected model layouts");
  }
  const auto graph_row = [&](uint32_t id) {
    need(id < 1000000, "internal ID range");
    return graph.p + 32 + size_t(id) * 140;
  };
  const auto raw_vector = [&](uint32_t id) {
    need(id < 1000000, "raw internal ID range");
    return reinterpret_cast<const float *>(raw.p + 16) + size_t(id) * D;
  };
  Codec codec(D, true);
  need(codec.batch_bytes() == B && codec.encoded_dim() == D &&
           codec.quantized_dim() == D,
       "actual codec dimensions");
  std::set<uint32_t> rotation_verified;
  std::vector<float> rotation;
  const auto rotated_vector = [&](uint32_t id) {
    const auto *reference =
        reinterpret_cast<const float *>(rotated.p) + size_t(id) * D;
    if (rotation_verified.insert(id).second) {
      codec.transform(raw_vector(id), rotation);
      need(rotation.size() == D &&
               std::memcmp(rotation.data(), reference, D * 4) == 0,
           "actual base rotation differs from frozen bank");
    }
    return reference;
  };
  std::array<Center, NC> center_bank{};
  std::unordered_map<uint32_t, uint32_t> center_rows;
  for (uint32_t ci = 0; ci < NC; ++ci) {
    auto &c = center_bank[ci];
    c.id = take<uint32_t>(ids.p + ci * 4);
    need(center_rows.emplace(c.id, ci).second, "duplicate candidate center");
    const char *g = graph_row(c.id);
    need(take<uint32_t>(g + 8) == K, "selected center full degree");
    std::vector<float> adjacent(K * D);
    for (uint32_t lane = 0; lane < K; ++lane) {
      c.neighbors[lane] = take<uint32_t>(neighbors.p + (ci * K + lane) * 4);
      need(c.neighbors[lane] == take<uint32_t>(g + 12 + lane * 4),
           "candidate neighbor lane order");
      std::copy_n(rotated_vector(c.neighbors[lane]), D,
                  adjacent.data() + lane * D);
    }
    codec.encode(rotated_vector(c.id), adjacent.data(), K, c.native.data());
    need(std::memcmp(c.native.data(), native_blocks.p + ci * B, B) == 0 &&
             std::memcmp(c.native.data(),
                         candidates.p + 64 + size_t(ci) * CAND * B, B) == 0,
         "candidate0/native_blocks/actual codec byte identity");
    for (uint32_t lane = 0; lane < K; ++lane)
      need(uint8_t(valid.p[(ci * CAND) * K + lane]) == 1,
           "native candidate must be valid");
    for (uint32_t model = 0; validation && model < 2; ++model) {
      std::array<uint32_t, K> indices{};
      for (uint32_t lane = 0; lane < K; ++lane)
        indices[lane] =
            take<uint32_t>(selections->p + ((model * NC + ci) * K + lane) * 4);
      // Geometry-only helper reassembles lanes; never scores unselected
      // candidates.
      need(candidate81::verify_selected_block(
               candidates.p + 64 + size_t(ci) * CAND * B,
               reinterpret_cast<const uint8_t *>(compact.p + size_t(ci) * CAND *
                                                                 K * (D / 8)),
               reinterpret_cast<const uint8_t *>(valid.p +
                                                 size_t(ci) * CAND * K),
               indices.data(), selected->p + 64 + (model * NC + ci) * B),
           "selected frozen model differs from candidate lane assembly");
    }
  }
  std::vector<StateInput> requests(state_count);
  std::memcpy(requests.data(), states.p + 32, state_count * sizeof(StateInput));
  std::array<bool, Q> used_query{};
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> wanted;
  std::pair<uint32_t, uint32_t> previous{0, 0};
  for (uint32_t i = 0; i < state_count; ++i) {
    const auto &r = requests[i];
    need(r.query < Q &&
             r.internal_id ==
                 (uint64_t(r.query) * 2654435761ULL + 6819) % 1000000 &&
             r.internal_id == take<uint32_t>(mapping.p + 32 + r.query * 84),
         "production DB query input mapping");
    need(r.step >= 1 && r.step <= 8 && center_rows.count(r.center),
         "frozen early center scope");
    const auto &c = center_bank[center_rows.at(r.center)];
    need(r.internal_id != r.center &&
             std::find(c.neighbors.begin(), c.neighbors.end(), r.internal_id) ==
                 c.neighbors.end(),
         "query=self or neighbor exclusion");
    const auto order = std::make_pair(r.query, r.step);
    need(i == 0 || previous < order, "state query/step order");
    previous = order;
    need(wanted.emplace(std::make_pair(r.query, r.center), i).second,
         "duplicate selected state");
    used_query[r.query] = true;
  }
  // Index full trace using metadata prefixes only. Never decode another
  // partition's score.
  std::vector<uint64_t> positions(state_count, UINT64_MAX);
  std::vector<std::array<uint8_t, K>> eligibility(state_count);
  std::array<uint64_t, Q> entry_positions{};
  entry_positions.fill(UINT64_MAX);
  size_t pos = 16;
  for (uint32_t qi = 0; qi < Q; ++qi) {
    uint32_t step = 0;
    std::set<uint32_t> expanded;
    while (true) {
      need(pos + 32 <= events.size, "trace metadata bounds");
      const Prefix e = take<Prefix>(events.p + pos);
      need(e.query == qi, "trace serial order");
      if (e.kind == 3) {
        need(e.step == step, "trace final step");
        pos += 32;
        break;
      }
      need(e.kind == 1 && e.step == ++step && expanded.insert(e.source).second,
           "native center metadata");
      const char *g = graph_row(e.source);
      const uint32_t degree = take<uint32_t>(g + 8);
      need(degree <= K, "graph degree during metadata walk");
      if (step == 1 && used_query[qi]) entry_positions[qi] = pos;
      const auto found = wanted.find({qi, e.source});
      if (found != wanted.end()) {
        const uint32_t i = found->second;
        need(requests[i].step == step && degree == K &&
                 positions[i] == UINT64_MAX,
             "selected state trace location");
        positions[i] = pos;
        for (uint32_t lane = 0; lane < K; ++lane)
          eligibility[i][lane] =
              !expanded.count(take<uint32_t>(g + 12 + lane * 4));
      }
      pos += (size_t(degree) + 1) * 32;
    }
  }
  need(pos == events.size, "complete native metadata walk");
  ProductionQuery production;
  production.initialize();
  auto reformer = core::IndexFactory::CreateReformer("CosineNormalizeReformer");
  const zvec::ailego::Params empty;
  need(reformer && reformer->init(empty) == 0,
       "independent production reformer");
  std::array<std::unique_ptr<ReadyQuery>, Q> query_bank;
  std::vector<QueryOutput> query_records;
  const uint32_t entry_id = take<uint32_t>(graph.p + 20);
  const auto check_bits = [&](float actual, float expected, const char *what) {
    need(std::isfinite(actual) && bits(actual) == bits(expected),
         std::string(what) + " native bit mismatch");
  };
  for (uint32_t qi = 0; qi < Q; ++qi) {
    if (!used_query[qi]) continue;
    need(entry_positions[qi] != UINT64_MAX,
         "entry position for active partition");
    auto query = std::make_unique<ReadyQuery>();
    const uint32_t internal =
        uint32_t((uint64_t(qi) * 2654435761ULL + 6819) % 1000000);
    std::string prepared, independent;
    const void *data = nullptr;
    core::IndexQueryMeta meta, other_meta;
    production.prepare(raw_vector(internal), prepared, data, meta);
    need(reformer->transform(raw_vector(internal),
                             core::IndexQueryMeta(core::IndexMeta::DT_FP32, D),
                             &independent, &other_meta) == 0 &&
             prepared == independent,
         "DB driver production query preprocessing");
    std::copy_n(static_cast<const float *>(data), D + 1,
                query->output.prepared.data());
    codec.prepare_query(query->output.prepared.data(), query->lookup);
    need(query->lookup.rotated.size() == D, "actual query rotation");
    std::copy_n(query->lookup.rotated.data(), D, query->output.rotated.data());
    const Event entry = take<Event>(events.p + entry_positions[qi]);
    need(entry.kind == 1 && entry.step == 1 && entry.source == entry_id &&
             entry.target == entry_id,
         "native entry identity");
    const float exact = production.distance(raw_vector(entry_id),
                                            query->output.prepared.data());
    check_bits(exact, entry.exact, "entry exact");
    check_bits(exact, entry.estimate, "entry estimate");
    Bucket bucket;
    bucket.reset(0);
    bucket.set_scale(exact);
    need(std::isfinite(bucket.width()) && bucket.width() > 0,
         "finite positive actual width");
    need(bits(bucket.width()) == bits(std::max(exact, 1e-6f) * .016f),
         "actual entry FP32 bucket width");
    query->output.query = qi;
    query->output.internal_id = internal;
    query->output.entry_bits = bits(exact);
    query->output.width_bits = bits(bucket.width());
    query->output_index = uint32_t(query_records.size());
    query_records.push_back(query->output);
    query_bank[qi] = std::move(query);
  }
  std::vector<StateOutput> state_records(state_count);
  size_t native_edges = 0, boundary_checks = 0, absent_boundaries = 0;
  // Complete ALL selected native score/truth gates before scoring ANY
  // candidate.
  for (uint32_t i = 0; i < state_count; ++i) {
    need(positions[i] != UINT64_MAX,
         "selected state missing from native trace");
    const auto &r = requests[i];
    const uint32_t ci = center_rows.at(r.center);
    const auto &c = center_bank[ci];
    auto &query = *query_bank[r.query];
    const Event center_event = take<Event>(events.p + positions[i]);
    need(center_event.kind == 1 && center_event.query == r.query &&
             center_event.source == r.center &&
             center_event.target == r.center && center_event.step == r.step &&
             center_event.reserved == 0,
         "native center full event");
    const float center_exact =
        production.distance(raw_vector(c.id), query.output.prepared.data());
    check_bits(center_exact, center_event.estimate, "center estimate");
    check_bits(center_exact, center_event.exact, "center exact");
    std::array<float, K> scores{};
    codec.scan(c.native.data(), query.lookup, center_exact, scores.data());
    Bucket bucket;
    bucket.reset(0);
    bucket.set_scale(unbits(query.output.entry_bits));
    auto &record = state_records[i];
    record.header = {r.query,
                     r.internal_id,
                     r.center,
                     r.step,
                     K,
                     query.output.entry_bits,
                     query.output.width_bits,
                     bits(center_exact),
                     (positions[i] - 16) / 32,
                     ci,
                     7,
                     entry_id,
                     uint32_t(validation),
                     query.output_index,
                     score_models};
    for (uint32_t lane = 0; lane < K; ++lane) {
      const Event edge = take<Event>(events.p + positions[i] + (lane + 1) * 32);
      need(edge.kind == 2 && edge.query == r.query && edge.step == r.step &&
               edge.source == c.id && edge.target == c.neighbors[lane] &&
               edge.reserved == 0,
           "native edge lane order");
      check_bits(scores[lane], edge.estimate, "FastScan");
      const float truth = production.distance(raw_vector(edge.target),
                                              query.output.prepared.data());
      check_bits(truth, edge.exact, "Turbo exact edge");
      const uint32_t bin = uint32_t(bucket.locate(scores[lane]));
      uint32_t valid_count = 0;
      for (uint32_t candidate = 0; candidate < CAND; ++candidate) {
        const auto flag = uint8_t(valid.p[(ci * CAND + candidate) * K + lane]);
        need(flag <= 1, "candidate validity flag");
        valid_count += flag;
      }
      Lane value{};
      value.target = edge.target;
      value.native_bits = bits(scores[lane]);
      value.truth_bits = bits(truth);
      value.native_bucket = bin;
      value.eligible_unexpanded = eligibility[i][lane];
      value.valid_candidates = valid_count;
      if (bin > 0) value.lower = transition(bucket, bin);
      if (bin + 1 < Bucket::N) value.upper = transition(bucket, bin + 1);
      boundary_checks += bool(value.lower.flags) + bool(value.upper.flags);
      absent_boundaries += !value.lower.flags + !value.upper.flags;
      for (const Boundary *bound : {&value.lower, &value.upper}) {
        if (!bound->flags) continue;
        need((bucket.locate(truth) >= bound->at_bucket) ==
                 (truth >= unbits(bound->at_bits)),
             "actual exact score/right-inclusive transition equivalence");
      }
      record.lanes[lane] = value;
      ++native_edges;
    }
  }
  const auto state_header = header(0x51474181, sizeof(StateOutput), state_count,
                                   score_models, validation);
  Output state_file(out + "/states.bin");
  state_file.write(state_header.data(), 64);
  state_file.write(state_records.data(),
                   state_records.size() * sizeof(StateOutput));
  state_file.close();
  const auto query_header =
      header(0x51475181, sizeof(QueryOutput), uint32_t(query_records.size()), 0,
             validation);
  Output query_file(out + "/queries.bin");
  query_file.write(query_header.data(), 64);
  query_file.write(query_records.data(),
                   query_records.size() * sizeof(QueryOutput));
  query_file.close();
  const auto score_header =
      header(0x51474581, K * 4, state_count, score_models, validation);
  Output score_file(out + "/scores.f32");
  score_file.write(score_header.data(), 64);
  std::array<float, K> scores{};
  need(query_records.size() == (validation ? 1024 : 3072),
       "all fixed partition queries retained");
  for (uint32_t i = 0; i < state_count; ++i) {
    const auto &state = state_records[i];
    const uint32_t ci = state.header.center_row;
    auto &query = *query_bank[state.header.query];
    Bucket bucket;
    bucket.reset(0);
    bucket.set_scale(unbits(state.header.entry_bits));
    for (uint32_t model = 0; model < score_models; ++model) {
      const char *code =
          validation
              ? (model == 0 ? center_bank[ci].native.data()
                            : selected->p + 64 + ((model - 1) * NC + ci) * B)
              : candidates.p + 64 + (size_t(ci) * CAND + model) * B;
      codec.scan(code, query.lookup, unbits(state.header.center_bits),
                 scores.data());
      for (uint32_t lane = 0; lane < K; ++lane) {
        need(std::isfinite(scores[lane]), "candidate score finite");
        if (model == 0)
          need(bits(scores[lane]) == state.lanes[lane].native_bits,
               "candidate0 actual native LUT control");
        for (const Boundary *bound :
             {&state.lanes[lane].lower, &state.lanes[lane].upper}) {
          if (!bound->flags) continue;
          need((bucket.locate(scores[lane]) >= bound->at_bucket) ==
                   (scores[lane] >= unbits(bound->at_bits)),
               "candidate/right-inclusive FP32 transition equivalence");
        }
      }
      score_file.write(scores.data(), K * 4);
    }
  }
  score_file.close();
  std::ostringstream result;
  result << "{\n  \"passed\":true,\n  \"partition\":\"" << mode
         << "\",\n  \"state_count\":" << state_count
         << ",\n  \"active_query_count\":" << query_records.size()
         << ",\n  \"center_count\":" << NC
         << ",\n  \"models_scored_per_state\":" << score_models
         << ",\n  \"native_edges_bit_identical\":" << native_edges
         << ",\n  \"actual_base_rotations_bit_identical\":"
         << rotation_verified.size()
         << ",\n  \"finite_boundaries_nextafter_verified\":" << boundary_checks
         << ",\n  \"absent_finite_boundaries\":" << absent_boundaries
         << ",\n  \"native_codec_blocks_byte_identical\":20,\n  "
            "\"native_gate_completed_before_candidate_scoring\":true,"
         << "\n  \"validation_scores_read_in_training_mode\":false,\n  "
            "\"all_candidate_scores_in_validation\":false,"
         << "\n  \"actual_db_driver_preparation_used\":true,\n  "
            "\"native_trace_only_off_policy_states\":true,"
         << "\n  \"retrieval_or_training_performed\":false\n}\n";
  const std::string text = result.str();
  save(out + "/scorer_result.json", text.data(), text.size());
  return 0;
} catch (const std::exception &error) {
  std::cerr << "PHASE81 SCORE FAILED " << error.what() << '\n';
  return 1;
}
