// Mechanism A: fixed matched-state scoring, no model optimization.
#define main path80_accepted_scorer_main
#include "path80_score.cc"
#undef main
#include "coupled73_codec.h"

namespace {
struct Observation {
  uint32_t kind, owner, center_row, proxy_index, center, query_id, center_bits,
      reserved;
  double rotated_delta_norm2;
  std::array<float, 32> exact, native, learned;
};
static_assert(sizeof(Observation) == 424, "matched observation layout");
struct Training {
  double center, lambda, native_objective, learned_objective;
  double saved_native, saved_learned, max_scale, max_hamming;
};
static_assert(sizeof(Training) == 64, "training-control layout");
}  // namespace

int main(int argc, char **argv) try {
  need(argc == 4,
       "usage matched_score bench_root frozen_table_directory "
       "exclusive_score_directory");
  const std::string root = argv[1], table = argv[2], out = argv[3];
  Map states(table + "/states.u32"), centers(table + "/centers_proxies.u32");
  need(states.size % 20 == 0 && centers.size % 68 == 0, "frozen table shapes");
  const size_t state_count = states.size / 20, center_count = centers.size / 68;
  need(state_count == 3793 && center_count == 3719, "frozen mechanism A scale");
  Map graph(root + "/models64/full_graph.bin"),
      raw(root + "/models64/full_vectors.f32");
  Map rotated(root + "/models77/rotated.f32"),
      proxies(root + "/models77/proxies.u32");
  Map compact(root + "/models77/full/compact.u8"),
      trained(root + "/models77/full/audit.bin");
  Map events(root + "/models68/primary/events.bin");
  Map overlay(root + "/path80_native_v1/native_path_learned_scores.bin");
  const size_t n = 1000000;
  need(graph.size == 32 + n * 140 && raw.size == 16 + n * D * 4 &&
           rotated.size == n * D * 4 && proxies.size == n * 16 * 4 &&
           compact.size == n * 3072 && trained.size == n * 64 &&
           events.size == overlay.size,
       "frozen model/input sizes");
  independent78::Config config;
  config.model_path = root + "/models77/full/blocks.bin";
  config.raw_vectors_path = root + "/models64/full_vectors.f32";
  config.rotated_path = root + "/models77/rotated.f32";
  config.trusted_header_path = root + "/models77/header1000000.bin";
  config.model_sha256 =
      "2ac03543d4a09ae655d38130419df41668e4d62bdd3231b49135d2603c39f138";
  config.docs = n;
  independent78::Import model;
  model.open(config);
  Codec codec(D, true);
  ProductionQuery production;
  production.initialize();
  const auto path_queries = queries(
      "/home/xingrui/bench/dataset/cohere/1m/"
      "cohere_test_vector_1m.1000.new.txt");
  const auto raw_vector = [&](uint32_t id) {
    need(id < n, "raw ID range");
    return reinterpret_cast<const float *>(raw.p + 16) + size_t(id) * D;
  };
  const auto graph_record = [&](uint32_t id) {
    need(id < n, "graph ID range");
    return graph.p + 32 + size_t(id) * 140;
  };
  const auto key = [&](uint32_t id) {
    return take<uint64_t>(graph_record(id));
  };
  std::vector<uint8_t> verified(n, 0);
  std::vector<float> transformed;
  size_t verified_count = 0;
  const auto vector = [&](uint32_t id) {
    const float *v =
        reinterpret_cast<const float *>(rotated.p) + size_t(id) * D;
    if (!verified.at(id)) {
      codec.transform(raw_vector(id), transformed);
      model.verify_vectors(id, key(id), raw_vector(id), transformed.data(), D);
      need(std::memcmp(v, transformed.data(), D * 4) == 0,
           "actual base rotation bit identity");
      verified[id] = 1;
      ++verified_count;
    }
    return v;
  };
  std::vector<Block> blocks(center_count);
  std::vector<Training> training(center_count);
  std::vector<float> neighbors(K * D);
  std::vector<int8_t> learned_signs(K * D);
  std::vector<double> train_q(16 * D), error_native(D), error_learned(D);
  for (size_t ci = 0; ci < center_count; ++ci) {
    const uint32_t center_id = take<uint32_t>(centers.p + ci * 68);
    if (ci)
      need(center_id > take<uint32_t>(centers.p + (ci - 1) * 68),
           "ascending distinct centers");
    const char *g = graph_record(center_id);
    const float *c = vector(center_id);
    Block &b = blocks[ci];
    b.degree = take<uint32_t>(g + 8);
    need(b.degree == 32, "matched degree32");
    std::array<uint64_t, 32> neighbor_keys;
    for (size_t lane = 0; lane < K; ++lane) {
      b.ids[lane] = take<uint32_t>(g + 12 + lane * 4);
      neighbor_keys[lane] = key(b.ids[lane]);
      std::copy_n(vector(b.ids[lane]), D, neighbors.data() + lane * D);
      for (size_t k = 0; k < D; ++k) {
        const auto byte =
            uint8_t(compact.p[size_t(center_id) * 3072 + lane * 96 + k / 8]);
        learned_signs[lane * D + k] = (byte & (1u << (7 - k % 8))) ? 1 : -1;
      }
    }
    codec.encode(c, neighbors.data(), K, b.native.data());
    b.imported = b.native;
    b.learned =
        model.copy_full_block(center_id, key(center_id), neighbor_keys.data(),
                              K, b.imported.data(), B);
    need(b.learned, "matched full block import");
    std::array<char, B> independent_codes;
    coupled73::encode(c, neighbors.data(), learned_signs.data(), K, D,
                      independent_codes.data());
    need(independent_codes == b.imported,
         "compact signs reproduce accepted learned block bytes");
    Training tr{};
    tr.center = center_id;
    for (size_t j = 0; j < 16; ++j) {
      const uint32_t id =
          take<uint32_t>(proxies.p + (size_t(center_id) * 16 + j) * 4);
      const float *v = vector(id);
      for (size_t k = 0; k < D; ++k) {
        const double x = (double(v[k]) - c[k]) / 4;
        train_q[j * D + k] = x;
        tr.lambda += x * x;
      }
    }
    tr.lambda /= D;
    for (size_t lane = 0; lane < K; ++lane) {
      const float *v = neighbors.data() + lane * D;
      double rr = 0, l1 = 0, u = 0;
      size_t hamming = 0;
      for (size_t k = 0; k < D; ++k) {
        const double r = double(v[k]) - c[k];
        rr += r * r;
        l1 += std::abs(r);
        u += r * learned_signs[lane * D + k];
        hamming += learned_signs[lane * D + k] != (r > 0 ? 1 : -1);
      }
      need(rr == 0 || (l1 > 0 && u > 0), "radial denominators");
      const double alpha_native = rr == 0 ? 0 : rr / l1,
                   alpha_learned = rr == 0 ? 0 : rr / u;
      tr.max_scale = std::max(tr.max_scale, rr == 0 ? 1 : l1 / u);
      tr.max_hamming = std::max(tr.max_hamming, double(hamming));
      double en = 0, el = 0, pn = 0, pl = 0;
      for (size_t k = 0; k < D; ++k) {
        const double r = double(v[k]) - c[k];
        error_native[k] = alpha_native * (r > 0 ? 1 : -1) - r;
        error_learned[k] = alpha_learned * learned_signs[lane * D + k] - r;
        en += error_native[k] * error_native[k];
        el += error_learned[k] * error_learned[k];
      }
      for (size_t j = 0; j < 16; ++j) {
        double vn = 0, vl = 0;
        for (size_t k = 0; k < D; ++k) {
          vn += train_q[j * D + k] * error_native[k];
          vl += train_q[j * D + k] * error_learned[k];
        }
        pn += vn * vn;
        pl += vl * vl;
      }
      tr.native_objective += pn + tr.lambda * en;
      tr.learned_objective += pl + tr.lambda * el;
    }
    const char *a = trained.p + size_t(center_id) * 64;
    need(take<uint32_t>(a + 4) == 1 && take<double>(a + 56) == 32,
         "training audit degree/flag");
    tr.saved_native = take<double>(a + 16);
    tr.saved_learned = take<double>(a + 24);
    need(std::abs(tr.lambda - take<double>(a + 8)) <= 1e-13 &&
             std::abs(tr.native_objective - tr.saved_native) <=
                 1e-12 * std::max(1., std::abs(tr.saved_native)) &&
             std::abs(tr.learned_objective - tr.saved_learned) <=
                 1e-12 * std::max(1., std::abs(tr.saved_learned)),
         "recomputed full training objective vs accepted audit");
    need(tr.learned_objective <= tr.native_objective &&
             tr.max_scale <= 1.02 + 1e-12 && tr.max_hamming <= 8,
         "complete training objective nonincrease and budgets");
    training[ci] = tr;
    if ((ci + 1) % 512 == 0)
      std::cerr << "MATCHED80 prepared centers=" << ci + 1 << '/'
                << center_count << '\n';
  }
  std::vector<Observation> observations;
  observations.reserve(state_count + center_count * 16);
  size_t negative_lanes = 0, path_native_lanes = 0, path_learned_lanes = 0,
         positive_controls = 0;
  const auto score = [&](uint32_t kind, uint32_t owner, uint32_t ci,
                         uint32_t pj, uint32_t query_id, const float *input) {
    Observation record{};
    record.kind = kind;
    record.owner = owner;
    record.center_row = ci;
    record.proxy_index = pj;
    record.center = take<uint32_t>(centers.p + size_t(ci) * 68);
    record.query_id = query_id;
    std::string storage;
    const void *prepared = nullptr;
    core::IndexQueryMeta meta;
    production.prepare(input, storage, prepared, meta);
    Codec::Query q;
    codec.prepare_query(prepared, q);
    const auto *c = vector(record.center);
    for (size_t k = 0; k < D; ++k) {
      const double delta = double(q.rotated[k]) - c[k];
      record.rotated_delta_norm2 += delta * delta;
    }
    const Block &b = blocks.at(ci);
    const float exact_center =
        production.distance(raw_vector(record.center), prepared);
    record.center_bits = bits(exact_center);
    codec.scan(b.native.data(), q, exact_center, record.native.data());
    std::array<float, 32> repeated;
    codec.scan(b.native.data(), q, exact_center, repeated.data());
    need(std::memcmp(repeated.data(), record.native.data(), 32 * 4) == 0,
         "native/native negative control");
    negative_lanes += 32;
    codec.scan(b.imported.data(), q, exact_center, record.learned.data());
    for (size_t lane = 0; lane < K; ++lane) {
      record.exact[lane] =
          production.distance(raw_vector(b.ids[lane]), prepared);
      need(std::isfinite(record.native[lane]) &&
               std::isfinite(record.learned[lane]) &&
               std::isfinite(record.exact[lane]),
           "finite matched scores");
    }
    const auto chosen =
        std::min_element(record.exact.begin(), record.exact.end());
    need(double(*chosen) -
                 *std::min_element(record.exact.begin(), record.exact.end()) <=
             1e-7,
         "exact Top1 positive control");
    ++positive_controls;
    return record;
  };
  for (size_t si = 0; si < state_count; ++si) {
    const char *s = states.p + si * 20;
    const uint32_t qi = take<uint32_t>(s), step = take<uint32_t>(s + 4),
                   center = take<uint32_t>(s + 8);
    const uint32_t ci = take<uint32_t>(s + 12), ei = take<uint32_t>(s + 16);
    need(qi < 1000 && (step == 10 || step == 14 || step == 18 || step == 22) &&
             ci < center_count &&
             center == take<uint32_t>(centers.p + size_t(ci) * 68),
         "state identity");
    Observation record =
        score(0, si, ci, UINT32_MAX, qi, path_queries[qi].data());
    need(16 + (size_t(ei) + 33) * sizeof(Event) <= events.size,
         "state event bounds");
    const Event center_event = take<Event>(events.p + 16 + size_t(ei) * 32);
    need(center_event.query == qi && center_event.step == step &&
             center_event.kind == 1 && center_event.source == center &&
             bits(center_event.exact) == record.center_bits,
         "actual matched center vs trace");
    for (size_t lane = 0; lane < 32; ++lane) {
      const auto e = take<Event>(events.p + 16 + (size_t(ei) + 1 + lane) * 32);
      const auto l = take<Event>(overlay.p + 16 + (size_t(ei) + 1 + lane) * 32);
      need(e.kind == 2 && e.query == qi && e.step == step &&
               e.source == center && e.target == blocks[ci].ids[lane],
           "trace matched lane order");
      need(bits(e.exact) == bits(record.exact[lane]) &&
               bits(e.estimate) == bits(record.native[lane]),
           "path native/exact bit gate");
      need(bits(l.estimate) == bits(record.learned[lane]),
           "path learned vs accepted scorer bit gate");
      ++path_native_lanes;
      ++path_learned_lanes;
    }
    observations.push_back(record);
  }
  for (size_t ci = 0; ci < center_count; ++ci) {
    const uint32_t center = take<uint32_t>(centers.p + ci * 68);
    for (uint32_t pj = 0; pj < 16; ++pj) {
      const uint32_t id = take<uint32_t>(centers.p + ci * 68 + 4 + pj * 4);
      need(id < n && id != center &&
               std::find(blocks[ci].ids.begin(), blocks[ci].ids.end(), id) ==
                   blocks[ci].ids.end(),
           "heldout exclusion");
      for (size_t j = 0; j < 16; ++j)
        need(id != take<uint32_t>(proxies.p + (size_t(center) * 16 + j) * 4),
             "heldout excludes training");
      observations.push_back(score(1, ci, ci, pj, id, raw_vector(id)));
    }
  }
  const uint32_t header[8] = {0x51474180,
                              1,
                              sizeof(Observation),
                              768,
                              uint32_t(state_count),
                              uint32_t(center_count),
                              32,
                              16};
  std::vector<char> data(32 + observations.size() * sizeof(Observation));
  std::memcpy(data.data(), header, 32);
  std::memcpy(data.data() + 32, observations.data(),
              observations.size() * sizeof(Observation));
  save(out + "/observations.bin", data.data(), data.size());
  save(out + "/training_controls.f64", training.data(),
       training.size() * sizeof(Training));
  std::ostringstream result;
  result << "{\"passed\":true,\"path_observations\":" << state_count
         << ",\"unique_centers\":" << center_count
         << ",\"proxy_observations\":" << center_count * 16
         << ",\"observations\":" << observations.size()
         << ",\"native_native_bit_identical_lanes\":" << negative_lanes
         << ",\"path_native_exact_trace_bit_identical_lanes\":"
         << path_native_lanes
         << ",\"path_learned_accepted_overlay_bit_identical_lanes\":"
         << path_learned_lanes
         << ",\"exact_top1_positive_control_observations\":"
         << positive_controls
         << ",\"full_training_objective_nonincreasing_centers\":"
         << training.size()
         << ",\"actual_base_rotations_verified\":" << verified_count
         << ",\"proxy_query_preparation\":\"actual HNSW prepare and FP32 Turbo "
            "callback on frozen normalized database vector\","
         << "\"no_training_or_parameter_fit\":true,\"off_policy_native_"
            "states\":true,\"retrieval_or_qps_run\":false}\n";
  const auto text = result.str();
  save(out + "/score_controls.json", text.data(), text.size());
  std::cout << text;
  return 0;
} catch (const std::exception &e) {
  std::cerr << "MATCHED80 FAILED " << e.what() << '\n';
  return 1;
}
