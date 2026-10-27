// Mechanics only: native query-dot partial scan, full center/code constants.
#define main phase80_reference_program_main
#include "path80_score.cc"
#undef main
#include <immintrin.h>
#include <cfenv>
#include <memory>

namespace partial82 {
constexpr uint32_t NC = 20, QN = 3072, NS = 6625, MODES = 2, BUDGETS = 5;
constexpr std::array<uint32_t, BUDGETS> dims{{128, 256, 384, 512, 768}};
constexpr uint32_t GROUPS = D / 4;
float fp_add(float a, float b) {
  volatile float value = a + b;
  return value;
}
float fp_mul(float a, float b) {
  volatile float value = a * b;
  return value;
}
float unbits(uint32_t value) {
  float result;
  std::memcpy(&result, &value, 4);
  return result;
}
// Match the two actual FMA instructions in the frozen native binary.
// V1's separately rounded mul/add is retained in its failed source snapshot.
float native_epilogue(uint32_t sum, float delta, float vl, float k1, float add,
                      float scale, float center) {
  float value = std::fma(delta, float(sum), vl);
  value = fp_add(value, k1);
  const float base = fp_add(add, center);
  return std::fma(scale, value, base);
}
float signed_query_partial(uint32_t sum, uint32_t m, float delta, float full_vl,
                           float prefix_k1, float full_k1, float add,
                           float scale, float center) {
  if (m == D)
    return native_epilogue(sum, delta, full_vl, full_k1, add, scale, center);
  const float rho = float(D) / float(m);
  const float scaled_delta = fp_mul(rho, delta);
  float value = std::fma(scaled_delta, float(sum), full_vl);
  value = fp_add(value, fp_mul(rho, prefix_k1));
  return std::fma(scale, value, fp_add(add, center));
}
float uncentered_ht(uint32_t sum, uint32_t m, float delta, float full_vl,
                    float full_k1, float add, float scale, float center) {
  if (m == D)
    return native_epilogue(sum, delta, full_vl, full_k1, add, scale, center);
  const float rho = float(D) / float(m);
  const float expanded = fp_mul(rho, float(sum));
  float value = std::fma(delta, expanded, full_vl);
  value = fp_add(value, full_k1);
  return std::fma(scale, value, fp_add(add, center));
}
// Independently spelled scalar oracle; explicit fma models the frozen binary,
// while volatile additions/multiplications enforce their separate FP32 rounds.
float scalar_reference(uint32_t sum, uint32_t m, bool centered, float delta,
                       float vl, float selected_k1, float full_k1, float a,
                       float b, float g) {
  volatile float rho = float(D) / float(m);
  volatile float shifted;
  volatile float correction;
  if (m == D) {
    shifted = std::fma(delta, float(sum), vl);
    correction = full_k1;
  } else if (centered) {
    volatile float product = rho * delta;
    shifted = std::fma(product, float(sum), vl);
    correction = rho * selected_k1;
  } else {
    volatile float expanded = rho * float(sum);
    shifted = std::fma(delta, expanded, vl);
    correction = full_k1;
  }
  volatile float dot = shifted + correction;
  volatile float base = a + g;
  volatile float result = std::fma(b, dot, base);
  return result;
}
using Accumulator = std::conditional_t<
    std::is_invocable_v<decltype(&rabitqlib::fastscan::accumulate),
                        const uint8_t *, const uint8_t *, int32_t *, size_t>,
    int32_t, uint16_t>;
std::array<uint32_t, K> accumulate(const uint8_t *codes, const uint8_t *lut,
                                   uint32_t dimension) {
  need(dimension > 0 && dimension <= D && dimension % 16 == 0,
       "valid partial kernel dimensions");
  std::array<Accumulator, K> small{};
  rabitqlib::fastscan::accumulate(codes, lut, small.data(), dimension);
  std::array<uint32_t, K> result{};
  for (size_t lane = 0; lane < K; ++lane) result[lane] = uint32_t(small[lane]);
  return result;
}
uint8_t packed_nibble(const uint8_t *codes, uint32_t cb, uint32_t lane) {
  const uint32_t pos = 2 * (lane & 7U) + ((lane >> 3) & 1U);
  return uint8_t((codes[cb * 16 + pos] >> (lane >= 16 ? 4 : 0)) & 15);
}
uint8_t compact_nibble(const uint8_t *compact, uint32_t cb, uint32_t lane) {
  return uint8_t((compact[lane * (D / 8) + cb / 2] >> (cb % 2 == 0 ? 4 : 0)) &
                 15);
}
struct Query {
  zvec::turbo::SymQuery full;
  std::array<std::vector<uint8_t>, BUDGETS> gathered_lut;
  float selected_k1[MODES][BUDGETS]{};
  uint32_t tail_low[MODES][BUDGETS]{}, tail_high[MODES][BUDGETS]{};
  uint32_t serial = 0;
};
struct Sample {
  uint32_t mode, m, lane, prefix_sum, full_sum;
  uint32_t signed_query_bits, uncentered_ht_bits, lower_bits, upper_bits,
      native_bits;
};
static_assert(sizeof(Sample) == 40, "diagnostic sample layout");
}  // namespace partial82

int main(int argc, char **argv) try {
  using namespace partial82;
  need(argc == 7,
       "usage: partial82_mechanics training_states training_queries "
       "training_scores candidate_dir subset_indices.u32 exclusive_output_dir");
  need(std::numeric_limits<float>::is_iec559 &&
           std::fegetround() == FE_TONEAREST,
       "IEEE round-to-nearest FP32 required");
  Map states(argv[1]), queries(argv[2]), saved_scores(argv[3]),
      subsets(argv[5]);
  const std::string bank = argv[4], out = argv[6];
  Map blocks(bank + "/native_blocks.bin"),
      compact(bank + "/candidate_compact.u8");
  const auto check_header = [](const Map &file, uint32_t magic, uint32_t record,
                               uint32_t rows, uint32_t models) {
    const uint32_t header[16] = {magic, 1, 64, record, D, K, rows, models,
                                 0,     0, 0,  0,      0, 0, 0,    0};
    need(file.size >= 64 && std::memcmp(file.p, header, 64) == 0,
         "frozen TRAINING export header");
  };
  check_header(states, 0x51474181, 3136, NS, 137);
  check_header(queries, 0x51475181, 6164, QN, 0);
  check_header(saved_scores, 0x51474581, 128, NS, 137);
  need(states.size == 64 + NS * 3136ULL && queries.size == 64 + QN * 6164ULL &&
           saved_scores.size == 64 + NS * 137ULL * K * 4,
       "training sizes");
  need(blocks.size == NC * B && compact.size == NC * 137ULL * K * (D / 8),
       "native code bank shape");
  need(subsets.size == MODES * BUDGETS * GROUPS * 4, "subset shape");
  uint32_t index[MODES][BUDGETS][GROUPS];
  std::memcpy(index, subsets.p, sizeof(index));
  for (uint32_t mode = 0; mode < MODES; ++mode)
    for (uint32_t mi = 0; mi < BUDGETS; ++mi) {
      std::array<bool, GROUPS> seen{};
      for (uint32_t pos = 0; pos < GROUPS; ++pos) {
        const uint32_t cb = index[mode][mi][pos];
        if (pos >= dims[mi] / 4) {
          need(cb == UINT32_MAX, "unused subset slots");
          continue;
        }
        need(cb < GROUPS && !seen[cb],
             "uniform subset contains distinct codebooks");
        seen[cb] = true;
        if (mode == 0)
          need(cb == pos, "fixed prefix order");
        else
          need(cb == index[mode][BUDGETS - 1][pos],
               "one consistent global random permutation");
      }
    }
  std::array<std::array<std::vector<uint8_t>, BUDGETS>, NC> gathered_codes;
  for (uint32_t ci = 0; ci < NC; ++ci) {
    rabitqlib::ConstQGBatchDataMap<float> batch(blocks.p + ci * B, D);
    for (uint32_t lane = 0; lane < K; ++lane)
      need(std::isfinite(batch.f_add()[lane]) &&
               std::isfinite(batch.f_rescale()[lane]),
           "finite original full-dimensional factors");
    for (uint32_t mi = 0; mi < BUDGETS; ++mi) {
      auto &codes = gathered_codes[ci][mi];
      codes.resize(dims[mi] * 4);
      for (uint32_t p = 0; p < dims[mi] / 4; ++p)
        std::copy_n(batch.bin_code() + index[1][mi][p] * 16, 16,
                    codes.data() + p * 16);
    }
  }
  std::vector<std::unique_ptr<Query>> query_bank;
  query_bank.reserve(QN);
  for (uint32_t qi = 0; qi < QN; ++qi) {
    const char *record = queries.p + 64 + qi * 6164ULL;
    auto query = std::make_unique<Query>();
    query->serial = take<uint32_t>(record);
    const auto *q = reinterpret_cast<const float *>(record + 16 + (D + 1) * 4);
    query->full.reset(q, D);
    need(std::isfinite(query->full.delta()) && query->full.delta() >= 0 &&
             std::isfinite(query->full.sum_vl_lut()) &&
             std::isfinite(query->full.k1xsumq()),
         "finite full LUT constants");
    for (uint32_t mode = 0; mode < MODES; ++mode)
      for (uint32_t mi = 0; mi < BUDGETS; ++mi) {
        const uint32_t m = dims[mi], h = m / 4;
        std::array<bool, GROUPS> selected{};
        std::vector<float> selected_query(m);
        if (mode == 1) query->gathered_lut[mi].resize(m * 4);
        for (uint32_t p = 0; p < h; ++p) {
          const uint32_t cb = index[mode][mi][p];
          selected[cb] = true;
          std::copy_n(q + cb * 4, 4, selected_query.data() + p * 4);
          if (mode == 1)
            std::copy_n(query->full.lut() + cb * 16, 16,
                        query->gathered_lut[mi].data() + p * 16);
        }
        if (m == D)
          query->selected_k1[mode][mi] = query->full.k1xsumq();
        else {
          zvec::turbo::SymQuery prefix_correction;
          prefix_correction.reset(selected_query.data(), m);
          // Only its query-sum correction is used. NEVER substitute its
          // LUT/delta/vl.
          query->selected_k1[mode][mi] = prefix_correction.k1xsumq();
        }
        for (uint32_t cb = 0; cb < GROUPS; ++cb)
          if (!selected[cb]) {
            const uint8_t *table = query->full.lut() + cb * 16;
            query->tail_low[mode][mi] += *std::min_element(table, table + 16);
            query->tail_high[mode][mi] += *std::max_element(table, table + 16);
          }
      }
    query_bank.push_back(std::move(query));
  }
  uint64_t native_bits = 0, integer_checks = 0, nibble_checks = 0,
           score_checks = 0, bound_checks = 0, full_bound_collapses = 0,
           actual_partial_scan_checks = 0;
  std::vector<Sample> samples;
  for (uint32_t si = 0; si < NS; ++si) {
    const char *state = states.p + 64 + si * 3136ULL;
    const uint32_t ci = take<uint32_t>(state + 40),
                   qi = take<uint32_t>(state + 56);
    need(ci < NC && qi < QN && query_bank[qi]->serial == take<uint32_t>(state),
         "state/query/center mapping");
    Query &query = *query_bank[qi];
    const float center = unbits(take<uint32_t>(state + 28));
    query.full.set_g_add(center);
    const char *code = blocks.p + ci * B;
    const auto *native_compact = reinterpret_cast<const uint8_t *>(
        compact.p + size_t(ci) * 137 * K * (D / 8));
    rabitqlib::ConstQGBatchDataMap<float> batch(
        code, D);  // Full factor offsets are ALWAYS D=768.
    const auto full_sum = accumulate(batch.bin_code(), query.full.lut(), D);
    std::array<float, K> native{};
    zvec::turbo::ScanSymphonyQGBatch(code, query.full, D, native.data());
    for (uint32_t lane = 0; lane < K; ++lane) {
      const uint32_t expected = take<uint32_t>(state + 64 + lane * 96 + 4);
      need(bits(native[lane]) == expected &&
               take<uint32_t>(saved_scores.p + 64 +
                              (size_t(si) * 137 * K + lane) * 4) == expected,
           "actual native saved state/score bits");
      need(bits(native_epilogue(full_sum[lane], query.full.delta(),
                                query.full.sum_vl_lut(), query.full.k1xsumq(),
                                batch.f_add()[lane], batch.f_rescale()[lane],
                                center)) == expected,
           "explicit two-FMA native epilogue bit identity");
      ++native_bits;
    }
    for (uint32_t mode = 0; mode < MODES; ++mode)
      for (uint32_t mi = 0; mi < BUDGETS; ++mi) {
        const uint32_t m = dims[mi], h = m / 4;
        const uint8_t *codes =
            mode == 0 ? batch.bin_code() : gathered_codes[ci][mi].data();
        const uint8_t *lut =
            mode == 0 ? query.full.lut() : query.gathered_lut[mi].data();
        const auto partial = accumulate(codes, lut, m);
        std::array<float, K> actual_shortened_scan{};
        const float rho = float(D) / float(m);
        const float scaled_delta =
            m == D ? query.full.delta() : fp_mul(rho, query.full.delta());
        const float scaled_k1 = m == D
                                    ? query.full.k1xsumq()
                                    : fp_mul(rho, query.selected_k1[mode][mi]);
        need(zvec::turbo::symqg_scan(codes, lut, m, batch.f_add(),
                                     batch.f_rescale(), center, scaled_delta,
                                     query.full.sum_vl_lut(), scaled_k1,
                                     actual_shortened_scan.data()),
             "actual shortened production AVX512 scan must execute");
        for (uint32_t lane = 0; lane < K; ++lane) {
          uint32_t scalar = 0;
          for (uint32_t p = 0; p < h; ++p) {
            const uint32_t cb = index[mode][mi][p];
            const uint8_t a = packed_nibble(batch.bin_code(), cb, lane);
            const uint8_t b = compact_nibble(native_compact, cb, lane);
            need(a == b, "independent packed/compact lane nibble");
            ++nibble_checks;
            scalar += query.full.lut()[cb * 16 + b];
          }
          need(partial[lane] == scalar && scalar <= 48960,
               "partial kernel vs independent uint32 scalar LUT sum");
          ++integer_checks;
          const float a = batch.f_add()[lane], b = batch.f_rescale()[lane];
          const float signed_score = signed_query_partial(
              partial[lane], m, query.full.delta(), query.full.sum_vl_lut(),
              query.selected_k1[mode][mi], query.full.k1xsumq(), a, b, center);
          const float ht = uncentered_ht(partial[lane], m, query.full.delta(),
                                         query.full.sum_vl_lut(),
                                         query.full.k1xsumq(), a, b, center);
          need(std::isfinite(signed_score) && std::isfinite(ht),
               "finite partial formulas");
          need(bits(actual_shortened_scan[lane]) == bits(signed_score),
               "actual shortened production vector scan vs scalar signed-query "
               "FP32");
          ++actual_partial_scan_checks;
          need(bits(signed_score) ==
                   bits(scalar_reference(scalar, m, true, query.full.delta(),
                                         query.full.sum_vl_lut(),
                                         query.selected_k1[mode][mi],
                                         query.full.k1xsumq(), a, b, center)),
               "signed-query partial scalar FP32 reference");
          need(bits(ht) ==
                   bits(scalar_reference(scalar, m, false, query.full.delta(),
                                         query.full.sum_vl_lut(),
                                         query.selected_k1[mode][mi],
                                         query.full.k1xsumq(), a, b, center)),
               "uncentered HT scalar FP32 reference");
          score_checks += 2;
          const uint32_t lo_sum = scalar + query.tail_low[mode][mi],
                         hi_sum = scalar + query.tail_high[mode][mi];
          need(lo_sum <= full_sum[lane] && full_sum[lane] <= hi_sum,
               "actual integer LUT-tail containment");
          const float endpoint0 = native_epilogue(
              lo_sum, query.full.delta(), query.full.sum_vl_lut(),
              query.full.k1xsumq(), a, b, center);
          const float endpoint1 = native_epilogue(
              hi_sum, query.full.delta(), query.full.sum_vl_lut(),
              query.full.k1xsumq(), a, b, center);
          need(std::isfinite(endpoint0) && std::isfinite(endpoint1),
               "finite native tail-bound endpoints");
          float low = std::min(endpoint0, endpoint1),
                high = std::max(endpoint0, endpoint1);
          need(low <= native[lane] && native[lane] <= high,
               "monotone actual FP32 native-score closed bound");
          if (m == D) {
            need(scalar == full_sum[lane] &&
                     bits(signed_score) == bits(native[lane]) &&
                     bits(ht) == bits(native[lane]) &&
                     bits(endpoint0) == bits(native[lane]) &&
                     bits(endpoint1) == bits(native[lane]),
                 "m=D exact score and tail-bound collapse");
            ++full_bound_collapses;
          } else {
            low = std::nextafter(low, -std::numeric_limits<float>::infinity());
            high = std::nextafter(high, std::numeric_limits<float>::infinity());
            need(low <= native[lane] && native[lane] <= high,
                 "outward FP32 interval containment");
          }
          ++bound_checks;
          if (si == 0)
            samples.push_back({mode, m, lane, scalar, full_sum[lane],
                               bits(signed_score), bits(ht), bits(low),
                               bits(high), bits(native[lane])});
        }
      }
  }
  // Degenerate scale/delta algebra controls; no new data/query sampling.
  need(bits(native_epilogue(0, 0, 0, 0, 1, -2, .5f)) ==
           bits(native_epilogue(48960, 0, 0, 0, 1, -2, .5f)),
       "zero-delta tail degeneracy");
  need(bits(native_epilogue(0, .1f, -1, -.5f, 1, 0, .5f)) ==
           bits(native_epilogue(48960, .1f, -1, -.5f, 1, 0, .5f)),
       "zero-scale tail degeneracy");
  const uint32_t sample_header[8] = {
      0x51475082, 1, 40, uint32_t(samples.size()), MODES, BUDGETS, K, 0};
  std::vector<char> sample_bytes(sizeof(sample_header) +
                                 samples.size() * sizeof(Sample));
  std::memcpy(sample_bytes.data(), sample_header, sizeof(sample_header));
  std::memcpy(sample_bytes.data() + sizeof(sample_header), samples.data(),
              samples.size() * sizeof(Sample));
  save(out + "/first_state_diagnostic_samples.bin", sample_bytes.data(),
       sample_bytes.size());
  std::ostringstream report;
  report
      << "{\n\"passed\":true,\"states\":" << NS
      << ",\"training_queries\":" << QN << ",\"centers\":" << NC
      << ",\"dimensions\":[128,256,384,512,768],\"subset_modes\":[\"prefix\","
         "\"frozen_global_random_codebook_subset\"],"
      << "\"native_score_bits_verified\":" << native_bits
      << ",\"partial_integer_checks\":" << integer_checks
      << ",\"actual_partial_vector_scan_checks\":" << actual_partial_scan_checks
      << ",\"packed_compact_nibble_checks\":" << nibble_checks
      << ",\"partial_fp32_formula_checks\":" << score_checks
      << ",\"native_tail_bound_containment_checks\":" << bound_checks
      << ",\"mD_bitexact_and_bound_collapses\":" << full_bound_collapses
      << ",\"actual_native_epilogue\":\"fma(delta,N,vl);add(k1);add(A,g);fma(B,"
         "dot,base)\""
      << ",\"mxcsr\":" << _mm_getcsr()
      << ",\"degenerate_delta_and_scale_controls\":2,"
      << "\"full_original_factors_and_query_lut_used\":true,\"prefix_reset_"
         "used_only_for_query_sum_correction\":true,"
      << "\"centered_formula_is_partial_q_dot_with_A_full\":true,\"is_split_"
         "residual_estimator\":false,"
      << "\"tail_bound_target\":\"actual native codec score, not exact "
         "distance\","
      << "\"bound_arithmetic\":\"integer remaining LUT min/max then explicit "
         "two-FMA native epilogue;partial bounds additionally outward "
         "nextafter\","
      << "\"validation_artifacts_or_truth_labels_used\":false,\"new_retrieval_"
         "or_qps\":false,\"rabitq_error_bound_claim\":false}\n";
  const std::string result = report.str();
  save(out + "/mechanics_result.json", result.data(), result.size());
  return 0;
} catch (const std::exception &error) {
  std::cerr << "PARTIAL82 FAILED " << error.what() << '\n';
  return 1;
}
