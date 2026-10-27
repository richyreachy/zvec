// Geometry-only phase81 candidate encoding. No query or event file is opened.
#define main path80_unused_main
#include "path80_score.cc"
#undef main
#include <numeric>
#include "candidate81_codec.h"
#include "coupled73_codec.h"

int main(int argc, char **argv) try {
  if (argc == 5 && std::string(argv[1]) == "--assemble") {
    const std::string input = argv[2], index_path = argv[3], out = argv[4];
    Map bank(input + "/candidates.bin"),
        compact(input + "/candidate_compact.u8");
    Map valid(input + "/valid.u8"), indices(index_path);
    constexpr size_t NC = 20, NQ = candidate81::kCandidates, NM = 2;
    const uint32_t expected[16] = {0x51474381, 1, 64, D, NC, NQ, K, B,
                                   0,          0, 0,  0, 0,  0,  0, 0};
    need(
        bank.size == 64 + NC * NQ * B && std::memcmp(bank.p, expected, 64) == 0,
        "candidate bank header for assembly");
    need(compact.size == NC * NQ * K * (D / 8) && valid.size == NC * NQ * K &&
             indices.size == NM * NC * K * 4,
         "assembly input shapes");
    std::vector<char> output(64 + NM * NC * B, 0);
    const uint32_t header[16] = {0x51474d81, 1, 64, D, NC, NM, K, B,
                                 0,          0, 0,  0, 0,  0,  0, 0};
    std::memcpy(output.data(), header, 64);
    for (size_t model = 0; model < NM; ++model) {
      for (size_t ci = 0; ci < NC; ++ci) {
        const char *blocks = bank.p + 64 + ci * NQ * B;
        const auto *codes = reinterpret_cast<const uint8_t *>(compact.p) +
                            ci * NQ * K * (D / 8);
        const auto *mask =
            reinterpret_cast<const uint8_t *>(valid.p) + ci * NQ * K;
        const auto *selected = reinterpret_cast<const uint32_t *>(indices.p) +
                               (model * NC + ci) * K;
        char *dst = output.data() + 64 + (model * NC + ci) * B;
        candidate81::assemble_selected_block(blocks, codes, mask, selected,
                                             dst);
        need(candidate81::verify_selected_block(blocks, codes, mask, selected,
                                                dst),
             "assembled model byte verification");
      }
    }
    save(out + "/selected_models.bin", output.data(), output.size());
    const std::string proof =
        "{\"passed\":true,\"models\":2,\"centers\":20,\"lanes\":32,\"assembled_"
        "blocks_verified\":40,\"factor_recomputed\":false,\"queries_or_labels_"
        "read\":false}\n";
    save(out + "/assembly_result.json", proof.data(), proof.size());
    std::cout << proof;
    return 0;
  }
  need(argc == 3,
       "usage: prepare81_candidate_codes bench_root exclusive_output");
  const std::string root = argv[1], out = argv[2];
  Map graph(root + "/models64/full_graph.bin");
  Map raw(root + "/models64/full_vectors.f32");
  Map rotated(root + "/models77/rotated.f32");
  Map center_ids(out + "/centers.u32"), neighbors_ids(out + "/neighbors.u32");
  Map degrees(out + "/degrees.u32"), coordinates(out + "/coords.u16");
  Map flips(out + "/flip_local.u16"), valid(out + "/valid.u8");
  Map signs_source(out + "/native_signs.i8");
  constexpr size_t NC = 20, NQ = candidate81::kCandidates;
  constexpr size_t N = 1000000;
  need(graph.size == 32 + N * 140 && raw.size == 16 + N * D * 4 &&
           rotated.size == N * D * 4,
       "frozen geometry sizes");
  need(center_ids.size == NC * 4 && neighbors_ids.size == NC * K * 4 &&
           degrees.size == NC * 4 && coordinates.size == NC * K * 16 * 2 &&
           flips.size == NQ * 2 * 2 && valid.size == NC * NQ * K &&
           signs_source.size == NC * K * D,
       "candidate metadata sizes");
  Codec codec(D, true);
  need(codec.encoded_dim() == D && codec.quantized_dim() == D &&
           codec.batch_bytes() == B,
       "frozen actual codec layout");
  const uint32_t endian = 1;
  need(*reinterpret_cast<const char *>(&endian) == 1 &&
           std::numeric_limits<float>::is_iec559,
       "little-endian IEEE FP32 required");
  std::vector<uint8_t> verified(N, 0);
  size_t rotations_checked = 0, invalid_candidates = 0, short_blocks = 0;
  size_t zero_residual_edges = 0, assemblies_checked = 0;
  std::vector<float> transformed;
  const auto vec = [&](uint32_t id) {
    need(id < N, "vector index range");
    const float *value =
        reinterpret_cast<const float *>(rotated.p) + size_t(id) * D;
    if (!verified[id]) {
      const float *source =
          reinterpret_cast<const float *>(raw.p + 16) + size_t(id) * D;
      codec.transform(source, transformed);
      need(transformed.size() == D &&
               std::memcmp(value, transformed.data(), D * 4) == 0,
           "actual production rotation differs from frozen bank");
      verified[id] = 1;
      ++rotations_checked;
    }
    return value;
  };
  std::vector<char> native_bank(NC * B, 0), candidates(NC * NQ * B, 0);
  std::vector<uint8_t> compact_bank(NC * NQ * K * (D / 8), 0);
  std::vector<double> geometry(NC * K * 8, 0), residuals(NC * K * D, 0);
  std::vector<float> native_norms(NC * K * 2, 0);
  std::vector<float> neighbors(K * D, 0.f);
  std::vector<int8_t> signs(K * D, -1);
  for (size_t ci = 0; ci < NC; ++ci) {
    const uint32_t center = take<uint32_t>(center_ids.p + ci * 4);
    need(center < N, "center ID range");
    const char *g = graph.p + 32 + size_t(center) * 140;
    const size_t degree = take<uint32_t>(g + 8);
    need(degree <= K && degree == take<uint32_t>(degrees.p + ci * 4),
         "degree identity");
    short_blocks += degree != K;
    const float *c = vec(center);
    std::fill(neighbors.begin(), neighbors.end(), 0.f);
    for (size_t lane = 0; lane < K; ++lane) {
      const uint32_t id = take<uint32_t>(neighbors_ids.p + (ci * K + lane) * 4);
      if (lane < degree) {
        need(id == take<uint32_t>(g + 12 + lane * 4) && id != center,
             "final neighbor order");
        std::copy_n(vec(id), D, neighbors.data() + lane * D);
      } else {
        need(id == UINT32_MAX, "short-block neighbor padding");
      }
      const uint16_t *coord =
          reinterpret_cast<const uint16_t *>(coordinates.p) +
          (ci * K + lane) * 16;
      for (size_t i = 0; i < 16; ++i) {
        need(coord[i] < D && (!i || coord[i - 1] < coord[i]),
             "lex physical coordinates");
      }
    }
    char *native = native_bank.data() + ci * B;
    if (degree) codec.encode(c, neighbors.data(), degree, native);
    rabitqlib::ConstQGBatchDataMap<float> baseline(native, D);
    for (size_t lane = 0; lane < degree; ++lane) {
      const float *v = neighbors.data() + lane * D;
      double rr = 0, l1 = 0, cn = 0, vn = 0;
      for (size_t k = 0; k < D; ++k) {
        const double r = double(v[k]) - c[k];
        need(std::isfinite(r), "finite residual");
        residuals[(ci * K + lane) * D + k] = r;
        rr += r * r;
        l1 += std::abs(r);
        cn += double(c[k]) * c[k];
        vn += double(v[k]) * v[k];
        need(
            int8_t(signs_source.p[(ci * K + lane) * D + k]) == (r > 0 ? 1 : -1),
            "native sign with zero=-1");
      }
      zero_residual_edges += rr == 0;
      double *dst = geometry.data() + (ci * K + lane) * 8;
      dst[0] = rr;
      dst[1] = l1;
      dst[2] = cn;
      dst[3] = vn;
      dst[4] = rr ? rr / l1 : 0;
      dst[5] = baseline.f_add()[lane];
      dst[6] = baseline.f_rescale()[lane];
      dst[7] = l1 / 102.;
      native_norms[(ci * K + lane) * 2] = std::inner_product(c, c + D, c, 0.f);
      native_norms[(ci * K + lane) * 2 + 1] =
          std::inner_product(v, v + D, v, 0.f);
    }
    for (size_t choice = 0; choice < NQ; ++choice) {
      char *code = candidates.data() + (ci * NQ + choice) * B;
      const uint8_t *mask =
          reinterpret_cast<const uint8_t *>(valid.p) + (ci * NQ + choice) * K;
      std::copy_n(reinterpret_cast<const int8_t *>(signs_source.p) + ci * K * D,
                  K * D, signs.data());
      for (size_t lane = 0; lane < K; ++lane) {
        need(mask[lane] <= 1, "binary validity mask");
        need(choice || mask[lane] == 1, "native candidate always available");
        if (choice && !mask[lane]) ++invalid_candidates;
        if (!choice || !mask[lane]) continue;
        need(degree == K && geometry[(ci * K + lane) * 8] > 0,
             "non-native requires full block and nonzero residual");
        for (size_t j = 0; j < 2; ++j) {
          const uint16_t local = take<uint16_t>(flips.p + (choice * 2 + j) * 2);
          if (local == UINT16_MAX) continue;
          need(local < 16, "local flip coordinate range");
          const uint16_t k = take<uint16_t>(coordinates.p +
                                            ((ci * K + lane) * 16 + local) * 2);
          signs[lane * D + k] *= -1;
        }
      }
      uint8_t *compact = compact_bank.data() + (ci * NQ + choice) * K * (D / 8);
      for (size_t lane = 0; lane < K; ++lane)
        for (size_t k = 0; k < D; ++k)
          if (signs[lane * D + k] > 0)
            compact[lane * (D / 8) + k / 8] |= uint8_t(1u << (7 - k % 8));
      if (choice == 0 || degree != K) {
        std::memcpy(code, native, B);
      } else {
        coupled73::encode(c, neighbors.data(), signs.data(), K, D, code);
        rabitqlib::QGBatchDataMap<float> candidate(code, D);
        for (size_t lane = 0; lane < K; ++lane) {
          if (!mask[lane]) {
            candidate.f_add()[lane] = baseline.f_add()[lane];
            candidate.f_rescale()[lane] = baseline.f_rescale()[lane];
          }
        }
      }
      // Verify actual packed sign bytes independently from the factor recipe.
      std::array<char, B> sign_check{};
      rabitqlib::QGBatchDataMap<float> packed(sign_check.data(), D);
      rabitqlib::fastscan::pack_codes(D, compact, K, packed.bin_code());
      rabitqlib::ConstQGBatchDataMap<float> candidate(code, D);
      need(std::memcmp(packed.bin_code(), candidate.bin_code(), K * D / 8) == 0,
           "compact signs vs actual FastScan packing");
    }
    const char *bank = candidates.data() + ci * NQ * B;
    const auto *compact = compact_bank.data() + ci * NQ * K * (D / 8);
    const auto *mask = reinterpret_cast<const uint8_t *>(valid.p) + ci * NQ * K;
    std::array<uint32_t, K> indices{};
    need(candidate81::verify_selected_block(bank, compact, mask, indices.data(),
                                            native),
         "selected native assembly bit identity");
    ++assemblies_checked;
    for (size_t choice = 1; choice < NQ; ++choice) {
      for (size_t lane = 0; lane < K; ++lane)
        indices[lane] = mask[choice * K + lane] ? choice : 0;
      need(candidate81::verify_selected_block(
               bank, compact, mask, indices.data(), bank + choice * B),
           "mixed valid/native candidate assembly bit identity");
      ++assemblies_checked;
    }
  }
  const uint32_t header[16] = {0x51474381, 1, 64, D, NC, NQ, K, B,
                               0,          0, 0,  0, 0,  0,  0, 0};
  std::vector<char> all(sizeof(header) + candidates.size());
  std::memcpy(all.data(), header, sizeof(header));
  std::memcpy(all.data() + sizeof(header), candidates.data(),
              candidates.size());
  save(out + "/candidates.bin", all.data(), all.size());
  save(out + "/native_blocks.bin", native_bank.data(), native_bank.size());
  save(out + "/candidate_compact.u8", compact_bank.data(), compact_bank.size());
  save(out + "/geometry.f64", geometry.data(),
       geometry.size() * sizeof(double));
  save(out + "/residuals.f64", residuals.data(),
       residuals.size() * sizeof(double));
  save(out + "/native_norms.f32", native_norms.data(),
       native_norms.size() * sizeof(float));
  std::ostringstream proof;
  proof << "{\"passed\":true,\"geometry_only\":true,\"candidate_count\":137,"
           "\"center_count\":20,"
        << "\"native_blocks_bit_identical\":true,\"actual_rotations_verified\":"
        << rotations_checked << ",\"native_and_candidate_assemblies_verified\":"
        << assemblies_checked
        << ",\"short_blocks_native_only\":" << short_blocks
        << ",\"zero_residual_edges_native_only\":" << zero_residual_edges
        << ",\"invalid_non_native_lane_candidates\":" << invalid_candidates
        << ",\"actual_codec_block_bytes\":3328,\"queries_or_labels_read\":"
           "false}\n";
  const auto text = proof.str();
  save(out + "/encoding_proof.json", text.data(), text.size());
  std::cout << text;
  return 0;
} catch (const std::exception &e) {
  std::cerr << "PREPARE81 FAILED: " << e.what() << '\n';
  return 1;
}
