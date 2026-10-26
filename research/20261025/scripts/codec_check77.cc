#define main graph65_program_main
#include "graph65_refine.cc"
#undef main
#include "turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h"
#include "coupled73_codec.h"
uint64_t read64(const char *p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}
int main(int argc, char **argv) try {
  check(argc == 6,
        "usage: codec_check77 graph rotated proxies model_dir model_filename");
  Map graph(argv[1]), rotated(argv[2]), proxies(argv[3]);
  std::string dir = argv[4];
  Map model(dir + "/" + argv[5]), compact(dir + "/compact.u8");
  check(model.size >= 4096 && std::memcmp(model.p, "QGIC77\0\0", 8) == 0 &&
            u32(model.p + 8) == 1 && read64(model.p + 24) == 3600,
        "model header");
  size_t count = read64(model.p + 16), n = u32(graph.p + 8);
  check(count <= n && model.size == 4096 + count * 3600 &&
            compact.size == count * 3072 && rotated.size == n * D * 4 &&
            proxies.size == n * 16 * 4,
        "sizes");
  const float *vectors = reinterpret_cast<const float *>(rotated.p);
  zvec::turbo::SymphonyQGBlockQuantizer codec(D, true);
  std::vector<float> neighbors(K * D);
  std::vector<int8_t> signs(K * D);
  std::vector<char> code(3328);
  size_t native = 0, learned = 0, scalar_checks = 0;
  std::array<float, 32> scores;
  for (size_t row = 0; row < count; ++row) {
    const char *g = graph.p + 32 + row * 140;
    const char *saved = model.p + 4096 + row * 3600;
    size_t degree = u32(g + 8);
    check(read64(saved) == read64(g) && u32(saved + 8) == degree &&
              u32(saved + 12) == (degree == K ? 1 : 0),
          "key/degree/flag");
    const float *center = vectors + row * D;
    std::fill(neighbors.begin(), neighbors.end(), 0);
    std::fill(code.begin(), code.end(), 0);
    for (size_t i = 0; i < K; ++i) {
      if (i < degree) {
        size_t id = u32(g + 12 + 4 * i);
        check(id < n &&
                  read64(saved + 16 + 8 * i) == read64(graph.p + 32 + id * 140),
              "neighbor key/order");
        std::copy_n(vectors + id * D, D, neighbors.data() + i * D);
      } else
        check(read64(saved + 16 + 8 * i) == UINT64_MAX, "neighbor padding");
      for (size_t k = 0; k < D; ++k) {
        auto byte =
            static_cast<uint8_t>(compact.p[row * 3072 + i * 96 + k / 8]);
        signs[i * D + k] = (byte & (1u << (7 - k % 8))) ? 1 : -1;
        if (i >= degree) check(byte == 0, "compact padding");
      }
    }
    if (degree == K) {
      coupled73::encode(center, neighbors.data(), signs.data(), degree, D,
                        code.data());
      ++learned;
    } else {
      codec.encode(center, neighbors.data(), degree, code.data());
      ++native;
      for (size_t i = 0; i < degree; ++i)
        for (size_t k = 0; k < D; ++k)
          check(signs[i * D + k] == (neighbors[i * D + k] > center[k] ? 1 : -1),
                "short native signs");
    }
    check(std::memcmp(code.data(), saved + 272, 3328) == 0,
          "actual codec byte identity");
    if (row < 128 && degree) {
      rabitqlib::ConstQGBatchDataMap<float> factors(saved + 272, D);
      for (size_t j = 0; j < 4; ++j) {
        size_t id =
            degree == K ? u32(proxies.p + 4 * (row * 16 + j)) : (row + j) % n;
        check(id < n, "query id");
        const float *query = vectors + id * D;
        zvec::turbo::SymphonyQGBlockQuantizer::Query lookup;
        lookup.lookup.reset(query, D);
        float dot = 0;
        for (size_t k = 0; k < D; ++k) dot += query[k] * center[k];
        float center_distance = 1 - dot;
        codec.scan(saved + 272, lookup, center_distance, scores.data());
        for (size_t i = 0; i < degree; ++i) {
          uint32_t sum = 0;
          for (size_t book = 0; book < D / 4; ++book) {
            unsigned nibble = 0;
            for (size_t k = 0; k < 4; ++k)
              if (signs[i * D + book * 4 + k] > 0) nibble |= 1u << (3 - k);
            sum += lookup.lookup.lut()[book * 16 + nibble];
          }
          float value = std::fma(lookup.lookup.delta(), float(sum),
                                 lookup.lookup.sum_vl_lut());
          value += lookup.lookup.k1xsumq();
          float expected = std::fma(factors.f_rescale()[i], value,
                                    factors.f_add()[i] + center_distance);
          check(std::isfinite(scores[i]) &&
                    std::memcmp(&expected, &scores[i], 4) == 0,
                "scalar LUT/FMA score");
          ++scalar_checks;
        }
      }
    }
  }
  std::printf(
      "{\"centers\":%zu,\"learned_code_blocks_byte_identical\":%zu,\"native_"
      "short_code_blocks_byte_identical\":%zu,\"all_keys_and_neighbor_order_"
      "checked\":true,\"scalar_lut_fma_bit_identical_scores\":%zu}\n",
      count, learned, native, scalar_checks);
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
