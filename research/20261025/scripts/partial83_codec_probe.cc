// Actual codec integration on frozen training queries; no search or timing.
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h>

#if !RABITQ_SUPPORTED || !defined(QG_ROTATION_768)
#error Probe requires actual native 768-dimensional codec.
#endif

namespace {
constexpr uint32_t D = 768, K = 32, B = 3328, QN = 3072, NS = 6625, NC = 20;
#if defined(QG_PARTIAL_SCAN_DIM)
constexpr uint32_t M = QG_PARTIAL_SCAN_DIM;
constexpr bool defined_partial = true;
#else
constexpr uint32_t M = D;
constexpr bool defined_partial = false;
#endif
using Codec = zvec::turbo::SymphonyQGBlockQuantizer;
void need(bool value, const std::string &message) {
  if (!value) throw std::runtime_error(message);
}
template <class T>
T take(const char *p) {
  T value;
  std::memcpy(&value, p, sizeof(value));
  return value;
}
uint32_t bits(float value) {
  return take<uint32_t>(reinterpret_cast<const char *>(&value));
}
float multiply(float a, float b) {
  volatile float value = a * b;
  return value;
}
struct Map {
  int fd = -1;
  size_t size = 0;
  const char *p = nullptr;
  explicit Map(const std::string &path) {
    fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    need(fd >= 0, "open " + path);
    struct stat st {};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0) {
      ::close(fd);
      throw std::runtime_error("stat " + path);
    }
    size = size_t(st.st_size);
    void *v = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (v == MAP_FAILED) {
      ::close(fd);
      throw std::runtime_error("mmap " + path);
    }
    p = static_cast<const char *>(v);
  }
  Map(const Map &) = delete;
  Map &operator=(const Map &) = delete;
  ~Map() {
    if (p) ::munmap(const_cast<char *>(p), size);
    if (fd >= 0) ::close(fd);
  }
};
void save(const std::string &path, const void *p, size_t n) {
  FILE *f = std::fopen(path.c_str(), "wbx");
  need(f != nullptr, "exclusive file " + path);
  bool wrote = std::fwrite(p, 1, n, f) == n;
  bool closed = std::fclose(f) == 0;
  need(wrote && closed, "write " + path);
}
void header(const Map &f, uint32_t magic, uint32_t bytes, uint32_t rows,
            uint32_t models) {
  const uint32_t expected[16] = {magic, 1, 64, bytes, D, K, rows, models,
                                 0,     0, 0,  0,     0, 0, 0,    0};
  need(f.size >= 64 && std::memcmp(f.p, expected, 64) == 0,
       "frozen training header");
}
struct Prepared {
  Codec::Query actual;
  zvec::turbo::SymQuery reference;
  float partial_delta = 0, partial_k1 = 0;
  uint32_t serial = 0;
};
}  // namespace

int main(int argc, char **argv) try {
  need(argc == 6, "usage: probe states queries scores native_blocks output");
  need(std::numeric_limits<float>::is_iec559 &&
           std::fegetround() == FE_TONEAREST,
       "IEEE FP32 round-to-nearest");
  Map states(argv[1]), queries(argv[2]), saved(argv[3]), blocks(argv[4]);
  const std::string out = argv[5];
  header(states, 0x51474181, 3136, NS, 137);
  header(queries, 0x51475181, 6164, QN, 0);
  header(saved, 0x51474581, 128, NS, 137);
  need(states.size == 64 + NS * 3136ULL && queries.size == 64 + QN * 6164ULL &&
           saved.size == 64 + NS * 137ULL * K * 4 && blocks.size == NC * B,
       "frozen sizes");
  Codec codec(D, true);
  need(codec.encoded_dim() == D && codec.quantized_dim() == D &&
           codec.batch_bytes() == B && Codec::kBatchSize == K,
       "full original block geometry");
  std::vector<std::unique_ptr<Prepared>> bank;
  uint64_t rotations = 0, lut_bytes = 0, query_constants = 0,
           partial_fields = 0;
  for (uint32_t qi = 0; qi < QN; ++qi) {
    const char *record = queries.p + 64 + qi * 6164ULL;
    auto query = std::make_unique<Prepared>();
    query->serial = take<uint32_t>(record);
    const auto *prepared = reinterpret_cast<const float *>(record + 16);
    const auto *rotated =
        reinterpret_cast<const float *>(record + 16 + (D + 1) * 4);
    codec.prepare_query(prepared, query->actual);
    need(query->actual.rotated.size() == D &&
             std::memcmp(query->actual.rotated.data(), rotated, D * 4) == 0,
         "actual prepare_query rotation bits query=" + std::to_string(qi));
    rotations += D;
    query->reference.reset(rotated, D);
    need(std::memcmp(query->actual.lookup.lut(), query->reference.lut(),
                     D * 4) == 0,
         "full LUT bytes");
    lut_bytes += D * 4;
    need(bits(query->actual.lookup.delta()) == bits(query->reference.delta()) &&
             bits(query->actual.lookup.sum_vl_lut()) ==
                 bits(query->reference.sum_vl_lut()) &&
             bits(query->actual.lookup.k1xsumq()) ==
                 bits(query->reference.k1xsumq()) &&
             bits(query->actual.lookup.g_add()) == bits(0.0f),
         "full query constants");
    query_constants += 4;
    if (M < D) {
      zvec::turbo::SymQuery prefix;
      prefix.reset(rotated, M);
      const float rho = float(D) / float(M);
      query->partial_delta = multiply(rho, query->reference.delta());
      query->partial_k1 = multiply(rho, prefix.k1xsumq());
    } else {
      query->partial_delta = query->reference.delta();
      query->partial_k1 = query->reference.k1xsumq();
    }
#if defined(QG_PARTIAL_SCAN_DIM) && QG_PARTIAL_SCAN_DIM < 768
    need(
        bits(query->actual.partial_delta) == bits(query->partial_delta) &&
            bits(query->actual.partial_k1xsumq) == bits(query->partial_k1),
        "actual lightweight prefix fields versus reference prefix LUT builder");
    partial_fields += 2;
#endif
    bank.push_back(std::move(query));
  }
  uint64_t native_checks = 0, actual_checks = 0, full_checks = 0;
  std::vector<uint32_t> output;
  output.reserve(NS * K);
  for (uint32_t si = 0; si < NS; ++si) {
    const char *state = states.p + 64 + si * 3136ULL;
    const uint32_t ci = take<uint32_t>(state + 40),
                   qi = take<uint32_t>(state + 56);
    need(ci < NC && qi < QN && bank[qi]->serial == take<uint32_t>(state),
         "state query mapping");
    auto &query = *bank[qi];
    const float center = take<float>(state + 28);
    const char *code = blocks.p + ci * B;
    rabitqlib::ConstQGBatchDataMap<float> batch(code, D);
    std::array<float, K> native{}, expected{}, actual{};
    need(zvec::turbo::symqg_scan(batch.bin_code(), query.reference.lut(), D,
                                 batch.f_add(), batch.f_rescale(), center,
                                 query.reference.delta(),
                                 query.reference.sum_vl_lut(),
                                 query.reference.k1xsumq(), native.data()),
         "actual native AVX512 reference");
    need(zvec::turbo::symqg_scan(
             batch.bin_code(), query.reference.lut(), M, batch.f_add(),
             batch.f_rescale(), center, query.partial_delta,
             query.reference.sum_vl_lut(), query.partial_k1, expected.data()),
         "actual shortened AVX512 reference");
    codec.scan(code, query.actual, center, actual.data());
    for (uint32_t lane = 0; lane < K; ++lane) {
      const uint32_t original = take<uint32_t>(state + 64 + lane * 96 + 4);
      need(bits(native[lane]) == original &&
               take<uint32_t>(saved.p + 64 +
                              (size_t(si) * 137 * K + lane) * 4) == original,
           "full native reference matches saved state/score bits");
      ++native_checks;
      need(std::isfinite(actual[lane]) &&
               bits(actual[lane]) == bits(expected[lane]),
           "actual codec scan reference bits state=" + std::to_string(si) +
               " lane=" + std::to_string(lane));
      ++actual_checks;
      if (M == D) {
        need(bits(actual[lane]) == original, "full codec saved native bits");
        ++full_checks;
      }
      output.push_back(bits(actual[lane]));
    }
  }
  save(out + "/actual_scores.u32", output.data(), output.size() * 4);
  std::ostringstream report;
  report << "{\"passed\":true,\"partial_macro_defined\":"
         << (defined_partial ? "true" : "false") << ",\"m\":" << M
         << ",\"queries\":" << QN << ",\"states\":" << NS
         << ",\"query_size_bytes\":" << sizeof(Codec::Query)
         << ",\"rotation_float_bit_checks\":" << rotations
         << ",\"full_lut_byte_checks\":" << lut_bytes
         << ",\"full_query_constant_bit_checks\":" << query_constants
         << ",\"partial_field_bit_checks\":" << partial_fields
         << ",\"full_native_reference_bit_checks\":" << native_checks
         << ",\"actual_codec_scan_bit_checks\":" << actual_checks
         << ",\"full_codec_saved_native_bit_checks\":" << full_checks
         << ",\"validation_or_truth_labels_used\":false,\"retrieval_or_QPS\":"
            "false}\n";
  const auto text = report.str();
  save(out + "/result.json", text.data(), text.size());
  return 0;
} catch (const std::exception &error) {
  std::cerr << "PARTIAL83 PROBE FAILED: " << error.what() << '\n';
  return 1;
}
