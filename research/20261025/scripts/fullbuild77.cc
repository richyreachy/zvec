#define main graph65_program_main
#include "graph65_refine.cc"
#undef main
#include <sys/resource.h>
#include <cerrno>
#include <filesystem>
#include "turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h"
#include "coupled73_codec.h"
#include "independent76.h"
uint64_t read64(const char *p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}
void put32(char *p, uint32_t v) {
  std::memcpy(p, &v, 4);
}
void put64(char *p, uint64_t v) {
  std::memcpy(p, &v, 8);
}
void write_at(int fd, const void *memory, size_t count, uint64_t offset) {
  const char *p = static_cast<const char *>(memory);
  while (count) {
    ssize_t n = pwrite(fd, p, count, offset);
    if (n < 0 && errno == EINTR) continue;
    check(n > 0, "pwrite");
    p += n;
    count -= n;
    offset += n;
  }
}
int main(int argc, char **argv) try {
  check(
      argc == 7,
      "usage: fullbuild77 graph rotated proxies header exclusive_dir threads");
  auto started = std::chrono::steady_clock::now();
  Map graph(argv[1]), rotated(argv[2]), proxies(argv[3]), header(argv[4]);
  check(graph.size >= 32 && u32(graph.p) == 0x51474764 &&
            u32(graph.p + 4) == 1 && u32(graph.p + 12) == D &&
            u32(graph.p + 16) == K,
        "graph header");
  size_t n = u32(graph.p + 8);
  check(graph.size == 32 + n * 140 && rotated.size == n * D * 4 &&
            proxies.size == n * 16 * 4,
        "input sizes");
  check(header.size == 4096 && std::memcmp(header.p, "QGIC77\0\0", 8) == 0 &&
            u32(header.p + 8) == 1 && u32(header.p + 12) <= 4064 &&
            read64(header.p + 24) == 3600,
        "model header");
  size_t count = read64(header.p + 16), threads = std::stoul(argv[6]);
  check(count > 0 && count <= n && threads > 0 && threads <= 128,
        "count/threads");
  std::filesystem::path output(argv[5]);
  check(std::filesystem::create_directory(output), "exclusive directory");
  std::array<std::string, 5> names = {"blocks.partial", "compact.u8",
                                      "audit.bin", "history.f64", "moves.u16"};
  std::array<int, 5> out;
  for (size_t i = 0; i < 5; ++i) {
    out[i] =
        open((output / names[i]).c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    check(out[i] >= 0, "exclusive output file");
  }
  write_at(out[0], header.p, 4096, 0);
  const float *vectors = reinterpret_cast<const float *>(rotated.p);
  std::atomic<size_t> finished{0}, learned{0}, native{0};
  std::atomic<uint64_t> updates{0};
  std::atomic<bool> cancel{false};
  std::exception_ptr failure;
  std::mutex lock;
  auto worker = [&](size_t thread) {
    try {
      zvec::turbo::SymphonyQGBlockQuantizer codec(D, true);
      check(codec.batch_bytes() == 3328, "codec size");
      std::vector<float> neighbors(K * D);
      std::vector<double> residuals(K * D), q(16 * D);
      std::vector<int8_t> signs(K * D);
      for (size_t row = thread; row < count && !cancel.load(); row += threads) {
        const char *g = graph.p + 32 + row * 140;
        size_t degree = u32(g + 8);
        check(degree <= K, "degree");
        const float *center = vectors + row * D;
        std::array<char, 3600> record{};
        put64(record.data(), read64(g));
        put32(record.data() + 8, degree);
        put32(record.data() + 12, degree == K ? 1 : 0);
        for (size_t i = 0; i < K; ++i)
          put64(record.data() + 16 + 8 * i, UINT64_MAX);
        for (size_t i = 0; i < degree; ++i) {
          size_t id = u32(g + 12 + 4 * i);
          check(id < n && id != row, "neighbor ID");
          put64(record.data() + 16 + 8 * i, read64(graph.p + 32 + id * 140));
          std::copy_n(vectors + id * D, D, neighbors.data() + i * D);
        }
        for (size_t i = 0; i < degree; ++i)
          for (size_t k = 0; k < D; ++k) {
            residuals[i * D + k] = double(neighbors[i * D + k]) - center[k];
            signs[i * D + k] = residuals[i * D + k] > 0 ? 1 : -1;
          }
        std::array<double, 257> history;
        history.fill(std::numeric_limits<double>::quiet_NaN());
        std::array<uint16_t, 768> moves;
        moves.fill(UINT16_MAX);
        uint32_t accepted = 0;
        double lambda = 0, initial = 0, final = 0, max_scale = 0,
               max_radial = 0, max_fp32_radial = 0;
        if (degree == K) {
          for (size_t j = 0; j < 16; ++j) {
            size_t id = u32(proxies.p + 4 * (row * 16 + j));
            check(id < n && id != row, "proxy ID");
            for (size_t i = 0; i < degree; ++i)
              check(id != u32(g + 12 + 4 * i), "proxy excludes neighbors");
            for (size_t prev = 0; prev < j; ++prev)
              check(id != u32(proxies.p + 4 * (row * 16 + prev)),
                    "unique proxies");
            for (size_t k = 0; k < D; ++k) {
              double v = (double(vectors[id * D + k]) - center[k]) / 4;
              q[j * D + k] = v;
              lambda += v * v;
            }
          }
          lambda /= D;
          independent76::Problem problem(D, 16, q.data(), lambda);
          auto model = independent76::optimize(problem, residuals.data(), K);
          signs = model.signs;
          accepted = model.moves.size() / 3;
          initial = model.objectives.front();
          final = model.objectives.back();
          check(accepted <= 256, "update budget");
          std::copy(model.objectives.begin(), model.objectives.end(),
                    history.begin());
          for (size_t j = 0; j < model.moves.size(); ++j)
            moves[j] = model.moves[j];
          coupled73::encode(center, neighbors.data(), signs.data(), degree, D,
                            record.data() + 272);
          ++learned;
        } else {
          for (size_t j = 0; j < 16; ++j)
            check(u32(proxies.p + 4 * (row * 16 + j)) == UINT32_MAX,
                  "short proxy sentinel");
          codec.encode(center, neighbors.data(), degree, record.data() + 272);
          ++native;
        }
        rabitqlib::ConstQGBatchDataMap<float> batch(record.data() + 272, D);
        std::array<uint8_t, 3072> compact{};
        for (size_t i = 0; i < degree; ++i) {
          double norm = 0, u = 0, l1 = 0;
          for (size_t k = 0; k < D; ++k) {
            double r = residuals[i * D + k];
            check(std::isfinite(r), "finite residual");
            norm += r * r;
            u += r * signs[i * D + k];
            l1 += std::abs(r);
            if (signs[i * D + k] > 0)
              compact[i * D / 8 + k / 8] |= uint8_t(1u << (7 - k % 8));
          }
          check(norm == 0 || u > 0, "radial denominator");
          double alpha = norm ? norm / u : 0;
          max_scale = std::max(max_scale, norm ? l1 / u : 1.);
          max_radial = std::max(max_radial, std::abs(alpha * u - norm));
          check(std::isfinite(batch.f_add()[i]) &&
                    std::isfinite(batch.f_rescale()[i]),
                "finite factors");
          max_fp32_radial = std::max(
              max_fp32_radial, std::abs(-.5 * batch.f_rescale()[i] * u - norm));
        }
        check(max_scale <= 1.02 + 1e-12 && max_radial < 1e-12 &&
                  max_fp32_radial < 1e-5,
              "numeric bounds");
        std::array<char, 64> audit{};
        put32(audit.data(), accepted);
        put32(audit.data() + 4, degree == K ? 1 : 0);
        std::array<double, 7> values = {
            lambda,     initial,         final,         max_scale,
            max_radial, max_fp32_radial, double(degree)};
        std::memcpy(audit.data() + 8, values.data(), 56);
        write_at(out[0], record.data(), record.size(),
                 4096 + row * record.size());
        write_at(out[1], compact.data(), compact.size(), row * compact.size());
        write_at(out[2], audit.data(), audit.size(), row * audit.size());
        write_at(out[3], history.data(), sizeof(history),
                 row * sizeof(history));
        write_at(out[4], moves.data(), sizeof(moves), row * sizeof(moves));
        updates += accepted;
        size_t done = ++finished;
        if (done % 8192 == 0) {
          std::lock_guard<std::mutex> guard(lock);
          std::fprintf(stderr, "FULLBUILD77 completed=%zu/%zu seconds=%.3f\n",
                       done, count,
                       std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - started)
                           .count());
        }
      }
    } catch (...) {
      cancel = true;
      std::lock_guard<std::mutex> guard(lock);
      if (!failure) failure = std::current_exception();
    }
  };
  std::vector<std::thread> workers;
  for (size_t t = 0; t < threads; ++t) workers.emplace_back(worker, t);
  for (auto &thread : workers) thread.join();
  if (failure) std::rethrow_exception(failure);
  check(finished == count, "all records completed");
  for (int fd : out)
    check(fsync(fd) == 0 && close(fd) == 0, "sync/close output");
  struct rusage usage {};
  check(getrusage(RUSAGE_SELF, &usage) == 0, "rusage");
  std::printf(
      "{\"centers\":%zu,\"threads\":%zu,\"learned_blocks\":%zu,\"native_"
      "blocks\":%zu,\"updates\":%llu,\"elapsed_seconds\":%.9f,\"peak_rss_kib\":"
      "%ld,\"requires_independent_validation\":true,\"no_retrieval_qps_claim\":"
      "true}\n",
      count, threads, learned.load(), native.load(),
      static_cast<unsigned long long>(updates.load()),
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count(),
      usage.ru_maxrss);
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
