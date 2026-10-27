#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include "independent76.h"
using independent76::require;
template <class T>
void save(const std::filesystem::path &path, const std::vector<T> &values) {
  FILE *f = std::fopen(path.c_str(), "wbx");
  require(f, "exclusive output");
  require(
      std::fwrite(values.data(), sizeof(T), values.size(), f) == values.size(),
      "write output");
  require(std::fclose(f) == 0, "close output");
}
int main(int argc, char **argv) try {
  require(argc == 4,
          "usage: builder76 fit.f32 exclusive_output_directory threads");
  const auto started = std::chrono::steady_clock::now();
  constexpr size_t D = 768, K = 32, P = 16, V = 49, Max = K * 8;
  int fd = open(argv[1], O_RDONLY);
  require(fd >= 0, "input open");
  struct stat st {};
  require(
      fstat(fd, &st) == 0 && st.st_size > 0 && st.st_size % (V * D * 4) == 0,
      "input shape");
  size_t n = st.st_size / (V * D * 4);
  void *memory = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  require(memory != MAP_FAILED, "input mmap");
  const float *input = static_cast<const float *>(memory);
  std::filesystem::path out(argv[2]);
  require(std::filesystem::create_directory(out), "exclusive output directory");
  size_t threads = std::stoul(argv[3]);
  require(threads > 0 && threads <= 128, "thread count");
  std::vector<int8_t> signs(n * K * D);
  std::vector<double> alpha(n * K),
      history(n * (Max + 1), std::numeric_limits<double>::quiet_NaN()),
      lambda(n);
  std::vector<uint32_t> moves(n * Max * 3, UINT32_MAX), counts(n);
  std::exception_ptr failure;
  std::mutex lock;
  const auto compute_start = std::chrono::steady_clock::now();
  auto worker = [&](size_t t) {
    try {
      std::vector<double> r(K * D), q(P * D);
      for (size_t row = t; row < n; row += threads) {
        const float *x = input + row * V * D;
        double lam = 0;
        for (size_t i = 0; i < K; ++i)
          for (size_t k = 0; k < D; ++k)
            r[i * D + k] = double(x[(i + 1) * D + k]) - double(x[k]);
        for (size_t j = 0; j < P; ++j)
          for (size_t k = 0; k < D; ++k) {
            double v = (double(x[(K + 1 + j) * D + k]) - double(x[k])) / 4;
            q[j * D + k] = v;
            lam += v * v;
          }
        lam /= D;
        lambda[row] = lam;
        independent76::Problem a(D, P, q.data(), lam);
        auto model = independent76::optimize(a, r.data(), K);
        std::copy(model.signs.begin(), model.signs.end(),
                  signs.begin() + row * K * D);
        std::copy(model.alphas.begin(), model.alphas.end(),
                  alpha.begin() + row * K);
        std::copy(model.objectives.begin(), model.objectives.end(),
                  history.begin() + row * (Max + 1));
        std::copy(model.moves.begin(), model.moves.end(),
                  moves.begin() + row * Max * 3);
        counts[row] = model.moves.size() / 3;
      }
    } catch (...) {
      std::lock_guard<std::mutex> guard(lock);
      if (!failure) failure = std::current_exception();
    }
  };
  std::vector<std::thread> workers;
  for (size_t t = 0; t < threads; ++t) workers.emplace_back(worker, t);
  for (auto &thread : workers) thread.join();
  if (failure) std::rethrow_exception(failure);
  double compute = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - compute_start)
                       .count();
  save(out / "signs.i8", signs);
  save(out / "alpha.f64", alpha);
  save(out / "history.f64", history);
  save(out / "moves.u32", moves);
  save(out / "counts.u32", counts);
  save(out / "lambda.f64", lambda);
  uint64_t updates = 0;
  for (auto count : counts) updates += count;
  struct rusage usage {};
  require(getrusage(RUSAGE_SELF, &usage) == 0, "rusage");
  double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  std::printf(
      "{\"centers\":%zu,\"threads\":%zu,\"updates\":%llu,\"compute_seconds\":%."
      "9f,\"elapsed_seconds\":%.9f,\"peak_rss_kib\":%ld,\"no_retrieval_qps_"
      "claim\":true}\n",
      n, threads, static_cast<unsigned long long>(updates), compute, elapsed,
      usage.ru_maxrss);
  require(munmap(memory, st.st_size) == 0 && close(fd) == 0, "input close");
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
