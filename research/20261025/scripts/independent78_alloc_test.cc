#include <atomic>
#include <cstdio>
#include <new>
#include "independent78_import.h"
static std::atomic<size_t> allocations{0};
static std::atomic<bool> tracking{false};
void *operator new(size_t n) {
  if (tracking.load(std::memory_order_relaxed))
    allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void *operator new[](size_t n) {
  return ::operator new(n);
}
void operator delete(void *p) noexcept {
  std::free(p);
}
void operator delete[](void *p) noexcept {
  std::free(p);
}
void operator delete(void *p, size_t) noexcept {
  std::free(p);
}
void operator delete[](void *p, size_t) noexcept {
  std::free(p);
}
int main(int argc, char **argv) try {
  independent78::require(
      argc == 7, independent78::ErrorCode::state,
      "usage: alloc_test model raw rotated header digest count");
  independent78::Config c;
  c.model_path = argv[1];
  c.raw_vectors_path = argv[2];
  c.rotated_path = argv[3];
  c.trusted_header_path = argv[4];
  c.model_sha256 = argv[5];
  c.docs = std::stoull(argv[6]);
  independent78::Import loader;
  loader.open(c);
  independent78::detail::Mapping model(c.model_path), raw(c.raw_vectors_path),
      rotated(c.rotated_path);
  size_t checks = 0;
  std::array<char, 3328> codes;
  std::array<uint64_t, 32> neighbors;
  tracking = true;
  for (size_t repeat = 0; repeat < 100; ++repeat)
    for (size_t id = 0; id < c.docs; ++id) {
      const uint8_t *row = model.data() + 4096 + id * 3600;
      auto key = independent78::detail::u64(row);
      auto degree = independent78::detail::u32(row + 8);
      loader.verify_vectors(
          id, key,
          reinterpret_cast<const float *>(raw.data() + 16 + id * 768 * 4),
          reinterpret_cast<const float *>(rotated.data() + id * 768 * 4));
      for (size_t j = 0; j < degree; ++j)
        neighbors[j] = independent78::detail::u64(row + 16 + 8 * j);
      bool copied = loader.copy_full_block(id, key, neighbors.data(), degree,
                                           codes.data(), codes.size());
      independent78::require(copied == (degree == 32),
                             independent78::ErrorCode::state, "copy mode");
      ++checks;
    }
  tracking = false;
  independent78::require(allocations == 0, independent78::ErrorCode::state,
                         "success path heap allocation");
  std::printf(
      "{\"verified_vector_pairs\":%zu,\"heap_allocations_during_successful_"
      "verification_and_copy\":%zu}\n",
      checks, allocations.load());
  return 0;
} catch (const std::exception &e) {
  tracking = false;
  std::fprintf(stderr, "ERROR %s\n", e.what());
  return 1;
}
