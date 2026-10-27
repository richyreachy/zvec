#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace zvec::core {
// Diagnostic-only, native little-endian trace. One writer thread per path;
// exclusive creation rejects accidental collisions and preserves artifacts.
struct QuantizedGraphMultiTrace {
  struct Record {
    uint32_t query, kind, step, source, target;
    float estimate, exact;
    uint32_t reserved;
  };
  static_assert(sizeof(Record) == 32, "trace layout");
  struct Writer {
    FILE *file{nullptr};
    bool failed{false};
    uint64_t serial{0};
    explicit Writer(const char *path) {
      if (!path) return;
      file = std::fopen(path, "wbx");
      if (!file) {
        failed = true;
        return;
      }
      const uint32_t h[4] = {0x51474d68, 1, 32, 768};
      failed = std::fwrite(h, sizeof(h), 1, file) != 1;
    }
    ~Writer() {
      if (file) std::fclose(file);
    }
    bool write(const Record &r) {
      if (failed) return false;
      if (!file) return true;
      failed = std::fwrite(&r, sizeof(r), 1, file) != 1;
      return !failed;
    }
    bool flush() {
      if (file && std::fflush(file) != 0) failed = true;
      return !failed;
    }
  };
  static Writer &writer() {
    static thread_local Writer sink(std::getenv("QG_MULTI_TRACE_FILE"));
    return sink;
  }
  Writer &sink;
  uint32_t query;
  QuantizedGraphMultiTrace() : sink(writer()), query(uint32_t(sink.serial++)) {
    if (sink.serial > uint64_t(UINT32_MAX) + 1) sink.failed = true;
  }
  bool ready() const {
    return !sink.failed;
  }
  bool active() const {
    return sink.file && !sink.failed;
  }
  bool event(uint32_t kind, uint32_t step, uint32_t source, uint32_t target,
             float estimate, float exact) {
    return sink.write({query, kind, step, source, target, estimate, exact, 0});
  }
  bool end(uint32_t steps) {
    return event(3, steps, UINT32_MAX, UINT32_MAX, 0, 0) && sink.flush();
  }
};
}  // namespace zvec::core
