#include <unistd.h>
#include <cassert>
#include <cstring>
#include <filesystem>
#include "core/algorithm/quantized_graph/quantized_graph_multi_trace.h"
int main() {
  using T = zvec::core::QuantizedGraphMultiTrace;
  const std::string path = "/tmp/trace68_" + std::to_string(getpid()) + ".bin";
  {
    T::Writer w(nullptr);
    assert(!w.failed && !w.file);
    assert(w.write({0, 2, 1, 7, 9, .2f, .3f, 0}));
    assert(w.flush());
  }
  {
    T::Writer w("/nonexistent/trace68/file");
    assert(w.failed);
    assert(!w.write({}));
    assert(!w.flush());
  }
  {
    T::Writer w(path.c_str());
    assert(!w.failed && w.file);
    assert(w.write({0, 1, 1, 7, 7, .3f, .3f, 0}));
    assert(w.write({0, 2, 1, 7, 9, .2f, .3f, 0}));
    assert(w.write({0, 3, 1, UINT32_MAX, UINT32_MAX, 0, 0, 0}));
    assert(w.flush());
    T::Writer duplicate(path.c_str());
    assert(duplicate.failed);
  }
  FILE *f = fopen(path.c_str(), "rb");
  assert(f);
  uint32_t header[4];
  assert(fread(header, 16, 1, f) == 1);
  assert(header[0] == 0x51474d68 && header[1] == 1 && header[2] == 32 &&
         header[3] == 768);
  T::Record r[3];
  assert(fread(r, 32, 3, f) == 3);
  assert(fgetc(f) == EOF);
  assert(fclose(f) == 0);
  assert(r[0].kind == 1 && r[1].kind == 2 && r[2].kind == 3);
  assert(r[1].source == 7 && r[1].target == 9 && r[1].estimate == .2f &&
         r[1].exact == .3f);
  assert(r[2].target == UINT32_MAX);
  assert(std::filesystem::remove(path));
  puts(
      "TRACE68 writer: absent path, failed open, exclusive collision, header "
      "and3records passed");
}
