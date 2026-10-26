#include <fstream>
#include "independent78_import.h"
static thread_local std::string last_error;
static thread_local int last_code = 0;
static thread_local uint64_t last_row = UINT64_MAX;
static void remember() {
  try {
    throw;
  } catch (const independent78::Error &e) {
    last_error = e.what();
    last_code = int(e.code());
    last_row = e.row();
  } catch (const std::exception &e) {
    last_error = e.what();
    last_code = 100;
    last_row = UINT64_MAX;
  }
}
extern "C" const char *error78() {
  return last_error.c_str();
}
extern "C" int error_code78() {
  return last_code;
}
extern "C" uint64_t error_row78() {
  return last_row;
}
extern "C" void *create78() {
  return new independent78::Import;
}
extern "C" void destroy78(void *object) {
  delete static_cast<independent78::Import *>(object);
}
extern "C" int loaded78(void *object) {
  return static_cast<independent78::Import *>(object)->loaded();
}
extern "C" void clear78(void *object) {
  static_cast<independent78::Import *>(object)->clear();
}
extern "C" int load78(void *object, const char *model, const char *raw,
                      const char *rotated, const char *header,
                      const char *digest, size_t docs, size_t dims,
                      size_t degree, int cosine) try {
  independent78::Config c;
  c.model_path = model;
  c.raw_vectors_path = raw;
  c.rotated_path = rotated;
  c.trusted_header_path = header;
  c.model_sha256 = digest;
  c.docs = docs;
  c.dimension = dims;
  c.max_neighbors = degree;
  c.cosine = cosine;
  static_cast<independent78::Import *>(object)->open(c);
  return 0;
} catch (...) {
  remember();
  return -1;
}
extern "C" int verify78(void *object, uint32_t id, uint64_t key,
                        const float *raw, const float *rotated,
                        size_t dims) try {
  static_cast<independent78::Import *>(object)->verify_vectors(id, key, raw,
                                                               rotated, dims);
  return 0;
} catch (...) {
  remember();
  return -1;
}
extern "C" int copy78(void *object, uint32_t id, uint64_t key,
                      const uint64_t *neighbors, size_t degree, char *out,
                      size_t bytes) try {
  return static_cast<independent78::Import *>(object)->copy_full_block(
             id, key, neighbors, degree, out, bytes)
             ? 1
             : 0;
} catch (...) {
  remember();
  return -1;
}
extern "C" int move78(void *object) try {
  auto &original = *static_cast<independent78::Import *>(object);
  independent78::Import next(std::move(original));
  independent78::require(!original.loaded(), independent78::ErrorCode::state,
                         "moved source still loaded");
  original = std::move(next);
  independent78::require(!next.loaded() && original.loaded(),
                         independent78::ErrorCode::state, "move assignment");
  return 0;
} catch (...) {
  remember();
  return -1;
}
extern "C" int sha78(const char *path, size_t chunk, char *out) try {
  independent78::require(chunk > 0 && chunk <= 1048576,
                         independent78::ErrorCode::digest, "chunk");
  std::ifstream f(path, std::ios::binary);
  independent78::require(bool(f), independent78::ErrorCode::io,
                         "SHA test open");
  std::vector<char> bytes(chunk);
  independent78::Sha256 hash;
  hash.update(nullptr, 0);
  while (f) {
    f.read(bytes.data(), bytes.size());
    hash.update(bytes.data(), f.gcount());
  }
  independent78::require(f.eof(), independent78::ErrorCode::io,
                         "SHA test read");
  auto digest = hash.hex();
  independent78::require(hash.hex() == digest, independent78::ErrorCode::digest,
                         "digest mutates state");
  std::memcpy(out, digest.c_str(), 65);
  return 0;
} catch (...) {
  remember();
  return -1;
}

extern "C" int number78(const char *text, double *out) try {
  *out = independent78::detail::JsonParser(text).parse().number();
  return 0;
} catch (...) {
  remember();
  return -1;
}
extern "C" int sha_extend78(const char *data, size_t size, char *out) try {
  independent78::Sha256 hash;
  size_t split = size / 2;
  hash.update(data, split);
  auto before = hash.hex();
  independent78::require(before == hash.hex(), independent78::ErrorCode::digest,
                         "repeat digest");
  hash.update(data + split, size - split);
  auto digest = hash.hex();
  std::memcpy(out, digest.c_str(), 65);
  return 0;
} catch (...) {
  remember();
  return -1;
}
