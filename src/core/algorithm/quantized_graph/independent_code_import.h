#pragma once
// Standalone research loader for the accepted phase77 immutable code format.
// No query hot-path changes and no graph ownership/lifecycle policy live here.
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace independent78 {
enum class ErrorCode {
  io = 1,
  header = 2,
  digest = 3,
  record = 4,
  compatibility = 5,
  state = 6
};
class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, const std::string &message, uint64_t row = UINT64_MAX)
      : std::runtime_error(message), code_(code), row_(row) {}
  ErrorCode code() const noexcept {
    return code_;
  }
  uint64_t row() const noexcept {
    return row_;
  }

 private:
  ErrorCode code_;
  uint64_t row_;
};
inline void require(bool condition, ErrorCode code, std::string_view reason,
                    uint64_t row = UINT64_MAX) {
  if (!condition) throw Error(code, std::string(reason), row);
}

class Sha256 {
 public:
  void update(const void *data, size_t count) {
    require(data || !count, ErrorCode::digest, "null SHA256 input");
    if (!count) return;
    require(count <= (UINT64_MAX / 8) - bytes_, ErrorCode::digest,
            "SHA256 length overflow");
    bytes_ += count;
    const auto *p = static_cast<const uint8_t *>(data);
    if (used_) {
      size_t n = std::min(count, 64 - used_);
      std::memcpy(buffer_.data() + used_, p, n);
      used_ += n;
      p += n;
      count -= n;
      if (used_ == 64) {
        transform(buffer_.data());
        used_ = 0;
      }
    }
    while (count >= 64) {
      transform(p);
      p += 64;
      count -= 64;
    }
    if (count) {
      std::memcpy(buffer_.data(), p, count);
      used_ = count;
    }
  }
  std::string hex() const {
    Sha256 copy = *this;
    uint64_t bits = copy.bytes_ * 8;
    copy.buffer_[copy.used_++] = 0x80;
    if (copy.used_ > 56) {
      std::fill(copy.buffer_.begin() + copy.used_, copy.buffer_.end(),
                uint8_t{0});
      copy.transform(copy.buffer_.data());
      copy.used_ = 0;
    }
    std::fill(copy.buffer_.begin() + copy.used_, copy.buffer_.begin() + 56,
              uint8_t{0});
    for (size_t i = 0; i < 8; ++i)
      copy.buffer_[63 - i] = uint8_t(bits >> (8 * i));
    copy.transform(copy.buffer_.data());
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint32_t word : copy.h_)
      for (int shift = 24; shift >= 0; shift -= 8) {
        uint8_t byte = uint8_t(word >> shift);
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 15]);
      }
    return out;
  }

 private:
  static uint32_t rotate(uint32_t v, unsigned shift) {
    return (v >> shift) | (v << (32 - shift));
  }
  void transform(const uint8_t *p) {
    static constexpr uint32_t constants[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (size_t i = 0; i < 16; ++i)
      w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
             (uint32_t(p[4 * i + 2]) << 8) | p[4 * i + 3];
    for (size_t i = 16; i < 64; ++i) {
      uint32_t a = rotate(w[i - 15], 7) ^ rotate(w[i - 15], 18) ^
                   (w[i - 15] >> 3),
               b = rotate(w[i - 2], 17) ^ rotate(w[i - 2], 19) ^
                   (w[i - 2] >> 10);
      w[i] = w[i - 16] + a + w[i - 7] + b;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5],
             g = h_[6], h = h_[7];
    for (size_t i = 0; i < 64; ++i) {
      uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25),
               choose = (e & f) ^ (~e & g),
               t1 = h + s1 + choose + constants[i] + w[i],
               s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22),
               majority = (a & b) ^ (a & c) ^ (b & c), t2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
  }
  std::array<uint32_t, 8> h_{{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
  std::array<uint8_t, 64> buffer_{};
  uint64_t bytes_ = 0;
  size_t used_ = 0;
};
namespace detail {
inline uint32_t u32(const uint8_t *p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}
inline uint64_t u64(const uint8_t *p) {
  return uint64_t(u32(p)) | (uint64_t(u32(p + 4)) << 32);
}
inline bool hex_digest(const std::string &s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
class Mapping {
 public:
  Mapping() = default;
  explicit Mapping(const std::string &path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(fd_ >= 0, ErrorCode::io, "open: " + path);
    struct stat s {};
    if (fstat(fd_, &s) || !S_ISREG(s.st_mode) || s.st_size <= 0 ||
        uintmax_t(s.st_size) > SIZE_MAX) {
      clear();
      throw Error(ErrorCode::io, "invalid regular file: " + path);
    }
    size_ = size_t(s.st_size);
    void *p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) {
      clear();
      throw Error(ErrorCode::io, "mmap: " + path);
    }
    data_ = static_cast<const uint8_t *>(p);
  }
  Mapping(const Mapping &) = delete;
  Mapping &operator=(const Mapping &) = delete;
  Mapping(Mapping &&rhs) noexcept {
    swap(rhs);
  }
  Mapping &operator=(Mapping &&rhs) noexcept {
    if (this != &rhs) {
      clear();
      swap(rhs);
    }
    return *this;
  }
  ~Mapping() {
    clear();
  }
  void clear() noexcept {
    if (data_) munmap(const_cast<uint8_t *>(data_), size_);
    if (fd_ >= 0) close(fd_);
    data_ = nullptr;
    size_ = 0;
    fd_ = -1;
  }
  void swap(Mapping &rhs) noexcept {
    std::swap(fd_, rhs.fd_);
    std::swap(data_, rhs.data_);
    std::swap(size_, rhs.size_);
  }
  const uint8_t *data() const {
    return data_;
  }
  size_t size() const {
    return size_;
  }
  std::string hash() const {
    Sha256 hash;
    hash.update(data_, size_);
    return hash.hex();
  }

 private:
  int fd_ = -1;
  const uint8_t *data_ = nullptr;
  size_t size_ = 0;
};
struct Json {
  enum class Type { object, string, number, boolean, null };
  Type type = Type::null;
  std::map<std::string, Json> fields;
  std::string text;
  const Json &field(const std::string &name) const {
    require(type == Type::object, ErrorCode::header, "JSON object required");
    auto i = fields.find(name);
    require(i != fields.end(), ErrorCode::header,
            "missing JSON field: " + name);
    return i->second;
  }
  std::string string() const {
    require(type == Type::string, ErrorCode::header, "JSON string required");
    return text;
  }
  uint64_t integer() const {
    require(type == Type::number && !text.empty() &&
                text.find_first_not_of("0123456789") == std::string::npos,
            ErrorCode::header, "JSON unsigned integer required");
    uint64_t out = 0;
    auto result = std::from_chars(text.data(), text.data() + text.size(), out);
    require(result.ec == std::errc{} && result.ptr == text.data() + text.size(),
            ErrorCode::header, "JSON integer overflow");
    return out;
  }
  double number() const {
    require(type == Type::number, ErrorCode::header, "JSON number required");
    double v = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(),
                                        v, std::chars_format::general);
    require(parsed.ec == std::errc{} &&
                parsed.ptr == text.data() + text.size() && std::isfinite(v),
            ErrorCode::header, "invalid finite JSON number");
    return v;
  }
  bool boolean() const {
    require(type == Type::boolean, ErrorCode::header, "JSON boolean required");
    return text == "true";
  }
};
class JsonParser {
 public:
  explicit JsonParser(std::string input) : input_(std::move(input)) {}
  Json parse() {
    Json result = value(0);
    space();
    require(at_ == input_.size(), ErrorCode::header, "trailing JSON content");
    return result;
  }

 private:
  void space() {
    while (at_ < input_.size() && (input_[at_] == ' ' || input_[at_] == '\t' ||
                                   input_[at_] == '\r' || input_[at_] == '\n'))
      ++at_;
  }
  char take() {
    require(at_ < input_.size(), ErrorCode::header, "truncated JSON");
    return input_[at_++];
  }
  bool consume(char c) {
    space();
    if (at_ < input_.size() && input_[at_] == c) {
      ++at_;
      return true;
    }
    return false;
  }
  uint32_t hex4() {
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) {
      char c = take();
      unsigned v = c >= '0' && c <= '9'   ? unsigned(c - '0')
                   : c >= 'a' && c <= 'f' ? unsigned(c - 'a' + 10)
                   : c >= 'A' && c <= 'F' ? unsigned(c - 'A' + 10)
                                          : 16;
      require(v < 16, ErrorCode::header, "invalid JSON unicode escape");
      out = 16 * out + v;
    }
    return out;
  }
  static void utf8(std::string &s, uint32_t cp) {
    if (cp < 128)
      s.push_back(char(cp));
    else if (cp < 2048) {
      s.push_back(char(0xc0 | (cp >> 6)));
      s.push_back(char(0x80 | (cp & 63)));
    } else if (cp < 65536) {
      s.push_back(char(0xe0 | (cp >> 12)));
      s.push_back(char(0x80 | ((cp >> 6) & 63)));
      s.push_back(char(0x80 | (cp & 63)));
    } else {
      s.push_back(char(0xf0 | (cp >> 18)));
      s.push_back(char(0x80 | ((cp >> 12) & 63)));
      s.push_back(char(0x80 | ((cp >> 6) & 63)));
      s.push_back(char(0x80 | (cp & 63)));
    }
  }
  std::string string() {
    require(take() == '"', ErrorCode::header, "JSON string delimiter");
    std::string out;
    while (true) {
      unsigned char c = static_cast<unsigned char>(take());
      if (c == '"') return out;
      require(c >= 32, ErrorCode::header, "JSON control character");
      if (c == '\\') {
        char e = take();
        switch (e) {
          case '"':
          case '\\':
          case '/':
            out.push_back(e);
            break;
          case 'b':
            out.push_back('\b');
            break;
          case 'f':
            out.push_back('\f');
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 'r':
            out.push_back('\r');
            break;
          case 't':
            out.push_back('\t');
            break;
          case 'u': {
            uint32_t cp = hex4();
            if (cp >= 0xd800 && cp <= 0xdbff) {
              require(take() == '\\' && take() == 'u', ErrorCode::header,
                      "missing unicode surrogate");
              uint32_t low = hex4();
              require(low >= 0xdc00 && low <= 0xdfff, ErrorCode::header,
                      "invalid low surrogate");
              cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
            } else
              require(!(cp >= 0xdc00 && cp <= 0xdfff), ErrorCode::header,
                      "unpaired low surrogate");
            utf8(out, cp);
            break;
          }
          default:
            throw Error(ErrorCode::header, "invalid JSON escape");
        }
      } else if (c < 128)
        out.push_back(char(c));
      else {
        size_t n = c >= 0xc2 && c <= 0xdf   ? 1
                   : c >= 0xe0 && c <= 0xef ? 2
                   : c >= 0xf0 && c <= 0xf4 ? 3
                                            : 0;
        require(n > 0, ErrorCode::header, "invalid UTF8 lead");
        uint32_t cp = c & ((1u << (6 - n)) - 1);
        for (size_t i = 0; i < n; ++i) {
          unsigned char b = static_cast<unsigned char>(take());
          require((b & 0xc0) == 0x80, ErrorCode::header,
                  "invalid UTF8 continuation");
          cp = (cp << 6) | (b & 63);
        }
        require(cp >= (n == 1   ? 0x80u
                       : n == 2 ? 0x800u
                                : 0x10000u) &&
                    cp <= 0x10ffff && !(cp >= 0xd800 && cp <= 0xdfff),
                ErrorCode::header, "noncanonical UTF8");
        utf8(out, cp);
      }
    }
  }
  Json value(size_t depth) {
    space();
    require(depth <= 16 && at_ < input_.size(), ErrorCode::header,
            "JSON depth/truncation");
    Json out;
    char c = input_[at_];
    if (c == '{') {
      ++at_;
      out.type = Json::Type::object;
      if (consume('}')) return out;
      while (true) {
        space();
        std::string key = string();
        require(consume(':'), ErrorCode::header, "missing JSON colon");
        auto item = value(depth + 1);
        require(out.fields.emplace(std::move(key), std::move(item)).second,
                ErrorCode::header, "duplicate JSON key");
        if (consume('}')) return out;
        require(consume(','), ErrorCode::header, "missing JSON comma");
      }
    }
    if (c == '"') {
      out.type = Json::Type::string;
      out.text = string();
      return out;
    }
    for (const char *word : {"true", "false", "null"}) {
      size_t n = std::strlen(word);
      if (input_.compare(at_, n, word) == 0) {
        at_ += n;
        out.type = word[0] == 'n' ? Json::Type::null : Json::Type::boolean;
        out.text = word;
        return out;
      }
    }
    // Phase77's schema has no arrays. Reject them rather than silently ignore.
    size_t begin = at_;
    if (c == '-') ++at_;
    require(at_ < input_.size(), ErrorCode::header, "truncated JSON number");
    if (input_[at_] == '0')
      ++at_;
    else {
      require(input_[at_] >= '1' && input_[at_] <= '9', ErrorCode::header,
              "invalid JSON value");
      while (at_ < input_.size() && input_[at_] >= '0' && input_[at_] <= '9')
        ++at_;
    }
    if (at_ < input_.size() && input_[at_] == '.') {
      ++at_;
      size_t start = at_;
      while (at_ < input_.size() && input_[at_] >= '0' && input_[at_] <= '9')
        ++at_;
      require(at_ > start, ErrorCode::header, "missing JSON fraction");
    }
    if (at_ < input_.size() && (input_[at_] == 'e' || input_[at_] == 'E')) {
      ++at_;
      if (at_ < input_.size() && (input_[at_] == '+' || input_[at_] == '-'))
        ++at_;
      size_t start = at_;
      while (at_ < input_.size() && input_[at_] >= '0' && input_[at_] <= '9')
        ++at_;
      require(at_ > start, ErrorCode::header, "missing JSON exponent");
    }
    out.type = Json::Type::number;
    out.text = input_.substr(begin, at_ - begin);
    (void)out.number();
    return out;
  }
  std::string input_;
  size_t at_ = 0;
};
}  // namespace detail

struct Config {
  std::string model_path, raw_vectors_path, rotated_path, trusted_header_path,
      model_sha256;
  size_t docs = 0, dimension = 768, max_neighbors = 32;
  bool cosine = true;
};
class Import {
 public:
  static constexpr size_t dimension = 768, code_bytes = 3328,
                          record_bytes = 3600, header_bytes = 4096;
  Import() = default;
  Import(const Import &) = delete;
  Import &operator=(const Import &) = delete;
  Import(Import &&rhs) noexcept {
    swap(rhs);
  }
  Import &operator=(Import &&rhs) noexcept {
    if (this != &rhs) {
      clear();
      swap(rhs);
    }
    return *this;
  }
  // Failed reload invalidates this object; it never retains a stale model.
  void open(const Config &c) {
    clear();
    Import next;
    next.open_impl(c);
    *this = std::move(next);
  }
  void clear() noexcept {
    model_.clear();
    raw_.clear();
    rotated_.clear();
    docs_ = 0;
  }
  bool loaded() const noexcept {
    return docs_ != 0;
  }
  size_t count() const noexcept {
    return docs_;
  }
  size_t mapped_bytes() const noexcept {
    return model_.size() + raw_.size() + rotated_.size();
  }
  void verify_raw_vector(uint32_t id, uint64_t key, const float *values,
                         size_t dims = dimension) const {
    identity(id, key);
    require(values && dims == dimension, ErrorCode::compatibility,
            "raw vector shape", id);
    require(std::memcmp(values, raw_.data() + 16 + size_t(id) * dimension * 4,
                        dimension * 4) == 0,
            ErrorCode::compatibility, "raw vector bytes mismatch", id);
    for (size_t k = 0; k < dimension; ++k)
      require(std::isfinite(values[k]), ErrorCode::compatibility,
              "nonfinite raw vector", id);
  }
  void verify_rotated_vector(uint32_t id, uint64_t key, const float *values,
                             size_t dims = dimension) const {
    identity(id, key);
    require(values && dims == dimension, ErrorCode::compatibility,
            "rotated vector shape", id);
    require(std::memcmp(values, rotated_.data() + size_t(id) * dimension * 4,
                        dimension * 4) == 0,
            ErrorCode::compatibility, "rotation bytes mismatch", id);
    for (size_t k = 0; k < dimension; ++k)
      require(std::isfinite(values[k]), ErrorCode::compatibility,
              "nonfinite rotated vector", id);
  }
  void verify_vectors(uint32_t id, uint64_t key, const float *raw,
                      const float *rotated, size_t dims = dimension) const {
    verify_raw_vector(id, key, raw, dims);
    verify_rotated_vector(id, key, rotated, dims);
  }
  // Validate the FINAL post-filter/augment/prune neighbor key order. Caller
  // retains original codec bytes on false (short block); true means copied.
  bool copy_full_block(uint32_t id, uint64_t key, const uint64_t *neighbors,
                       size_t degree, char *destination,
                       size_t capacity) const {
    const uint8_t *row = identity(id, key);
    require(degree <= 32 && detail::u32(row + 8) == degree,
            ErrorCode::compatibility, "final degree mismatch", id);
    require(neighbors || degree == 0, ErrorCode::compatibility,
            "null neighbor keys", id);
    for (size_t i = 0; i < degree; ++i)
      require(detail::u64(row + 16 + 8 * i) == neighbors[i],
              ErrorCode::compatibility, "final neighbor key/order mismatch",
              id);
    if (degree < 32) return false;
    require(destination && capacity >= code_bytes, ErrorCode::compatibility,
            "code destination capacity", id);
    std::memcpy(destination, row + 272, code_bytes);
    return true;
  }

 private:
  void swap(Import &rhs) noexcept {
    model_.swap(rhs.model_);
    raw_.swap(rhs.raw_);
    rotated_.swap(rhs.rotated_);
    std::swap(docs_, rhs.docs_);
  }
  const uint8_t *identity(uint32_t id, uint64_t key) const {
    require(loaded(), ErrorCode::state, "import not loaded");
    require(id < docs_, ErrorCode::compatibility, "internal ID out of range",
            id);
    const uint8_t *row =
        model_.data() + header_bytes + size_t(id) * record_bytes;
    require(detail::u64(row) == key, ErrorCode::compatibility,
            "center key mismatch", id);
    return row;
  }
  void open_impl(const Config &c) {
    require(c.docs > 0 && c.docs <= UINT32_MAX && c.dimension == dimension &&
                c.max_neighbors == 32 && c.cosine,
            ErrorCode::compatibility, "unsupported snapshot shape/metric");
    uint16_t endian = 1;
    require(*reinterpret_cast<uint8_t *>(&endian) == 1 && sizeof(float) == 4 &&
                std::numeric_limits<float>::is_iec559,
            ErrorCode::compatibility, "requires little-endian IEEE FP32");
    require(detail::hex_digest(c.model_sha256), ErrorCode::digest,
            "trusted model SHA256 required");
    model_ = detail::Mapping(c.model_path);
    detail::Mapping trusted(c.trusted_header_path);
    require(model_.size() >= header_bytes && trusted.size() == header_bytes,
            ErrorCode::header, "header length");
    const uint8_t *h = model_.data();
    require(std::memcmp(h, "QGIC77\0\0", 8) == 0 && detail::u32(h + 8) == 1,
            ErrorCode::header, "magic/version");
    uint32_t json_bytes = detail::u32(h + 12);
    uint64_t n = detail::u64(h + 16), stride = detail::u64(h + 24);
    require(json_bytes > 0 && json_bytes <= 4064 && n == c.docs &&
                stride == record_bytes,
            ErrorCode::header, "header shape");
    require(n <= (SIZE_MAX - header_bytes) / record_bytes &&
                model_.size() == header_bytes + size_t(n) * record_bytes,
            ErrorCode::header, "model length/overflow");
    require(std::memcmp(h, trusted.data(), header_bytes) == 0,
            ErrorCode::header, "frozen header mismatch");
    for (size_t i = 32 + json_bytes; i < header_bytes; ++i)
      require(h[i] == 0, ErrorCode::header, "nonzero header padding");
    const auto meta =
        detail::JsonParser(
            std::string(reinterpret_cast<const char *>(h + 32), json_bytes))
            .parse();
    auto integer = [&](const char *key, uint64_t expected) {
      require(meta.field(key).integer() == expected, ErrorCode::header,
              std::string("metadata integer mismatch: ") + key);
    };
    auto number = [&](const char *key, double expected) {
      require(meta.field(key).number() == expected, ErrorCode::header,
              std::string("metadata number mismatch: ") + key);
    };
    auto string = [&](const char *key, const char *expected) {
      require(meta.field(key).string() == expected, ErrorCode::header,
              std::string("metadata string mismatch: ") + key);
    };
    integer("database_count", n);
    integer("record_count", n);
    integer("dimension", dimension);
    integer("max_neighbors", 32);
    integer("proxies_per_full_center", 16);
    integer("codec_bytes", code_bytes);
    integer("record_bytes", record_bytes);
    integer("sweeps", 2);
    integer("flips_per_edge_per_sweep", 4);
    integer("proxy_seed", 7602);
    integer("audit_record_bytes", 64);
    number("proxy_scale", .25);
    number("alpha_cap_ratio", 1.02);
    number("minimum_delta_relative", 1e-13);
    string("objective", "independent sum(e.T@(Q.T@Q+lambda*I)@e)");
    string("lambda", "trace(Q.T@Q)/768");
    string("short_blocks", "native codec");
    string("center_and_neighbor_keys",
           "exact original graph order;unused UINT64_MAX");
    string(
        "audit_layout",
        "u32 updates,u32 "
        "flags,7xf64(lambda,initial,final,maxscale,radial,FP32radial,degree)");
    string("history_layout", "257xf64 per center;unused NaN;short allNaN");
    string("moves_layout", "256x3xu16(sweep,lane,bit);unused UINT16_MAX");
    string("compact_layout", "32x96bytes;big-endian sign bits;unused lanes0");
    require(
        meta.field("requires_independent_validation_before_loading").boolean(),
        ErrorCode::header, "validation requirement flag");
    for (const char *key : {"proxy_sha256", "rotated_sha256",
                            "algorithm_header_sha256", "builder_source_sha256",
                            "builder_binary_sha256", "rotation_binary_sha256"})
      require(detail::hex_digest(meta.field(key).string()), ErrorCode::header,
              std::string("metadata SHA256: ") + key);
    const auto &inputs = meta.field("input_sha256");
    for (const char *key :
         {"models64/full_graph.bin", "models64/full_vectors.f32",
          "models64/full_candidates.bin"})
      require(detail::hex_digest(inputs.field(key).string()), ErrorCode::header,
              std::string("input SHA256: ") + key);
    require(model_.hash() == c.model_sha256, ErrorCode::digest,
            "model SHA256 mismatch");
    raw_ = detail::Mapping(c.raw_vectors_path);
    rotated_ = detail::Mapping(c.rotated_path);
    require(n <= (SIZE_MAX - 16) / (dimension * 4) &&
                raw_.size() == 16 + size_t(n) * dimension * 4 &&
                rotated_.size() == size_t(n) * dimension * 4,
            ErrorCode::header, "reference lengths");
    require(detail::u32(raw_.data()) == 0x51475664 &&
                detail::u32(raw_.data() + 4) == 1 &&
                detail::u32(raw_.data() + 8) == n &&
                detail::u32(raw_.data() + 12) == dimension,
            ErrorCode::header, "raw vector header");
    require(raw_.hash() == inputs.field("models64/full_vectors.f32").string(),
            ErrorCode::digest, "raw vectors SHA256 mismatch");
    require(rotated_.hash() == meta.field("rotated_sha256").string(),
            ErrorCode::digest, "rotated vectors SHA256 mismatch");
    docs_ = size_t(n);
    validate_records();
  }
  void validate_records() const {
    std::unordered_set<uint64_t> keys;
    keys.reserve(docs_);
    for (size_t i = 0; i < docs_; ++i) {
      uint64_t key =
          detail::u64(model_.data() + header_bytes + i * record_bytes);
      require(key != UINT64_MAX && keys.insert(key).second, ErrorCode::record,
              "invalid/duplicate center key", i);
    }
    for (size_t i = 0; i < docs_; ++i) {
      const uint8_t *row = model_.data() + header_bytes + i * record_bytes;
      uint32_t degree = detail::u32(row + 8), flag = detail::u32(row + 12);
      uint64_t key = detail::u64(row);
      require(degree <= 32 && flag == (degree == 32 ? 1u : 0u),
              ErrorCode::record, "degree/flag mismatch", i);
      for (size_t j = 0; j < 32; ++j) {
        uint64_t v = detail::u64(row + 16 + 8 * j);
        if (j >= degree) {
          require(v == UINT64_MAX, ErrorCode::record, "neighbor padding", i);
          continue;
        }
        require(v != key && keys.count(v), ErrorCode::record,
                "invalid neighbor key", i);
        for (size_t k = 0; k < j; ++k)
          require(v != detail::u64(row + 16 + 8 * k), ErrorCode::record,
                  "duplicate neighbor key", i);
      }
      for (size_t j = 0; j < 64; ++j) {
        uint32_t bits = detail::u32(row + 272 + 3072 + 4 * j);
        float factor;
        std::memcpy(&factor, &bits, 4);
        require(std::isfinite(factor), ErrorCode::record,
                "nonfinite codec factor", i);
      }
    }
  }
  detail::Mapping model_, raw_, rotated_;
  size_t docs_ = 0;
};
}  // namespace independent78
