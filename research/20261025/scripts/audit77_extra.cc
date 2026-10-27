// Independent phase77 artifact audit. This file neither trains nor rewrites
// models. Compile with -std=c++17 -O3 -march=native -ffp-contract=off. Endpoint
// objectives are optional; no mode recomputes every intermediate objective.
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t kDim = 768;
constexpr size_t kLanes = 32;
constexpr size_t kProxies = 16;
constexpr size_t kCompactBytes = 3072;
constexpr size_t kRecordBytes = 3600;
constexpr size_t kHistoryCount = 257;
constexpr size_t kMoveCount = 256;
void check(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
void check(bool ok, const char *message, size_t row) {
  if (!ok)
    throw std::runtime_error(std::string(message) +
                             " row=" + std::to_string(row));
}
struct Map {
  int fd = -1;
  size_t size = 0;
  const char *data = nullptr;
  explicit Map(const std::string &path) {
    fd = open(path.c_str(), O_RDONLY);
    check(fd >= 0, "open " + path);
    struct stat status {};
    check(fstat(fd, &status) == 0 && status.st_size > 0, "stat " + path);
    size = size_t(status.st_size);
    void *mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    check(mapped != MAP_FAILED, "mmap " + path);
    data = static_cast<const char *>(mapped);
  }
  Map(const Map &) = delete;
  Map &operator=(const Map &) = delete;
  ~Map() {
    if (data) munmap(const_cast<char *>(data), size);
    if (fd >= 0) close(fd);
  }
};
template <class T>
T read(const char *p) {
  T result;
  std::memcpy(&result, p, sizeof(result));
  return result;
}
bool close_number(double measured, double stored, double relative = 1e-12,
                  double absolute = 1e-14) {
  return std::isfinite(measured) && std::isfinite(stored) &&
         std::abs(measured - stored) <= absolute + relative * std::abs(stored);
}
// Independent direct residual objective, using the stored final signs only when
// requested. No candidate-delta formula or optimizer code is shared here.
double endpoint_objective(const std::vector<double> &r,
                          const std::vector<int8_t> &s,
                          const std::array<double, kLanes> &norm,
                          const std::vector<double> &q, double lambda,
                          bool initial) {
  std::array<double, kDim> error;
  double total = 0;
  for (size_t lane = 0; lane < kLanes; ++lane) {
    double u = 0;
    for (size_t k = 0; k < kDim; ++k) {
      const size_t index = lane * kDim + k;
      const double sign = initial ? (r[index] > 0 ? 1. : -1.) : s[index];
      u += r[index] * sign;
    }
    check(norm[lane] == 0 || u > 0, "endpoint denominator");
    const double alpha = norm[lane] == 0 ? 0 : norm[lane] / u;
    double euclidean = 0;
    for (size_t k = 0; k < kDim; ++k) {
      const size_t index = lane * kDim + k;
      const double sign = initial ? (r[index] > 0 ? 1. : -1.) : s[index];
      error[k] = alpha * sign - r[index];
      euclidean += error[k] * error[k];
    }
    double projected = 0;
    for (size_t j = 0; j < kProxies; ++j) {
      double value = 0;
      for (size_t k = 0; k < kDim; ++k) value += q[j * kDim + k] * error[k];
      projected += value * value;
    }
    total += projected + lambda * euclidean;
  }
  return total;
}
}  // namespace

int main(int argc, char **argv) try {
  check(argc == 7 || argc == 8,
        "usage: audit77_extra graph rotated proxies model_dir model_filename "
        "none|all|sample [objective_ids.u32]");
  const std::string mode = argv[6];
  check((mode == "none" || mode == "all") ? argc == 7
                                          : (mode == "sample" && argc == 8),
        "objective mode/arguments");
  const uint16_t endian = 1;
  check(*reinterpret_cast<const uint8_t *>(&endian) == 1,
        "little-endian host required");
  Map graph(argv[1]), rotated(argv[2]), proxies(argv[3]);
  const std::string directory = argv[4];
  Map model(directory + "/" + argv[5]), compact(directory + "/compact.u8"),
      audit(directory + "/audit.bin"), moves(directory + "/moves.u16"),
      history(directory + "/history.f64");
  check(graph.size >= 32 && read<uint32_t>(graph.data) == 0x51474764 &&
            read<uint32_t>(graph.data + 4) == 1 &&
            read<uint32_t>(graph.data + 12) == kDim &&
            read<uint32_t>(graph.data + 16) == kLanes,
        "graph header");
  const size_t database_count = read<uint32_t>(graph.data + 8);
  check(model.size >= 4096 && std::memcmp(model.data, "QGIC77\0\0", 8) == 0 &&
            read<uint32_t>(model.data + 8) == 1 &&
            read<uint64_t>(model.data + 24) == kRecordBytes,
        "model fixed header");
  const size_t count = read<uint64_t>(model.data + 16);
  check(count > 0 && count <= database_count, "record count");
  check(graph.size == 32 + database_count * 140 &&
            rotated.size == database_count * kDim * sizeof(float) &&
            proxies.size == database_count * kProxies * sizeof(uint32_t),
        "input sizes");
  check(model.size == 4096 + count * kRecordBytes &&
            compact.size == count * kCompactBytes && audit.size == count * 64 &&
            moves.size == count * kMoveCount * 3 * sizeof(uint16_t) &&
            history.size == count * kHistoryCount * sizeof(double),
        "output sizes");
  std::vector<uint8_t> endpoint_selected(count, mode == "all" ? 1 : 0);
  size_t expected_endpoint_checks = 0;
  if (mode == "sample") {
    Map ids(argv[7]);
    check(ids.size % 4 == 0, "objective ID length");
    for (size_t i = 0; i < ids.size / 4; ++i) {
      const size_t id = read<uint32_t>(ids.data + i * 4);
      check(id < count && endpoint_selected[id] == 0,
            "unique objective IDs in range");
      check(read<uint32_t>(graph.data + 32 + id * 140 + 8) == kLanes,
            "objective ID must have full degree");
      endpoint_selected[id] = 1;
      ++expected_endpoint_checks;
    }
  }
  const float *vectors = reinterpret_cast<const float *>(rotated.data);
  std::vector<double> residual(kLanes * kDim), query(kProxies * kDim);
  std::vector<int8_t> signs(kLanes * kDim);
  uint64_t accepted_total = 0;
  size_t learned = 0, native = 0, endpoint_checks = 0;
  double largest_scale = 0, largest_radial = 0, largest_fp32_radial = 0,
         largest_endpoint_difference = 0, largest_audit_difference = 0;
  for (size_t row = 0; row < count; ++row) {
    const char *g = graph.data + 32 + row * 140;
    const char *record = model.data + 4096 + row * kRecordBytes;
    const char *saved_audit = audit.data + row * 64;
    const char *saved_moves = moves.data + row * kMoveCount * 6;
    const char *saved_history = history.data + row * kHistoryCount * 8;
    const size_t degree = read<uint32_t>(g + 8);
    const uint32_t flags = degree == kLanes ? 1 : 0;
    const uint32_t updates = read<uint32_t>(saved_audit);
    check(degree <= kLanes && updates <= kMoveCount, "degree/update bound",
          row);
    check(read<uint64_t>(record) == read<uint64_t>(g) &&
              read<uint32_t>(record + 8) == degree &&
              read<uint32_t>(record + 12) == flags &&
              read<uint32_t>(saved_audit + 4) == flags,
          "center/degree/flags", row);
    std::array<double, 7> values;
    for (size_t j = 0; j < 7; ++j) {
      values[j] = read<double>(saved_audit + 8 + j * 8);
      check(std::isfinite(values[j]) && values[j] >= 0, "finite audit", row);
    }
    check(values[6] == double(degree), "audit degree", row);
    if (flags)
      ++learned;
    else {
      ++native;
      check(updates == 0 && values[0] == 0 && values[1] == 0 && values[2] == 0,
            "short block training fields", row);
      for (size_t j = 0; j < kProxies; ++j)
        check(read<uint32_t>(proxies.data + (row * kProxies + j) * 4) ==
                  UINT32_MAX,
              "short proxy sentinel", row);
    }
    std::array<uint8_t, kCompactBytes> rebuilt{};
    std::array<double, kLanes> norm{}, l1{}, denominator{};
    const float *center = vectors + row * kDim;
    for (size_t lane = 0; lane < kLanes; ++lane) {
      if (lane >= degree) {
        check(read<uint64_t>(record + 16 + lane * 8) == UINT64_MAX,
              "unused neighbor key", row);
        check(read<float>(record + 272 + 3072 + lane * 4) == 0 &&
                  read<float>(record + 272 + 3200 + lane * 4) == 0,
              "unused factors are zero", row);
        continue;
      }
      const size_t id = read<uint32_t>(g + 12 + lane * 4);
      check(id < database_count && id != row, "neighbor ID", row);
      for (size_t previous = 0; previous < lane; ++previous)
        check(id != read<uint32_t>(g + 12 + previous * 4), "unique neighbors",
              row);
      check(read<uint64_t>(record + 16 + lane * 8) ==
                read<uint64_t>(graph.data + 32 + id * 140),
            "neighbor order/key", row);
      const float *neighbor = vectors + id * kDim;
      for (size_t k = 0; k < kDim; ++k) {
        const size_t index = lane * kDim + k;
        const double r = double(neighbor[k]) - center[k];
        check(std::isfinite(r), "finite residual", row);
        residual[index] = r;
        signs[index] = r > 0 ? 1 : -1;
        norm[lane] += r * r;
        l1[lane] += std::abs(r);
        if (r > 0) rebuilt[index / 8] |= uint8_t(1u << (7 - k % 8));
      }
      denominator[lane] = l1[lane];
    }
    std::array<uint32_t, 64> budget{};
    uint32_t last_label = 0;
    for (size_t j = 0; j < kMoveCount; ++j) {
      const uint32_t sweep = read<uint16_t>(saved_moves + j * 6);
      const uint32_t lane = read<uint16_t>(saved_moves + j * 6 + 2);
      const uint32_t bit = read<uint16_t>(saved_moves + j * 6 + 4);
      if (j >= updates) {
        check(sweep == UINT16_MAX && lane == UINT16_MAX && bit == UINT16_MAX,
              "move padding", row);
        continue;
      }
      check(flags == 1 && sweep < 2 && lane < degree && bit < kDim,
            "valid accepted move", row);
      const uint32_t order = sweep * kLanes + lane;
      check(j == 0 || order >= last_label, "sweep/lane move order", row);
      last_label = order;
      check(++budget[order] <= 4, "per-sweep/lane update budget", row);
      check(norm[lane] > 0, "zero residual cannot accept a flip", row);
      const size_t index = lane * kDim + bit;
      denominator[lane] -= 2 * signs[index] * residual[index];
      signs[index] *= -1;
      rebuilt[index / 8] ^= uint8_t(1u << (7 - bit % 8));
      if (norm[lane] > 0) {
        check(denominator[lane] > 0 &&
                  l1[lane] / denominator[lane] <= 1.02 + 1e-12,
              "accepted move radial feasibility", row);
      } else
        check(denominator[lane] == 0, "zero residual denominator", row);
    }
    check(std::memcmp(rebuilt.data(), compact.data + row * kCompactBytes,
                      kCompactBytes) == 0,
          "all moves reconstruct compact", row);
    accepted_total += updates;
    for (size_t j = 0; j < kHistoryCount; ++j) {
      const double value = read<double>(saved_history + j * 8);
      if (!flags || j > updates)
        check(std::isnan(value), "history padding", row);
      else {
        check(std::isfinite(value) && value >= 0, "history finite", row);
        if (j > 0)
          check(value < read<double>(saved_history + (j - 1) * 8),
                "recorded objective monotonicity", row);
      }
    }
    if (flags) {
      check(read<double>(saved_history) == values[1] &&
                read<double>(saved_history + updates * 8) == values[2],
            "recorded endpoint objective", row);
    }
    double max_scale = 0, max_radial = 0, max_fp32_radial = 0;
    for (size_t lane = 0; lane < degree; ++lane) {
      double u = 0;
      for (size_t k = 0; k < kDim; ++k)
        u += residual[lane * kDim + k] * signs[lane * kDim + k];
      check(norm[lane] == 0 || u > 0, "final denominator", row);
      const double alpha = norm[lane] == 0 ? 0 : norm[lane] / u;
      const double ratio = norm[lane] == 0 ? 1 : l1[lane] / u;
      const float add = read<float>(record + 272 + 3072 + lane * 4);
      const float scale = read<float>(record + 272 + 3200 + lane * 4);
      check(std::isfinite(add) && std::isfinite(scale), "finite stored factors",
            row);
      max_scale = std::max(max_scale, ratio);
      max_radial = std::max(max_radial, std::abs(alpha * u - norm[lane]));
      max_fp32_radial =
          std::max(max_fp32_radial, std::abs(-.5 * scale * u - norm[lane]));
    }
    check(max_scale <= 1.02 + 1e-12 && max_radial < 1e-12 &&
              max_fp32_radial < 1e-5,
          "final radial bounds", row);
    const std::array<double, 3> recomputed = {max_scale, max_radial,
                                              max_fp32_radial};
    for (size_t j = 0; j < 3; ++j) {
      check(close_number(recomputed[j], values[j + 3]),
            "recomputed geometry audit", row);
      largest_audit_difference = std::max(
          largest_audit_difference, std::abs(recomputed[j] - values[j + 3]));
    }
    largest_scale = std::max(largest_scale, max_scale);
    largest_radial = std::max(largest_radial, max_radial);
    largest_fp32_radial = std::max(largest_fp32_radial, max_fp32_radial);
    if (flags && endpoint_selected[row]) {
      double lambda = 0;
      for (size_t j = 0; j < kProxies; ++j) {
        const size_t id =
            read<uint32_t>(proxies.data + (row * kProxies + j) * 4);
        check(id < database_count && id != row, "objective proxy ID", row);
        for (size_t previous = 0; previous < j; ++previous)
          check(id != read<uint32_t>(proxies.data +
                                     (row * kProxies + previous) * 4),
                "unique objective proxies", row);
        for (size_t lane = 0; lane < degree; ++lane)
          check(id != read<uint32_t>(g + 12 + lane * 4),
                "proxy excludes neighbors", row);
        for (size_t k = 0; k < kDim; ++k) {
          const double value = (double(vectors[id * kDim + k]) - center[k]) / 4;
          check(std::isfinite(value), "finite objective query", row);
          query[j * kDim + k] = value;
          lambda += value * value;
        }
      }
      lambda /= kDim;
      check(close_number(lambda, values[0], 1e-12, 1e-15),
            "lambda recomputation", row);
      const double initial =
          endpoint_objective(residual, signs, norm, query, lambda, true);
      const double final =
          endpoint_objective(residual, signs, norm, query, lambda, false);
      check(close_number(initial, values[1], 1e-11, 1e-12) &&
                close_number(final, values[2], 1e-11, 1e-12),
            "direct endpoint objectives", row);
      largest_endpoint_difference =
          std::max({largest_endpoint_difference, std::abs(initial - values[1]),
                    std::abs(final - values[2])});
      ++endpoint_checks;
    }
    if ((row + 1) % 32768 == 0)
      std::fprintf(stderr, "EXTRA77 checked=%zu/%zu mode=%s\n", row + 1, count,
                   mode.c_str());
  }
  if (mode == "sample")
    check(endpoint_checks == expected_endpoint_checks,
          "all objective sample IDs checked");
  if (mode == "all")
    check(endpoint_checks == learned, "all full-degree endpoints checked");
  std::printf(
      "{\"centers\":%zu,\"learned_blocks\":%zu,\"native_short_blocks\":%zu,"
      "\"all_moves_reconstruct_compact\":true,\"accepted_moves_replayed\":%llu,"
      "\"all_intermediate_radial_constraints_checked\":true,"
      "\"all_final_radial_geometry_recomputed\":true,"
      "\"recorded_history_monotonicity_checked\":true,"
      "\"all_intermediate_objectives_recomputed\":false,"
      "\"objective_mode\":\"%s\",\"direct_endpoint_centers\":%zu,"
      "\"max_scale_ratio\":%.17g,\"max_radial_error\":%.17g,"
      "\"max_fp32_radial_error\":%.17g,\"max_geometry_audit_difference\":%.17g,"
      "\"max_endpoint_objective_difference\":%.17g,\"no_query_performance_"
      "claim\":true}\n",
      count, learned, native, static_cast<unsigned long long>(accepted_total),
      mode.c_str(), endpoint_checks, largest_scale, largest_radial,
      largest_fp32_radial, largest_audit_difference,
      largest_endpoint_difference);
  return 0;
} catch (const std::exception &error) {
  std::fprintf(stderr, "ERROR %s\n", error.what());
  return 1;
}
