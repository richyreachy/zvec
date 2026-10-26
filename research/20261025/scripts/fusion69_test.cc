#define QG_BUCKET_WIDTH .016f
#define QG_FUSION_AUDIT 1
#include <cassert>
#include <cstdio>
#include <random>
#include <unordered_map>
#include "core/algorithm/quantized_graph/quantized_graph_fusion.h"
using zvec::core::QuantizedGraphBucket;
using zvec::core::QuantizedGraphFusion;
using zvec::core::QuantizedGraphFusionBucket;
using F = QuantizedGraphFusion;
using B = QuantizedGraphFusionBucket;
__attribute__((target("f16c"))) float hardware_decode(uint16_t h) {
  return _cvtsh_ss(h);
}
int main() {
  assert(__builtin_cpu_supports("f16c"));
  for (unsigned h = 0; h < 65536; ++h) {
    float a = F::decode(h), b = hardware_decode(h);
    assert((std::isnan(a) && std::isnan(b)) ||
           (a == b && std::signbit(a) == std::signbit(b)));
  }
  F f;
  f.reset(true);
  assert(f.observe(7, .1f, .2f) == .1f && f.version(7) == 0);
  float x = f.observe(7, .3f, .4f);
  double mean = (2. / 3) * .1 + (1. / 3) * .3;
  double expected =
      mean - double(F::kLambda) * std::sqrt(2 * mean * (.2 * .4 / .6) / 768);
  assert(std::abs(x - expected) < 1e-7 && f.version(7) == 1);
  assert(!f.valid(7, 0) && f.valid(7, 1));
  assert(f.observe(7, -.2f, .3f) == -.2f && f.version(7) == 1);
  f.reset(true);
  assert(f.version(7) == 0);
  assert(f.observe(7, -.2f, .2f) == -.2f);
  assert(f.observe(7, -.1f, .2f) == -.15f);
  f.reset(true);
  f.observe(7, .2f, 0);
  assert(f.observe(7, .4f, 0) == .4f && f.version(7) == 0);
  f.reset(true);
  f.observe(7, .2f, .1f);
  assert(f.observe(7, .2f, .3f) == .2f && f.version(7) == 0);
  f.reset(true);
  f.observe(7, .2f, .1f);
  assert(std::isnan(f.observe(7, NAN, .3f)));
  assert(f.version(7) == 0);
  // Collision overflow must not evict previous live versions or drop scores.
  f.reset(true);
  for (unsigned i = 0; i < F::kProbes; ++i)
    f.observe(7 + i * F::kSlots, .1f, .2f);
  for (unsigned i = 0; i < F::kProbes; ++i) {
    f.observe(7 + i * F::kSlots, .3f, .4f);
    assert(f.version(7 + i * F::kSlots) == 1);
  }
  auto overflow = 7 + F::kProbes * F::kSlots;
  assert(f.observe(overflow, .5f, .3f) == .5f && f.version(overflow) == 0);
  assert(f.observe(overflow, .2f, .3f) == .2f && f.version(overflow) == 0);
  for (unsigned i = 0; i < F::kProbes; ++i)
    assert(f.version(7 + i * F::kSlots) == 1);
  // Same-bin LIFO order and cross-bin stale removal before stop.
  B q;
  q.reset(1);
  q.set_scale(1);
  q.insert(3, .1f, 0);
  q.insert(4, .101f, 0);
  assert(q.pop() == 4);
  assert(q.pop() == 3);
  q.insert(7, .01f, 0);
  q.insert(7, .3f, 1);
  q.insert(8, .2f, 0);
  while (q.has_next() && !f.valid(q.next_id_at(0), q.next_version())) q.pop();
  assert(q.next_id_at(0) == 8 && q.next_distance() > .1f);
  // Disabled cache/64-bit queue exactly reproduces the frozen 32-bit bucket.
  std::mt19937 rng(6901);
  std::uniform_real_distribution<float> score(-.2f, 2.f);
  uint64_t compared = 0;
  for (int round = 0; round < 100; ++round) {
    f.reset(false);
    q.reset(1);
    q.set_scale(.2f);
    QuantizedGraphBucket orig;
    orig.reset(1);
    orig.set_scale(.2f);
    for (int j = 0; j < 2000; ++j) {
      uint32_t id = rng() % 3000;
      float d = score(rng);
      float out = f.observe(id, d, .2f);
      assert(out == d && f.version(id) == 0);
      q.insert(id, out, 0);
      orig.insert(id, d);
      if (j % 7 == 0) {
        assert(q.next_distance() == orig.next_distance());
        assert(q.pop() == orig.pop());
        ++compared;
      }
    }
    while (orig.has_next()) {
      assert(q.has_next() && q.next_distance() == orig.next_distance());
      assert(q.pop() == orig.pop());
      ++compared;
    }
    assert(!q.has_next());
  }
  // Random enabled operations compared with an independent unbounded list of
  // (bin, insertion serial, id, version); supports duplicates and invalidation.
  struct Item {
    size_t bin;
    unsigned serial;
    uint32_t id, version;
  };
  uint64_t simulated = 0;
  for (int round = 0; round < 100; ++round) {
    f.reset(true);
    q.reset(1);
    q.set_scale(.3f);
    std::vector<Item> ref;
    for (unsigned j = 0; j < 2000; ++j) {
      uint32_t id = rng() % 500;
      float d = f.observe(id, score(rng), .1f + float(rng() % 100) / 100);
      uint32_t version = f.version(id);
      q.insert(id, d, version);
      ref.push_back({q.locate(d), j, id, version});
      if (j % 5 == 0) {
        ref.erase(
            std::remove_if(ref.begin(), ref.end(),
                           [&](auto &a) { return !f.valid(a.id, a.version); }),
            ref.end());
        while (q.has_next() && !f.valid(q.next_id_at(0), q.next_version()))
          q.pop();
        assert(q.has_next() == !ref.empty());
        if (!ref.empty()) {
          auto it =
              std::min_element(ref.begin(), ref.end(), [](auto &a, auto &b) {
                return a.bin < b.bin || (a.bin == b.bin && a.serial > b.serial);
              });
          assert(q.first() == it->bin && q.next_id_at(0) == it->id &&
                 q.next_version() == it->version);
          q.pop();
          ref.erase(it);
          ++simulated;
        }
      }
    }
  }
  // Independent direct reconstruction verifies the analytic error norm.
  std::normal_distribution<float> normal;
  std::array<float, 768> a{}, b{};
  double max_relative = 0;
  for (int row = 0; row < 4096; ++row) {
    for (size_t j = 0; j < 768; ++j) {
      a[j] = normal(rng) * .03f;
      b[j] = row == 0 ? a[j] : normal(rng) * .03f;
    }
    uint16_t out;
    assert(F::error_scale(a.data(), b.data(), 768, out));
    double norm = 0, l1 = 0;
    for (size_t j = 0; j < 768; ++j) {
      double r = double(b[j]) - a[j];
      norm += r * r;
      l1 += std::abs(r);
    }
    double h = l1 ? norm / l1 : 0, expected_e2 = 0;
    for (size_t j = 0; j < 768; ++j) {
      double r = double(b[j]) - a[j];
      double e = (r > 0 ? h : -h) - r;
      expected_e2 += e * e;
    }
    assert(out == F::encode(float(expected_e2)));
    if (expected_e2)
      max_relative = std::max(
          max_relative, std::abs(F::decode(out) - expected_e2) / expected_e2);
  }
  uint16_t out;
  a.fill(0);
  b.fill(INFINITY);
  assert(!F::error_scale(a.data(), b.data(), 768, out));
  b.fill(0);
  b[0] = 1000;
  assert(!F::error_scale(a.data(), b.data(), 768, out));
  std::printf(
      "FUSION69 PASS half=65536 disabled_queue=%llu enabled_queue=%llu "
      "geometry=4096 max_half_relative=%.9g\n",
      (unsigned long long)compared, (unsigned long long)simulated,
      max_relative);
}
