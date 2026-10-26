#define main graph65_program_main
#include "graph65_refine.cc"
#undef main
#include <cassert>
#include <random>
int main() {
  std::mt19937 gen(65);
  std::uniform_real_distribution<float> dis(-2, 2);
  std::array<float, D> a{}, b{};
  double max_error = 0;
  for (int t = 0; t < 1000; ++t) {
    long double exact = 0;
    for (size_t j = 0; j < D; ++j) {
      a[j] = dis(gen);
      b[j] = dis(gen);
      long double d = static_cast<long double>(a[j]) - b[j];
      exact += d * d;
    }
    max_error =
        std::max(max_error, std::abs(l2(a.data(), b.data()) - double(exact)));
    assert(std::abs(l2(a.data(), b.data()) - double(exact)) < 1e-10);
  }
  assert(l2(a.data(), a.data()) == 0);
  // More than 32 coincident candidates: undefined directions never reach
  // chord().
  Scratch s;
  s.pool.resize(40);
  s.coords.assign(40 * D, 0);
  for (size_t i = 0; i < 40; ++i) s.pool[i] = {U(i), 0};
  s.prune();
  s.fills();
  assert(s.seeds.size() == 32 && s.angular == s.nearest &&
         s.coverage(s.angular) == 0);
  // One mandatory zero direction and collinear nonzero candidates: exhausted
  // directions eventually fall back to remaining candidates deterministically.
  s.pool.resize(40);
  s.coords.assign(40 * D, 0);
  for (size_t i = 0; i < 40; ++i) {
    s.pool[i] = {U(i), double(i * i)};
    s.coords[i * D] = float(i);
  }
  s.prune();
  s.fills();
  assert(s.seeds.size() == 2 && s.angular.size() == 32 &&
         std::isfinite(s.coverage(s.angular)));
  std::printf(
      "PASS 1000 long-double distance checks, max error %.17g; zero/collinear "
      "cases\n",
      max_error);
}
