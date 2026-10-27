#include <string>
#include "independent76.h"
static thread_local std::string last_error;
extern "C" const char *error76() {
  return last_error.c_str();
}
extern "C" int costs76(size_t d, size_t p, const double *q, double lambda,
                       const double *r, const double *s, double cap,
                       double *costs, double *objective) try {
  independent76::Problem a(d, p, q, lambda);
  independent76::Edge e(a, r, s);
  e.costs(a, cap, costs);
  *objective = e.objective(a);
  return 0;
} catch (const std::exception &e) {
  last_error = e.what();
  return 1;
}
extern "C" int optimize76(size_t d, size_t p, size_t lanes, const double *q,
                          double lambda, const double *r, int8_t *signs,
                          double *alphas, double *history, uint32_t *moves,
                          uint32_t *count) try {
  independent76::Problem a(d, p, q, lambda);
  auto out = independent76::optimize(a, r, lanes);
  std::copy(out.signs.begin(), out.signs.end(), signs);
  std::copy(out.alphas.begin(), out.alphas.end(), alphas);
  std::copy(out.objectives.begin(), out.objectives.end(), history);
  std::copy(out.moves.begin(), out.moves.end(), moves);
  *count = out.moves.size() / 3;
  return 0;
} catch (const std::exception &e) {
  last_error = e.what();
  return 1;
}
