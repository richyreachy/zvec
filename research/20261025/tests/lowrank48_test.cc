#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h>
int main() {
#if RABITQ_SUPPORTED
  std::mt19937 rng(48823);
  std::normal_distribution<float> normal(0, .1f);
  size_t checks = 0;
  double worst = 0, worst_lut = 0;
  const char *path = std::getenv("QG_LOWRANK_BASIS");
  assert(path);
  for (size_t d : {64, 768, 960}) {
    std::vector<float> basis(QG_LOWRANK * d);
    for (size_t j = 0; j < QG_LOWRANK; ++j) {
      for (size_t z = 0; z < d; ++z) basis[j * d + z] = normal(rng);
      for (size_t k = 0; k < j; ++k) {
        double a = 0;
        for (size_t z = 0; z < d; ++z)
          a += double(basis[j * d + z]) * basis[k * d + z];
        for (size_t z = 0; z < d; ++z) basis[j * d + z] -= a * basis[k * d + z];
      }
      double norm = 0;
      for (size_t z = 0; z < d; ++z)
        norm += double(basis[j * d + z]) * basis[j * d + z];
      for (size_t z = 0; z < d; ++z) basis[j * d + z] /= std::sqrt(norm);
    }
    {
      std::ofstream f(path, std::ios::binary);
      f.write(reinterpret_cast<const char *>(basis.data()),
              basis.size() * sizeof(float));
    }
    for (bool cosine : {false, true})
      for (size_t count : {1, 7, 32}) {
        zvec::turbo::SymphonyQGBlockQuantizer codec(d, cosine);
        const size_t p = codec.encoded_dim();
        std::vector<float> center_raw(d), center, raw(count * d),
            values(count * p), tmp, rb(QG_LOWRANK * p);
        for (auto &v : center_raw) v = normal(rng);
        codec.transform(center_raw.data(), center);
        for (size_t j = 0; j < QG_LOWRANK; ++j) {
          codec.transform(basis.data() + j * d, tmp);
          std::copy(tmp.begin(), tmp.end(), rb.begin() + j * p);
        }
        for (size_t i = 0; i < count; ++i) {
          for (size_t z = 0; z < d; ++z)
            raw[i * d + z] = center_raw[z] + (i == 0 ? 0 : normal(rng));
          codec.transform(raw.data() + i * d, tmp);
          std::copy(tmp.begin(), tmp.end(), values.begin() + i * p);
        }
        std::vector<char> codes(codec.batch_bytes());
        codec.encode(center.data(), values.data(), count, codes.data());
        rabitqlib::ConstQGBatchDataMap<float> batch(codes.data(), p);
        const float *correction = reinterpret_cast<const float *>(
            codes.data() + rabitqlib::QGBatchDataMap<float>::data_bytes(p));
        for (int trial = 0; trial < 16; ++trial) {
          std::vector<float> q(d);
          if (trial < 8) {
            q = center_raw;
            for (size_t j = 0; j < QG_LOWRANK; ++j) {
              float a = normal(rng);
              for (size_t z = 0; z < d; ++z) q[z] += a * basis[j * d + z];
            }
          } else
            for (auto &v : q) v = normal(rng);
          decltype(codec)::Query query;
          codec.prepare_query(q.data(), query);
          double dc = cosine ? 1 : 0;
          for (size_t z = 0; z < d; ++z)
            dc += cosine ? -double(q[z]) * center_raw[z]
                         : std::pow(double(q[z]) - center_raw[z], 2);
          float scores[32];
          codec.scan(codes.data(), query, float(dc), scores);
          for (size_t i = 0; i < count; ++i) {
            double rr = 0, l1 = 0;
            std::vector<double> error(p);
            for (size_t z = 0; z < p; ++z) {
              double r = double(values[i * p + z]) - center[z];
              rr += r * r;
              l1 += std::abs(r);
            }
            double amp = l1 ? rr / l1 : 0;
            double ideal = batch.f_add()[i] + dc;
            for (size_t z = 0; z < p; ++z) {
              double r = double(values[i * p + z]) - center[z];
              error[z] = (r > 0 ? amp : -amp) - r;
              ideal +=
                  batch.f_rescale()[i] * query.rotated[z] * (r > 0 ? .5 : -.5);
            }
            for (size_t j = 0; j < QG_LOWRANK; ++j) {
              double pe = 0;
              for (size_t z = 0; z < p; ++z) pe += error[z] * rb[j * p + z];
              for (size_t z = 0; z < p; ++z) error[z] -= pe * rb[j * p + z];
              ideal += query.projection[j] * correction[j * 32 + i];
            }
            double truth = cosine ? 1 : 0;
            for (size_t z = 0; z < d; ++z)
              truth += cosine ? -double(q[z]) * raw[i * d + z]
                              : std::pow(double(q[z]) - raw[i * d + z], 2);
            double residual = 0;
            for (size_t z = 0; z < p; ++z)
              residual += (center[z] - query.rotated[z]) * error[z];
            residual *= cosine ? 1 : 2;
            double gap = std::abs(ideal - truth - residual);
            worst = std::max(worst, gap);
            assert(gap < 2e-5 * (1 + std::abs(truth)));
            if (trial < 8)
              assert(std::abs(ideal - truth) < 3e-5 * (1 + std::abs(truth)));
            double lut_gap = std::abs(scores[i] - ideal);
            worst_lut = std::max(worst_lut, lut_gap);
            double bound = std::abs(batch.f_rescale()[i]) *
                               query.lookup.delta() * (p / 4) * .5 +
                           2e-5 * (1 + std::abs(truth));
            assert(lut_gap <= bound);
            ++checks;
          }
        }
      }
  }
  std::cout << "low-rank identity/span/LUT checks=" << checks
            << " worst_identity=" << worst << " worst_lut=" << worst_lut
            << " passed\n";
#endif
}
