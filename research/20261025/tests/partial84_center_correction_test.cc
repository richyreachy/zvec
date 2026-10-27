// Verify the packed-code adapter against an independent signed-dot formula.
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h>

#ifndef QG_PARTIAL_CENTER_CORRECTION
#error This probe exercises center-corrected partial scanning.
#endif

int main() {
  constexpr size_t d = 768, m = QG_PARTIAL_SCAN_DIM, k = 32;
  std::mt19937 rng(8401);
  std::uniform_real_distribution<float> random(-.1f, .1f);
  size_t checked = 0;
  for (bool cosine : {false, true}) {
    if (!cosine && m < d) {
      bool rejected = false;
      try {
        zvec::turbo::SymphonyQGBlockQuantizer unsupported(d, cosine);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      if (!rejected) throw std::runtime_error("partial L2 must be rejected");
      continue;
    }
    zvec::turbo::SymphonyQGBlockQuantizer codec(d, cosine);
    if (codec.batch_bytes() != 3328)
      throw std::runtime_error("code size changed");
    for (size_t count : {1U, 17U, 32U}) {
      for (size_t trial = 0; trial < 16; ++trial) {
        std::vector<float> input(d), center(d), vectors(count * d);
        for (auto &x : input) x = random(rng);
        decltype(codec)::Query query;
        codec.prepare_query(input.data(), query);
        for (size_t j = 0; j < d; ++j)
          center[j] = trial % 2 ? random(rng) : query.rotated[j];
        for (size_t i = 0; i < count; ++i)
          for (size_t j = 0; j < d; ++j)
            vectors[i * d + j] = center[j] + (trial == 0 ? 0 : random(rng));
        std::vector<char> native(3328, 0), adjusted(3329, 0);
        rabitqlib::quant::quantize_qg_batch(vectors.data(), center.data(),
                                            count, d, native.data(),
                                            rabitqlib::METRIC_L2);
        rabitqlib::QGBatchDataMap<float> original(native.data(), d);
        if (cosine)
          zvec::turbo::ConvertSymphonyQGCosineFactors(
              center.data(), vectors.data(), count, d, original.f_add(),
              original.f_rescale());
        codec.encode(center.data(), vectors.data(), count, adjusted.data() + 1);
        rabitqlib::ConstQGBatchDataMap<float> result(adjusted.data() + 1, d);
        if (std::memcmp(original.bin_code(), result.bin_code(), d * 4) != 0)
          throw std::runtime_error("native code bits changed");
        if (m == d &&
            std::memcmp(native.data(), adjusted.data() + 1, 3328) != 0)
          throw std::runtime_error("full endpoint changed");
        std::array<float, k> scores;
        constexpr float center_score = .375f;
        codec.scan(adjusted.data() + 1, query, center_score, scores.data());
        for (size_t i = 0; i < count; ++i) {
          const double b = original.f_rescale()[i];
          if (b != result.f_rescale()[i])
            throw std::runtime_error("scale changed");
          double full_center = 0, sampled_residual = 0;
          for (size_t j = 0; j < d; ++j) {
            const float residual = vectors[i * d + j] - center[j];
            const double sign = residual > 0 ? .5 : -.5;
            full_center += double(center[j]) * sign;
            if (j < m)
              sampled_residual += (double(query.rotated[j]) - center[j]) * sign;
          }
          // A + B*c.z + B*rho*(q-c)_prefix.z_prefix, not a sampled q dot
          // with an unchanged center correction.
          const double expected =
              center_score + double(original.f_add()[i]) +
              b * (full_center + double(d) / m * sampled_residual);
          const double lut_bound =
              std::abs(b) * query.lookup.delta() * (d / 4) * .501;
          const double round_bound =
              2e-4 * (1 + std::abs(expected) + std::abs(b));
          if (!std::isfinite(scores[i]) ||
              std::abs(scores[i] - expected) > lut_bound + round_bound)
            throw std::runtime_error("center residual formula mismatch");
          ++checked;
        }
      }
    }
  }
  std::cout << "PASS prefix=" << m << " lanes=" << checked
            << " native bits/scales, unchanged block size, residual formula, "
               "full endpoint\n";
}
