// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "diskann_rabitq_quantizer.h"

#if RABITQ_SUPPORTED
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <rabitqlib/index/estimator.hpp>
#include <rabitqlib/utils/rotator.hpp>
#include <zvec/ailego/utility/float_helper.h>

namespace zvec::core {
namespace {
class DiskAnnRabitqQuantizer : public turbo::Quantizer {
 public:
  DiskAnnRabitqQuantizer() : Quantizer(turbo::QuantizeType::kRabit) {}

  int init(const IndexMeta &meta, const ailego::Params &params) override {
    if (meta.dimension() == 0 || meta.dimension() > 4095 ||
        (meta.data_type() != IndexMeta::DT_FP32 &&
         meta.data_type() != IndexMeta::DT_FP16) ||
        meta.metric_name() != "SquaredEuclidean") {
      return turbo::kErrUnsupported;
    }
    uint32_t bits = 1;
    if (params.has("total_bits") && !params.get("total_bits", &bits)) {
      return turbo::kErrInvalidArgument;
    }
    if (bits < 1 || bits > 9) return turbo::kErrInvalidArgument;
    meta_ = meta;
    ex_bits_ = bits - 1;
    padded_dim_ = ((meta.dimension() + 63) / 64) * 64;
    rotator_.reset(rabitqlib::choose_rotator<float>(
        std::max(64u, meta.dimension()), rabitqlib::RotatorType::FhtKacRotator,
        padded_dim_));
    mean_.assign(meta.dimension(), 0);
    zero_.assign(padded_dim_, 0);
    bin_bytes_ = rabitqlib::BinDataMap<float>::data_bytes(padded_dim_);
    ex_bytes_ = ex_bits_ ? rabitqlib::ExDataMap<float>::data_bytes(padded_dim_,
                                                                   ex_bits_)
                         : 0;
    return 0;
  }
  const IndexMeta &meta() const override {
    return meta_;
  }
  turbo::DataType input_data_type() const override {
    return meta_.data_type() == IndexMeta::DT_FP16 ? turbo::DataType::kFp16
                                                   : turbo::DataType::kFp32;
  }
  int dim() const override {
    return meta_.dimension();
  }
  bool require_train() const override {
    return true;
  }
  int train(IndexHolder::Pointer holder) override {
    if (!holder || !holder->count() || !holder->is_matched(meta_)) {
      return turbo::kErrInvalidArgument;
    }
    auto it = holder->create_iterator();
    if (!it) return turbo::kErrInvalidArgument;
    std::vector<double> sum(dim(), 0);
    std::vector<float> row(dim());
    size_t count = 0;
    for (; it->is_valid(); it->next()) {
      read(it->data(), row.data());
      for (int j = 0; j < dim(); ++j) {
        if (!std::isfinite(row[j])) return turbo::kErrInvalidArgument;
        sum[j] += row[j];
      }
      ++count;
    }
    if (!count) return turbo::kErrInvalidArgument;
    for (int j = 0; j < dim(); ++j) mean_[j] = sum[j] / count;
    return 0;
  }
  size_t quantized_datapoint_vector_length() const override {
    return bin_bytes_ + ex_bytes_;
  }
  size_t quantized_query_vector_length() const override {
    return padded_dim_ * sizeof(float);
  }
  void quantize_query(const void *input, void *output) const override {
    auto row = rotate(input);
    std::memcpy(output, row.data(), row.size() * sizeof(float));
  }
  void quantize_data(const void *input, void *output) const override {
    auto row = rotate(input);
    auto *out = static_cast<char *>(output);
    // rabitqlib handles a zero residual explicitly (including constant data).
    rabitqlib::quant::quantize_split_single(
        row.data(), zero_.data(), padded_dim_, ex_bits_, out, out + bin_bytes_,
        rabitqlib::METRIC_L2);
  }
  turbo::DistanceImpl distance(const void *query,
                               const IndexQueryMeta &) const override {
    // SplitSingleQuery borrows its rotated input. Share both objects with the
    // returned callable so context reuse and concurrent queries are
    // independent.
    auto row = std::make_shared<std::vector<float>>(padded_dim_);
    std::memcpy(row->data(), query, quantized_query_vector_length());
    auto q = std::make_shared<rabitqlib::SplitSingleQuery<float>>(
        row->data(), padded_dim_, ex_bits_, rabitqlib::quant::RabitqConfig(),
        rabitqlib::METRIC_L2);
    float norm2 = 0;
    for (float v : *row) norm2 += v * v;
    const float norm = std::sqrt(norm2);
    auto bits = ex_bits_;
    auto padded = padded_dim_;
    auto bin_bytes = bin_bytes_;
    auto ip_func = rabitqlib::select_excode_ipfunc(bits);
    return turbo::DistanceImpl(
        [row, q, bits, padded, bin_bytes, ip_func, norm2, norm](
            const void *dp, const void *, size_t, float *out) {
          (void)row;  // Retain the input borrowed by SplitSingleQuery.
          const auto *code = static_cast<const char *>(dp);
          float ip = 0, low = 0;
          if (bits) {
            rabitqlib::split_single_fulldist(code, code + bin_bytes, ip_func,
                                             *q, padded, bits, *out, low, ip,
                                             norm2, norm);
          } else {
            rabitqlib::split_single_estdist(code, *q, padded, ip, *out, low,
                                            norm2, norm);
          }
        },
        std::string(), padded_dim_);
  }
  float calc_distance_dp_query(const void *dp, const void *q) const override {
    return distance(q, IndexQueryMeta())(dp);
  }
  void calc_distance_dp_query_batch(const void *const *dp, int n, const void *q,
                                    float *out) const override {
    auto d = distance(q, IndexQueryMeta());
    for (int i = 0; i < n; ++i) out[i] = d(dp[i]);
  }
  float calc_distance_dp_query_unquantized(const void *dp,
                                           const void *q) const override {
    auto row = rotate(q);
    return calc_distance_dp_query(dp, row.data());
  }
  void calc_distance_dp_query_batch_unquantized(const void *const *dp, int n,
                                                const void *q,
                                                float *out) const override {
    auto row = rotate(q);
    calc_distance_dp_query_batch(dp, n, row.data(), out);
  }
  float calc_distance_dp_dp(const void *, const void *) const override {
    // DiskANN builds and reranks with full vectors; symmetric code distances
    // are deliberately unsupported by this navigation-only quantizer.
    return std::numeric_limits<float>::quiet_NaN();
  }
  int serialize(std::string *out) const override {
    if (!out || !rotator_) return turbo::kErrInvalidArgument;
    uint32_t bits = ex_bits_ + 1;
    turbo::QuantizerSerHeader hdr{};
    hdr.magic = turbo::kQuantizerMagic;
    hdr.version = turbo::kQuantizerSerVersion;
    hdr.quant_type = static_cast<uint16_t>(type());
    hdr.dim = dim();
    hdr.metric = static_cast<uint32_t>(turbo::MetricType::kSquaredEuclidean);
    hdr.data_type = static_cast<uint16_t>(input_data_type());
    hdr.payload_size =
        sizeof(bits) + mean_.size() * sizeof(float) + rotator_->dump_bytes();
    out->resize(sizeof(hdr) + hdr.payload_size);
    char *p = out->data();
    std::memcpy(p, &hdr, sizeof(hdr));
    p += sizeof(hdr);
    std::memcpy(p, &bits, sizeof(bits));
    p += sizeof(bits);
    std::memcpy(p, mean_.data(), mean_.size() * sizeof(float));
    rotator_->save(p + mean_.size() * sizeof(float));
    return 0;
  }
  int deserialize(std::string &in) override {
    return deserialize(in.data(), in.size());
  }
  int deserialize(const void *data, size_t len) override {
    if (!data || !rotator_ ||
        len < sizeof(turbo::QuantizerSerHeader) + sizeof(uint32_t))
      return turbo::kErrInvalidArgument;
    turbo::QuantizerSerHeader hdr;
    std::memcpy(&hdr, data, sizeof(hdr));
    const auto *p = static_cast<const char *>(data) + sizeof(hdr);
    uint32_t bits;
    std::memcpy(&bits, p, sizeof(bits));
    if (hdr.magic != turbo::kQuantizerMagic ||
        hdr.version != turbo::kQuantizerSerVersion ||
        hdr.quant_type != static_cast<uint16_t>(type()) ||
        hdr.dim != mean_.size() ||
        hdr.metric !=
            static_cast<uint32_t>(turbo::MetricType::kSquaredEuclidean) ||
        hdr.data_type != static_cast<uint16_t>(input_data_type()) ||
        hdr.reserved != 0 || bits < 1 || bits > 9 ||
        hdr.payload_size != sizeof(bits) + mean_.size() * sizeof(float) +
                                rotator_->dump_bytes() ||
        len != sizeof(hdr) + hdr.payload_size)
      return turbo::kErrInvalidArgument;
    ailego::Params params;
    params.set("total_bits", bits);
    auto meta = meta_;
    int ret = init(meta, params);
    if (ret) return ret;
    p += sizeof(bits);
    std::memcpy(mean_.data(), p, mean_.size() * sizeof(float));
    for (float v : mean_)
      if (!std::isfinite(v)) return turbo::kErrInvalidArgument;
    rotator_->load(p + mean_.size() * sizeof(float));
    return 0;
  }

 private:
  void read(const void *in, float *out) const {
    if (meta_.data_type() == IndexMeta::DT_FP16) {
      auto *p = static_cast<const ailego::Float16 *>(in);
      for (int i = 0; i < dim(); ++i) out[i] = static_cast<float>(p[i]);
    } else {
      std::memcpy(out, in, dim() * sizeof(float));
    }
  }
  std::vector<float> rotate(const void *in) const {
    std::vector<float> row(std::max(64, dim()), 0), out(padded_dim_);
    read(in, row.data());
    for (int j = 0; j < dim(); ++j) row[j] -= mean_[j];
    rotator_->rotate(row.data(), out.data());
    return out;
  }
  IndexMeta meta_;
  size_t padded_dim_{0}, ex_bits_{0}, bin_bytes_{0}, ex_bytes_{0};
  std::vector<float> mean_, zero_;
  std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
};
}  // namespace

turbo::Quantizer::Pointer CreateDiskAnnRabitqQuantizer() {
  return std::make_shared<DiskAnnRabitqQuantizer>();
}
}  // namespace zvec::core
#else
namespace zvec::core {
turbo::Quantizer::Pointer CreateDiskAnnRabitqQuantizer() {
  return nullptr;
}
}  // namespace zvec::core
#endif
