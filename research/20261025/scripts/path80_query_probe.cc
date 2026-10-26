#define main path80_score_program_main
#include "path80_score.cc"
#undef main
#include <zvec/core/interface/index.h>
#include "turbo/quantizer/quantizer.h"
namespace zvec::ailego {
float InnerProductFp32SSE(const float *, const float *, size_t);
float InnerProductFp32AVX(const float *, const float *, size_t);
float InnerProductFp32AVX512(const float *, const float *, size_t);
float InnerProductFp32Scalar(const float *, const float *, size_t);
}  // namespace zvec::ailego
struct QueryProbe : zvec::core_interface::HNSWIndex {
  int initialize(const zvec::core_interface::HNSWIndexParam &param) {
    return init(param);
  }
  bool uses_turbo() const {
    return bool(turbo_quantizer_);
  }
  float distance(const float *v, const void *q, size_t dim) const {
    return turbo_quantizer_ ? turbo_quantizer_->calc_distance_dp_query(v, q)
                            : metric_->distance(v, q, dim);
  }
  int prepare(const float *q, std::string &storage, const void *&prepared,
              core::IndexQueryMeta &meta) {
    return _prepare_dense_query({zvec::core_interface::DenseVector{q}},
                                &storage, &prepared, &meta);
  }
};
int main(int argc, char **argv) try {
  need(argc == 4, "usage probe events raw query_text");
  Map trace(argv[1]), raw(argv[2]);
  const auto q = queries(argv[3]);
  QueryProbe index;
  zvec::core_interface::HNSWIndexParam param(
      zvec::core_interface::MetricType::kCosine, D, 50, 500);
  param.data_type = zvec::core_interface::DataType::DT_FP32;
  param.use_id_map = true;
  need(index.initialize(param) == 0, "real index init");
  const zvec::ailego::Params empty;
  auto reformer = core::IndexFactory::CreateReformer("CosineNormalizeReformer");
  auto metric = core::IndexFactory::CreateMetric("Cosine");
  need(reformer && reformer->init(empty) == 0 && metric &&
           metric->init(core::IndexMeta(core::IndexMeta::DT_FP32, D + 1),
                        empty) == 0,
       "factory init");
  core::IndexQueryMeta prepared_meta, own_meta;
  std::string prepared, own;
  const void *real_query = nullptr;
  need(index.prepare(q[0].data(), prepared, real_query, prepared_meta) == 0,
       "actual prepare");
  need(reformer->transform(q[0].data(),
                           core::IndexQueryMeta(core::IndexMeta::DT_FP32, D),
                           &own, &own_meta) == 0,
       "own prepare");
  std::cout << "{\"prepared_dim\":" << prepared_meta.dimension()
            << ",\"prepared_bytes\":" << prepared.size()
            << ",\"own_bytes\":" << own.size()
            << ",\"prepared_equal\":" << (prepared == own ? "true" : "false")
            << ",\"uses_turbo\":" << (index.uses_turbo() ? "true" : "false")
            << "}\n";
  for (size_t i = 0; i < 40; ++i) {
    Event e = take<Event>(trace.p + 16 + i * 32);
    const auto *v =
        reinterpret_cast<const float *>(raw.p + 16) + size_t(e.target) * D;
    const auto *qp = static_cast<const float *>(real_query);
    const float m = metric->distance(v, qp, D + 1);
    const float actual =
        index.distance(v, real_query, prepared_meta.dimension());
    const float sse = 1.f - zvec::ailego::InnerProductFp32SSE(v, qp, D);
    const float avx = 1.f - zvec::ailego::InnerProductFp32AVX(v, qp, D);
    const float avx512 = 1.f - zvec::ailego::InnerProductFp32AVX512(v, qp, D);
    const float scalar = 1.f - zvec::ailego::InnerProductFp32Scalar(v, qp, D);
    std::cout << "{\"event\":" << i << ",\"target\":" << e.target
              << ",\"expected\":" << bits(e.exact)
              << ",\"actual\":" << bits(actual) << ",\"metric\":" << bits(m)
              << ",\"sse\":" << bits(sse) << ",\"avx\":" << bits(avx)
              << ",\"avx512\":" << bits(avx512)
              << ",\"scalar\":" << bits(scalar) << "}\n";
  }
  return 0;
} catch (const std::exception &e) {
  std::cerr << "PROBE FAILED " << e.what() << '\n';
  return 1;
}
