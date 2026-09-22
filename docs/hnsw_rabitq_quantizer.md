# RaBitQ in the standard HNSW index

`HNSWIndexParam` accepts `RabitqQuantizerParam(total_bits)`. The bit count is
1–8, with a default of 7, matching the existing RaBitQ index setting. This
uses `HnswStreamer` and the original-vector build capability: graph distances
come from the original FP32 provider, while the index stores packed RaBitQ
codes and evaluates queries against those codes.

```cpp
using namespace zvec::core_interface;

// Populate this provider with the original vectors, keyed by the same doc IDs
// passed to index->add(). Keep it available for every insert/update.
auto provider = std::make_shared<zvec::core::MultiPassIndexProvider<
    zvec::core::IndexMeta::DT_FP32>>(dimension);
zvec::core::IndexMeta original_meta(zvec::core::IndexMeta::DT_FP32, dimension);
original_meta.set_metric("SquaredEuclidean", 0, zvec::ailego::Params{});

auto param = HNSWIndexParamBuilder()
    .with_data_type(DataType::DT_FP32)
    .with_dimension(dimension)
    .with_metric_type(MetricType::kL2sq)
    .with_quantizer_param(RabitqQuantizerParam(7))
    .with_provider(provider, original_meta)
    .build();
auto index = IndexFactory::CreateAndInitIndex(*param);
// Check return values as usual, then open, add, and search with HNSWQueryParam.
```

The provider must contain each original vector **before** its insertion and
must expose plain FP32 vectors with matching dimensionality. Inserts without
a provider fail. Search and fetch after reopening do not require the provider;
reattach it when reopening for further inserts. Configure the same bit count,
dimension, and metric when reopening. The quantizer's rotation is saved with
the index and restored before queries or inserts are accepted. Missing or
corrupt quantizer state causes open to fail.

Supported input: dense FP32, dimensions 2–4095, L2 squared, inner product, and
cosine. Cosine normalization is performed inside the quantizer. RaBitQ always
rotates vectors; `enable_rotate` does not add a second rotation. External-vector
storage is not supported by this integration. Mmap, buffer-pool, and contiguous
in-index storage use the normal HNSW paths.

This is a zero-centered, training-free encoder using the bundled RaBitQ
library's multi-bit quantization and Turbo's portable FHT rotation. It does
not use the separate HNSW-RaBitQ index's centroid training or file format. Each
record uses `ceil(dimension * total_bits / 8) + 12` bytes; queries use rotated
FP32 values. Fetched vectors are approximate reconstructions. Distances use
a scalar asymmetric estimator; SIMD performance tuning and benchmarking are
not part of this integration. The existing dedicated HNSW-RaBitQ API remains
available.
