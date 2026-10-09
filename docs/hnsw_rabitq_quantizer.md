# RaBitQ in the standard HNSW index

`HNSWIndexParam` accepts `RabitqQuantizerParam(total_bits)`. The bit count is
1–9, defaulting to 7, as in the dedicated HNSW-RaBitQ index.
The same `RABITQ_SUPPORTED` platform gate applies to both integrated HNSW
and dedicated HNSW/IVF RaBitQ: only Linux x86_64 builds with the required SIMD
compiler support enable the backend. Windows, macOS, and other unsupported
platforms do not compile or register the RaBitQ quantizer and reject index creation. The RaBitQ
initialization path returns `IndexError_Unsupported`; the index factory returns
null. There is no fallback to an unquantized index.

Graph construction uses the original FP32 provider; stored vectors use centroid-based RaBitQ codes.

```cpp
using namespace zvec::core_interface;

// Populate the provider BEFORE opening a new index. Its vectors are used to
// train centroids and must use the same IDs passed to index->add().
auto provider = std::make_shared<zvec::core::MultiPassIndexProvider<
    zvec::core::IndexMeta::DT_FP32>>(dimension);
// ... provider->emplace(doc_id, original_vector) ...

zvec::core::IndexMeta original_meta(zvec::core::IndexMeta::DT_FP32, dimension);
original_meta.set_metric("SquaredEuclidean", 0, zvec::ailego::Params{});
RabitqQuantizerParam quantizer(7);
quantizer.num_clusters = 16;  // default; fewer samples use fewer centers
quantizer.sample_count = 0;   // default: all provider vectors; >0 samples them

auto param = HNSWIndexParamBuilder()
    .with_data_type(DataType::DT_FP32)
    .with_dimension(dimension)
    .with_metric_type(MetricType::kL2sq)
    .with_quantizer_param(quantizer)
    .with_provider(provider, original_meta)
    .build();
auto index = IndexFactory::CreateAndInitIndex(*param);
// Check return values, then open, add, and search with HNSWQueryParam.
```

Opening a new index trains the centroids with `OptKmeansCluster`, the trainer
used by the dedicated index. A provider with matching FP32 layout and at least
one training vector must be populated before this open. An index opened without
training data cannot accept inserts or searches. The model is frozen after
training; later inserts require their original vectors in the provider but do
not retrain the centroids.

Queries follow the dedicated index's split-estimator strategy:

1. Upper levels navigate using a 1-bit estimate and a 4-bit quantized query.
2. At level zero, compare the 1-bit distance lower estimate with the current
   worst complete distance. Only promising neighbors read the extra bits.
3. Use complete estimates for candidate and result heaps. The level-zero entry
   point is also recomputed with the complete estimate, avoiding mixed precision
   in the heap. With only 1 bit, the coarse estimate is the final estimate.

The error radius uses the RaBitQ library's factor and epsilon (1.9). This is a
statistical screening bound, not a deterministic guarantee of exact recall.
Full scans and group expansion use complete estimates. There is no automatic
original-vector reranking in this path. Construction, neighbor selection, and
pruning continue to use original FP32 distances.

Centroids and rotation are persisted with the quantizer. Reopening for search
or fetch needs no provider; reopening for insertion needs the original provider.
Reopening never retrains, even if the provider has changed. Configure the same
bit count, cluster-count setting, dimension, and metric. Missing, corrupt, or
incompatible model state fails to open. The previous single-stage RaBitQ format
is rejected; rebuild indexes created with that implementation.

Supported input: dense FP32, dimensions 2–4095, L2 squared, inner product, and
cosine. Dimensions are padded to a multiple of 64. Cosine vectors are normalized
inside the quantizer. `enable_rotate` does not add another rotation. Mmap,
buffer-pool, and contiguous in-index storage are supported; external-vector
storage is not. Fetch returns an approximate reconstruction.

The encoding uses the bundled RaBitQ library's binary and extra-bit factor
routines, with Turbo's portable FHT rotation and scalar equivalents of the
split distance estimators. Each record contains a centroid ID, reconstruction
metadata, independent binary/full-distance factors, and separately packed
1-bit/extra-bit codes: `36 + padded_dimension * total_bits / 8` bytes. Queries
keep rotated FP32 coordinates, the reconstructed 4-bit query, and per-centroid
factors. The container and packing are specific to standard HNSW; files are not
interchangeable with the dedicated HNSW-RaBitQ index. Its SIMD kernels and binary
file format are not migrated. Performance equivalence has not been benchmarked.
