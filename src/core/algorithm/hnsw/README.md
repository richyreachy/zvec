# Quantized graph scanning on ordinary HNSW

Enable this experimental mode with `HnswIndexParam(metric_type=MetricType.L2,
symphony_qg=True)` in Python, `HnswIndexParams::set_symphony_qg(true)` in the
database C++ API, or `HNSWIndexParamBuilder::with_symphony_qg(true)` in the core
API. The low-level streamer parameter is `proxima.hnsw.symphony_qg`.
It defaults to false. The index type and persistent graph format remain HNSW.

## Codec selection

```python
HnswIndexParam(metric_type=MetricType.L2, quantized_graph_codec="sq8")
HnswIndexParam(metric_type=MetricType.L2, quantized_graph_codec="pq")
HnswIndexParam(metric_type=MetricType.COSINE, quantized_graph_codec="pq_residual")
HnswIndexParam(metric_type=MetricType.COSINE, quantized_graph_codec="symphony_qg")
```

`quantized_graph_codec` defaults to the empty string (disabled). The DB C++
setter is `set_quantized_graph_codec()`, the core builder method is
`with_quantized_graph_codec()`, and the streamer key is
`proxima.hnsw.quantized_graph.codec`. The selection is preserved by schema
cloning, manifests, core parameter JSON, and Python pickling. Old manifests
without this field retain their behavior. `symphony_qg=True` still selects
SymphonyQG; combining it with `sq8`, `pq`, or `pq_residual` is rejected, as are unknown codec names.
The low-level `proxima.hnsw.quantized_graph.degree` overrides the legacy
`proxima.hnsw.symphony_qg.degree`; both default to 32.

`HnswQuantizedGraph` selects a templated implementation once at construction.
Virtual dispatch occurs at the search/prebuild boundary, never per neighbor.
`HnswSymphonyQG` remains a source-compatible alias with SymphonyQG as its default.

The portable `SQ8GraphCodec` uses symmetric signed 8-bit values with one FP32
scale per neighbor (`max(abs(x)) / 127`), in batches of 16. It calibrates each
vector independently and needs no global training or RaBitQ build. Each full
batch stores `16 * (dimension + 4)` bytes of codes and scales, in addition to
neighbor IDs, cached FP32 centers, headers, and padding. Zero vectors stay zero.
For cosine, the existing vector pipeline supplies normalized coordinates;
original-norm metadata is retained for exact scoring/fetch but never quantized.
SQ8 estimates steer the beam; final scores still come from FP32 distances.
This scalar implementation validates the extension interface; no SIMD speedup
or performance parity with SymphonyQG is claimed.

`PQGraphCodec` (`quantized_graph_codec="pq"`) uses the existing Turbo
`PqFastQuantizer`: 4-bit product quantization, 16 centroids per subspace, eight
coordinates per subspace, and 32-vector FastScan blocks. Coordinates are zero
padded to a multiple of eight, so dimensions 1..4096 are supported. A full
neighbor block occupies `32 * ceil(ceil(dimension / 8) / 2)` code bytes,
excluding IDs, FP32 centers, headers, padding, and the shared codebook.

Opening/prebuilding trains on up to 4096 live vectors selected by reservoir
sampling. Corpora smaller than 16 repeat samples; an empty corpus defers
training until the first search after insertion. Codebooks and codes are
derived in-memory caches, rebuilt from FP32 storage on reopen, with an added
training cost. Insertions invalidate blocks but retain the codebook; explicit
prebuild/reopen retrains it. Sampling has a fixed seed; KMeans initialization
may still produce different approximate traversal orders across rebuilds.
Concurrent first queries synchronize model publication, then share an immutable
model with separate query LUTs. Training memory is bounded by the sample cap.

For cosine, the existing HNSW pipeline supplies normalized coordinates. PQ
uses an inner-product LUT and returns `1 - dot(query, reconstructed_neighbor)`;
it does not assume reconstructed centroids have unit norm. Zero queries retain
distance 1. Final results still use exact FP32 scores, so an external refiner
is unnecessary for the same vectors and metric. PQ compresses neighbor scan
codes, not the stored FP32 vectors. No speed or recall improvement is claimed
without corpus-specific benchmarks.

`PQResidualGraphCodec` (`quantized_graph_codec="pq_residual"`) is an opt-in
node-centered PQ4 variant. It does not change `pq`. Training samples up to 4096
live directed graph edges with a fixed-seed reservoir, loads both endpoints,
and trains one shared codebook on `v - u`. Invalid/deleted endpoints are skipped;
read failures and non-finite sampled residuals fail training. Small samples are
repeated to reach 16 vectors. An isolated live corpus trains a zero-residual
codebook so exact search still works; a completely empty corpus defers training.
The model/cache lifetime is the same as `pq`, including retaining the codebook
across insertions and retraining on explicit prebuild/reopen. Edge enumeration
cost is O(E); the number of loaded training pairs is bounded by the sample cap.

All residual arithmetic lives in Turbo's
[`PQResidualBlockQuantizer`](../../../turbo/quantizer/pq_residual_block_quantizer/pq_residual_block_quantizer.h).
It uses an InnerProduct-search PQ4 model for both metrics (encoding and training
still minimize L2 reconstruction error). Residuals are not normalized, and
centering/OPQ are disabled. One query LUT of `-dot(q, residual_centroid)` is
reused across all expanded nodes. For reconstructed residual `r_hat`:

- L2: `d(q,u) + (||r_hat||^2 + 2*dot(u,r_hat)) - 2*dot(q,r_hat)`.
  The center distance is exact. The middle term is computed at encoding time,
  accumulated in double and stored as one FP32 correction per lane. Block layout
  is `[existing packed PQ4 codes][32 FP32 corrections]`: 128 extra bytes per
  full block, including a padded partial block. The final addition uses double
  before returning FP32; large offsets can still magnify LUT/correction error.
- Cosine: `d_cos(q,u) - dot(q,r_hat)`, on already normalized input coordinates.
  Zero vectors follow `1 - dot` semantics. Blocks use the existing PQ4 packed
  layout without a correction tail. No reconstruction renormalization is done.

Approximate scores guide traversal; results retain exact FP32 scores. Negative
approximate distances are not clamped. The reconstruction helper on
`PqFastQuantizer` is restricted to trained, plain FP32 L2/IP codebooks and does
not change its generic dequantization or SDC support. Residual codebooks are
trained randomly as before; neither identical rebuilt codes nor better recall
or QPS on every dataset is guaranteed.

The opt-in `pq_graph_benchmark` executable compares both codecs on the same
fixed synthetic graph and exact top-10 ground truth. It reports independent
training trials, recall/ef curves, latency percentiles, training/prebuild cost,
and code payload/arena-size estimates. See the checked-in benchmark results
under `tests/core/algorithm/hnsw/benchmarks/` for the measured environment and
limitations; these are graph-engine measurements, not end-to-end DB throughput.

All four codecs currently require inline FP32 L2/cosine vectors with 1..4096
coordinates and no main-vector compression. Only SymphonyQG requires the
RaBitQ build and supported x86 SIMD. Codes remain a derived in-memory cache,
rebuilt on open and invalidated by mutation; codec payloads are not serialized.

To add a fixed-batch codec, implement the contract below and register its model
in `HnswQuantizedGraph::supported()` and its constructor, then extend public
schema validation. Search and caching do not need to change. The PQ adapter manages training before publishing its graph. Model
persistence, variable-length codes, and shared global code storage are not
part of the generic codec interface.

## Quantized graph framework

The reusable engine is `QuantizedGraph<Codec>` in
[`../quantized_graph/quantized_graph.h`](../quantized_graph/quantized_graph.h).
It owns neighbor blocks, degree pruning, centroid entry selection, eager/lazy
caching, quantized search policies, and result completion. It has no dependency on HNSW,
RaBitQ, or a particular storage implementation. Its current contract is dense
FP32 coordinates and fixed-size encoded neighbor batches. The codec and query
adapter must agree on the score metric. Centroid selection and diversity
pruning use squared L2 in coordinate space; cosine uses normalized coordinates.
Variable-length codes are outside this contract.

There are three compile-time extension points:

- **Graph adapter:** exposes `valid(id)`, `neighbors(id)`, and
  `get_vector(id, Vector&)`. IDs are uint32 values in `[0, doc_cnt)`, with
  `UINT32_MAX` reserved. `Vector::data()` supplies FP32 data and the vector
  handle keeps it alive; a returned neighbor view likewise owns/pins its data.
  The adapter must support concurrent reads during prebuild and search.
- **Codec:** supplies `kBatchSize`, `encoded_dim()`, `batch_bytes()`,
  `transform()`, `encode()`, and per-query `Query`, `prepare_query()`, `scan()`.
  Transform produces `encoded_dim()` floats; encode consumes a center and up
  to one batch of transformed vectors. Scan writes `kBatchSize` distance
  estimates; the engine ignores padded lanes. The codec must be immutable
  during reads; scratch and query state belong to the worker/query.
- **Query adapter:** supplies the query, top-k, bounded result heap, exact
  distance computation, exclusion predicate, scan-budget check, and expansion
  callback. The heap exposes `limit/size/clear/emplace`. Exact distance calls
  account for the budget. Excluded nodes can still be traversed.

A backend constructs `QuantizedGraph<Codec>(dimension, codec, max_neighbors)`,
then calls `prebuild(graph, doc_cnt, threads)` and
`search(entry, graph, query_context)`. Prebuild chooses the live node nearest
its corpus mean; search also accepts a caller-selected entry. The degree bound
is rounded down to a codec batch multiple, with a minimum of one batch.
An optional fourth constructor argument, `stored_dimension`, preserves trailing
FP32 metadata in cached centers without using it in centroid selection, pruning,
or encoding. It defaults to `dimension`. The graph and context are borrowed
for each call, not retained by the engine.

[`hnsw_quantized_graph.cc`](hnsw_quantized_graph.cc) is the thin HNSW adapter. It maps
level-zero adjacency and `HnswContext` to those contracts.
The core codec headers are thin adapters to Turbo's block quantizers:
[`SQ8BlockQuantizer`](../../../turbo/quantizer/sq8_block_quantizer/sq8_block_quantizer.h),
[`PQBlockQuantizer`](../../../turbo/quantizer/pq_block_quantizer/pq_block_quantizer.h), and
[`SymphonyQGBlockQuantizer`](../../../turbo/quantizer/symphony_qg_quantizer/symphony_qg_quantizer.h).
Turbo owns transforms, encoding, query LUTs, distance scanning, and SIMD dispatch.
Core retains graph traversal, caches, PQ reservoir sampling and vector loading,
and model publication/retraining on reopen. No on-disk graph format changes.

The formats and arithmetic are preserved: SQ8 stores `[float scale][D int8]`
per lane in batches of 16; PQ uses eight coordinates per subquantizer and PQ4
FastScan batches of 32; SymphonyQG uses node-centered one-bit RaBitQ batches of
32 with a fixed-seed, single-pass, zero-padded normalized Hadamard rotation.
This rotation is separate from Turbo's existing FHT rotator, and SQ8's symmetric
layout is separate from its existing affine Int8Quantizer. SymphonyQG supports
both uint16 and int32 RaBitQ accumulation APIs, using slices of at most 1024
dimensions to avoid overflow with the older API. Its AVX512 translation unit is
compiled by Turbo's per-ISA build rules and retains runtime checks and reference
fallbacks. Cosine norm corrections, zero-vector handling, and exact FP32 result
scoring are unchanged.

## Shared traversal experiment

Ordinary HNSW and `QuantizedGraph` now call the same templated
[`SearchGraph`](../../utility/graph_search.h) loop. It pops candidates, opens
adjacency, scans neighbor batches, and updates the frontier. Policies are
resolved at compile time, with no virtual dispatch in this loop.

- HNSW marks neighbors on discovery and scores compact batches with its existing
  distance calculator. Direct storage retains its prefetch behavior; buffered
  storage pins vector blocks through distance computation. The existing
  LinearPool/BlockHeap and fallback candidate/result heaps remain in use.
- Quantized search marks nodes on expansion, scans complete codec batches, and
  allows a better center-dependent estimate to re-admit an unexpanded node.
  Expanded centers still contribute exact distances to the result heap.
- Each frontier retains its stopping rule and scan-budget behavior. The common
  loop does not compare approximate candidate scores to exact result scores.
  Filtering remains in the result policy so excluded nodes can connect paths.

The HNSW fallback used by upper levels, filtered queries, and construction also
uses this loop with exact distances. Quantized caching, encoding, entry
selection, and result completion remain in `QuantizedGraph`; they are not
required by ordinary HNSW. This is a traversal refactor with no measured
performance claim.

## HNSW search and cache lifecycle

HNSW construction still uses the existing exact distances. After prebuild,
search starts directly at the centroid entry on level zero. After cache
invalidation, HNSW upper-layer navigation selects an entry for lazy search.
The beam capacity is `max(ef, topk)`; estimates prioritize expansion and each
expanded node contributes an exact score in the configured metric. Group-by and brute-force
queries retain the ordinary HNSW paths. The SymphonyQG graph builder is not used.

Raw vectors and adjacency remain in HNSW storage. Opening prebuilds a fixed-
stride arena, while blocks after insertion are rebuilt lazily. The caller must
exclude clear/prebuild and graph mutation from concurrent searches, and clear
the cache before changing vectors, validity, or edges. The HNSW streamer
provides this synchronization and invalidates on insertion/close. Concurrent
queries share immutable cached blocks and use their own query state. There is
no eviction policy. Cached centers retain the stored vectors, including any
auxiliary norm; codes are derived.

The legacy flag is stored in optional HNSW manifest field 7; the new codec
name uses optional field 8. Old manifests default to false and an empty codec.
Existing HNSW graph/vector storage remains compatible. This implementation has no established speedup or
recall improvement; benchmark it on the intended corpus and hardware.

## Cosine distance

Use `HnswIndexParam(metric_type=MetricType.COSINE, symphony_qg=True)`. The
default path uses Turbo `Fp32Quantizer` to normalize vectors and queries;
callers do not need to normalize them. Existing converter-format indexes
retain `CosineFp32Converter` and its reformer when reopened. Stored FP32 cosine vectors append their
original norm after the coordinates. This auxiliary value is retained for the
existing storage contract but excluded from rotation, encoding, pruning, and
centroid selection. A 4096-dimensional cosine vector therefore occupies 4097
stored floats while still using a 4096-dimensional RaBitQ code.

The codec converts the L2 coefficients once when encoding each neighbor:
`f_add_cos = (f_add_l2 + norm2(center) - norm2(neighbor)) / 2` and
`f_rescale_cos = f_rescale_l2 / 2`. With the exact center score `1 - dot(q, c)`,
the scan estimates `1 - dot(q, x)`. The norm correction is necessary for zero
vectors; simply halving L2 is only valid for unit vectors. Original-vector
norms are not used in this formula: these norms are of the normalized, rotated
coordinates. The one-bit codes and two FP32 factors keep the same layout.

Final results use the existing exact cosine metric, including its convention
that a zero vector has distance 1 from every vector, including another zero.
Scores are cosine distances (smaller is better), not similarities or squared
L2 scores. Cache invalidation, filtering, and reopen follow the same HNSW paths.

## Turbo FP32 and legacy storage

New indexes use the ordinary HNSW Turbo FP32 selection for both L2 and cosine
(unless another option, such as converter-side rotation, selects the legacy
pipeline). SymphonyQG reuses the streamer's bound Turbo distance callbacks
for exact scores; its one-bit adjacency codec remains separate.

Legacy cosine metadata counts the norm as a dimension (`D + 1`). Turbo keeps
`dimension = D` and stores the norm in `extra_meta_size = sizeof(float)`.
The streamer resolves coordinate count and full stored-vector size separately,
so the norm is preserved for reconstruction but never quantized. It validates
that the quantizer is FP32 and that its metric and record/query sizes agree
with index metadata.

The existing HNSW reopen dispatch checks persisted metadata before opening the
streamer. An index without a quantizer name retains its legacy pipeline; a
new Turbo index retains `Fp32Quantizer`. Reopening does not rewrite vector
storage or migrate an existing file to the other layout.

## Supported configuration and validation

The SymphonyQG integration supports FP32 squared L2 or cosine, dimensions 1
through 4096,
inline vectors, and a RaBitQ-enabled build with the required SIMD support.
Turbo FP32 is supported; FP16/INT8/INT4 compression and external-vector
storage remain unsupported.
The ordinary HNSW path remains available when the flag is false. These codec
restrictions do not constrain other codecs plugged into `QuantizedGraph`.

Unit tests remain in `symphony_qg_test.cc`. A standalone graph, result adapter,
and scalar int8 codec with batches of three exercise the generic engine without
HNSW or SIMD. They cover multiple/partial batches, filtering and scan budgets,
centroid holes, invalidation, failed prebuild, and concurrent lazy searches.
Shared-loop tests additionally cover exact pool ordering, duplicate discovery,
read failures, partial codec batches, and center-dependent candidate re-entry.
The existing beam, rotation, RaBitQ scanner, and HNSW adapter tests remain;
SIMD runtime cases need a supported machine. Collection/manifest tests continue
to cover insertion, persistence, and reopen through the public API. Cosine
coverage includes auxiliary norm preservation, zero vectors, coefficient
conversion, exact scores, filters, both storage modes, and reopen.

The interface regression also creates and reopens both legacy and Turbo FP32
files, checking pipeline selection, exact scores, vector reconstruction, and
insertion after reopen. Python cosine tests verify reconstructed vectors across
optimize/reopen for mmap, buffer-pool, and contiguous storage configurations.
