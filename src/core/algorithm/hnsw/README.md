# SymphonyQG scanning on ordinary HNSW

Enable this experimental mode with `HnswIndexParam(metric_type=MetricType.L2,
symphony_qg=True)` in Python, `HnswIndexParams::set_symphony_qg(true)` in the
database C++ API, or `HNSWIndexParamBuilder::with_symphony_qg(true)` in the core
API. The low-level streamer parameter is `proxima.hnsw.symphony_qg`.
It defaults to false. The index type and persistent graph format remain HNSW.

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

[`hnsw_symphony_qg.cc`](hnsw_symphony_qg.cc) is the thin HNSW adapter. It maps
level-zero adjacency and `HnswContext` to those contracts.
[`SymphonyQGCodec`](../quantized_graph/symphony_qg_codec.h) implements the first
production codec: node-centered one-bit RaBitQ, batches of 32, deterministic
single-pass normalized Hadamard rotation, and per-query LUT scanning. It
supports both uint16 and int32 RaBitQ accumulation APIs, using slices of at
most 1024 dimensions to avoid overflow with the older API. AVX512 kernels are
codec details with runtime dispatch and portable/reference fallbacks.

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

The flag is stored in optional HNSW manifest field 7; old manifests default to
false. The public parameters, persistence format, and unsupported-platform
behavior are unchanged. This implementation has no established speedup or
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
