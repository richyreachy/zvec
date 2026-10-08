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
FP32 vectors with squared-L2 scores and fixed-size encoded neighbor batches.
Other metrics and variable-length codes are outside this contract.

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
- **Query adapter:** supplies the query, top-k, bounded result heap, exact L2
  distance computation, exclusion predicate, scan-budget check, and expansion
  callback. The heap exposes `limit/size/clear/emplace`. Exact distance calls
  account for the budget. Excluded nodes can still be traversed.

A backend constructs `QuantizedGraph<Codec>(dimension, codec, max_neighbors)`,
then calls `prebuild(graph, doc_cnt, threads)` and
`search(entry, graph, query_context)`. Prebuild chooses the live node nearest
its corpus mean; search also accepts a caller-selected entry. The degree bound
is rounded down to a codec batch multiple, with a minimum of one batch.
The graph and context are borrowed for each call, not retained by the engine.

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
expanded node contributes an exact squared-L2 score. Group-by and brute-force
queries retain the ordinary HNSW paths. The SymphonyQG graph builder is not used.

Raw vectors and adjacency remain in HNSW storage. Opening prebuilds a fixed-
stride arena, while blocks after insertion are rebuilt lazily. The caller must
exclude clear/prebuild and graph mutation from concurrent searches, and clear
the cache before changing vectors, validity, or edges. The HNSW streamer
provides this synchronization and invalidates on insertion/close. Concurrent
queries share immutable cached blocks and use their own query state. There is
no eviction policy. Cached centers are original vectors; codes are derived.

The flag is stored in optional HNSW manifest field 7; old manifests default to
false. The public parameters, persistence format, and unsupported-platform
behavior are unchanged. This implementation has no established speedup or
recall improvement; benchmark it on the intended corpus and hardware.

## Supported configuration and validation

The SymphonyQG integration requires FP32 squared L2, dimensions 1 through 4096,
inline vectors, and a RaBitQ-enabled build with the required SIMD support.
Extra FP16/INT8/Turbo quantization and external-vector storage are rejected.
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
to cover insertion, persistence, and reopen through the public API.
