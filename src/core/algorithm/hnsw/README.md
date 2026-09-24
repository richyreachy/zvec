# SymphonyQG scanning on ordinary HNSW

Enable this experimental mode with `HnswIndexParam(metric_type=MetricType.L2,
symphony_qg=True)` in Python, `HnswIndexParams::set_symphony_qg(true)` in the
database C++ API, or `HNSWIndexParamBuilder::with_symphony_qg(true)` in the core
API. The low-level streamer parameter is `proxima.hnsw.symphony_qg`.
It defaults to false. The index type remains HNSW.

HNSW construction and upper-layer navigation use the existing exact distances.
Level-zero traversal uses SymphonyQG's node-centered one-bit RaBitQ neighbor
blocks, scanned in batches of 32. A lookup table is prepared once per query;
each expanded node contributes an exact squared-L2 result. The beam size is
`max(ef, topk)`. Filters exclude results but allow traversal. Group-by and
brute-force queries retain the ordinary HNSW paths. This integrates the search
method, not the standalone SymphonyQG iterative graph builder.

The pinned RaBitQ-Library provides `quantize_qg_batch`, `BatchQuery`, and runtime
AVX2/AVX512 `fastscan::accumulate`. Accumulation is split at 1024 dimensions to
avoid 16-bit overflow. Rotation uses three normalized randomized Hadamard passes
with fixed signs and power-of-two zero padding. This preserves L2 distances and
rebuilds identical derived codes after reopen without storing a trained rotator.

## Storage and updates

Raw vectors and graph adjacency remain in the existing HNSW storage. No separate
RaBitQ index, trained clustering, or raw Flat provider is needed. Neighbor blocks
are an immutable in-memory cache, built on first visit and cleared on insertion
or close. In this mode insertions serialize with graph searches; concurrent
queries share blocks and access storage through their own context entities.

Cold queries rotate and quantize adjacency lists. Cache memory per visited node
is approximately `4 * dimension + ceil(degree / 32) * (4 * padded_dimension + 256)`
bytes, plus neighbor IDs and container overhead. There is no eviction policy.
The flag is stored in optional HNSW manifest field 7; old manifests default to
false. The graph format is unchanged. Benchmark cold and warm queries separately;
this implementation has no established speedup or recall improvement.

## Supported configuration

FP32 squared L2, dimensions 1 through 4096, inline vectors, and a RaBitQ-enabled
Linux x86-64 build with AVX2/FMA or AVX512 are required. Extra FP16/INT8/Turbo
quantization and external-vector storage are rejected. The normal HNSW path
remains available on all existing platforms when the flag is false.

Tests cover portable rotation and beam behavior, batch estimation (including
partial blocks, duplicate vectors, and large dimensions), HNSW insertion,
filtering and reopen, manifest compatibility, and Python optimize/reopen with
mmap/buffered and contiguous storage. The SIMD and end-to-end cases need a
supported machine.
